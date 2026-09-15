from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


def adaptive_average(x, output_size):
    """Exact adaptive average pooling using separable linear maps (MPS supports all sizes)."""
    def weights(length, bins):
        return x.new_tensor([[1. / (math.ceil((i + 1) * length / bins) - math.floor(i * length / bins))
                              if math.floor(i * length / bins) <= j < math.ceil((i + 1) * length / bins)
                              else 0. for j in range(length)] for i in range(bins)])
    horizontal = F.linear(x, weights(x.shape[-1], output_size[1]))
    return F.linear(horizontal.transpose(-1, -2), weights(x.shape[-2], output_size[0])).transpose(-1, -2)


class Backbone(nn.Module):
    def __init__(self, dim=192, pretrained=True, with_masks=False):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.project16 = nn.Conv2d(256, dim, 1)
        self.project32 = nn.Conv2d(512, dim, 1)
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1),
                                  nn.GroupNorm(8, dim), nn.GELU())
        self.with_masks = with_masks
        if with_masks:
            self.pixel8 = nn.Conv2d(128, dim, 1)
            self.pixel4 = nn.Conv2d(64, dim, 1)
            self.refine8 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
            self.refine4 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x, return_pyramid=False):
        c4 = self.layer1(self.stem(x))
        c8 = self.layer2(c4)
        c16 = self.layer3(c8)
        c32 = self.layer4(c16)
        fused = self.fuse(self.project16(c16) + F.interpolate(
            self.project32(c32), size=c16.shape[-2:], mode="bilinear", align_corners=False))
        if not self.with_masks:
            return fused
        p8 = self.refine8(self.pixel8(c8) + F.interpolate(fused, size=c8.shape[-2:], mode="bilinear", align_corners=False))
        p4 = self.refine4(self.pixel4(c4) + F.interpolate(p8, size=c4.shape[-2:], mode="bilinear", align_corners=False))
        if return_pyramid:
            return {4:p4, 8:p8, 16:fused, 32:self.project32(c32)}
        return fused, p4


def spatial_position(height, width, dim, device, dtype):
    # Fixed spatial encoding, independent of batch composition and padding.
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device=device, dtype=dtype),
                          torch.linspace(0, 1, width, device=device, dtype=dtype), indexing="ij")
    freq = 10000 ** (-torch.arange(dim // 4, device=device, dtype=dtype) / (dim // 4))
    axes = [axis.flatten()[:, None] * (2 * math.pi) * freq for axis in (x, y)]
    return torch.cat([f(v) for v in axes for f in (torch.sin, torch.cos)], dim=-1)[None]


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.current_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q, current, current_pos, current_mask, memory, memory_pos,
                memory_mask, history_valid, variant, visualize, value_position=False):
        q = self.norms[0](q + self.self_attention(q, q, q, need_weights=False)[0])
        q = self.norms[1](q + self.current_attention(
            q, current + current_pos, current + current_pos if value_position else current, key_padding_mask=current_mask, need_weights=False)[0])
        gates = q.new_zeros(*q.shape[:2], 1)
        weights = None
        if variant != "single":
            # Avoid all-masked softmax rows. The dummy's contribution is then explicitly zeroed.
            valid = history_valid.any(1)
            safe_mask = memory_mask.clone()
            safe_mask[~valid, 0] = False
            context, weights = self.temporal_attention(
                q, memory + memory_pos, memory, key_padding_mask=safe_mask,
                need_weights=visualize, average_attn_weights=True)
            context = context * valid[:, None, None]
            gates = torch.sigmoid(self.gate(torch.cat((q, context), -1))) if variant == "gated" else torch.ones_like(gates)
            gates = gates * valid[:, None, None]
            q = q + gates * context
            if weights is not None:
                weights = weights * valid[:, None, None]
        q = self.norms[2](q + self.ffn(q))
        return q, gates, weights


class JALO(nn.Module):
    """Current-first causal clip -> current-frame detections.

    images: [B, 3, 3, H, W]; padding: [B, 3, H, W], True = invalid.
    history_valid: [B, 2]; time_deltas: [B, 2], strictly positive for valid history.
    All variants instantiate identical parameters for reproducible branch initialization.
    """
    def __init__(self, variant="gated", num_classes=8, dim=192, heads=6, layers=3,
                 queries=100, memory_grid=(12, 20), history_seconds=(.2, .4), pretrained=True,
                 task="detection", mask_positions=False, mask_box_prior=False):
        super().__init__()
        if variant not in {"single", "temporal", "gated"}:
            raise ValueError(f"Invalid variant: {variant}")
        if dim % 8 or dim % heads or len(history_seconds) != 2:
            raise ValueError("dim must divide 8 and heads; exactly two history offsets are required")
        self.variant, self.memory_grid = variant, tuple(memory_grid)
        if task not in {"detection", "instance_segmentation"}:
            raise ValueError(f"Unknown task: {task}")
        if task == "instance_segmentation" and variant != "single":
            raise ValueError("Mask training currently supports the single-frame variant only")
        self.task, self.num_classes = task, num_classes
        self.mask_positions = bool(mask_positions)
        self.mask_box_prior = bool(mask_box_prior)
        self.history_seconds = tuple(history_seconds)
        self.dim = dim
        self.backbone = Backbone(dim, pretrained, with_masks=task == "instance_segmentation")
        self.queries = nn.Embedding(queries, dim)
        self.time_position = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.decoder = nn.ModuleList(DecoderBlock(dim, heads) for _ in range(layers))
        self.classifier = nn.Linear(dim, num_classes + 1)
        self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
                                      nn.GELU(), nn.Linear(dim, 4))
        if task == "instance_segmentation":
            self.mask_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, images, padding, history_valid, time_deltas, visualize=False):
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError("Expected [batch, current+2 past frames, RGB, height, width]")
        if torch.any(history_valid & (time_deltas <= 0)):
            raise ValueError("History must be strictly in the past")
        b, t, c, h, w = images.shape
        if self.variant == "single":
            current_features = self.backbone(images[:, 0])
            if self.task == "instance_segmentation":
                current_features, pixels = current_features
            features = current_features[:, None]
        else:
            # Invalid frames never enter CNN/normalization or consume compute.
            keep = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=images.device), history_valid), 1)
            selected = self.backbone(images[keep])
            features = selected.new_zeros(b, t, *selected.shape[1:])
            features[keep] = selected
        fh, fw = features.shape[-2:]
        masks = F.interpolate(padding.float(), size=(fh, fw), mode="nearest").bool()
        current = features[:, 0].flatten(2).transpose(1, 2)
        current_mask = masks[:, 0].flatten(1)
        if current_mask.all(1).any():
            raise ValueError("Current frame contains no valid image pixels")
        pos = spatial_position(fh, fw, self.dim, images.device, images.dtype)
        gh, gw = self.memory_grid
        memory = current.new_zeros(b, 2 * gh * gw, self.dim)
        memory_pos = torch.zeros_like(memory)
        memory_mask = torch.ones(b, 2 * gh * gw, device=images.device, dtype=torch.bool)
        if self.variant != "single":
            past = features[:, 1:].reshape(b * 2, self.dim, fh, fw)
            valid_pixels = (~masks[:, 1:]).reshape(b * 2, 1, fh, fw).float()
            counts = adaptive_average(valid_pixels, self.memory_grid)
            pooled = adaptive_average(past * valid_pixels, self.memory_grid) / counts.clamp(min=1e-6)
            memory = pooled.flatten(2).transpose(1, 2).reshape(b, 2 * gh * gw, self.dim)
            memory_mask = (counts[:, 0] == 0).reshape(b, 2, gh * gw) | ~history_valid[:, :, None]
            memory_mask = memory_mask.flatten(1)
            p = spatial_position(gh, gw, self.dim, images.device, images.dtype)
            tp = self.time_position(time_deltas[:, :, None])[:, :, None, :]
            memory_pos = (p[:, None] + tp).reshape(b, 2 * gh * gw, self.dim)
        q = self.queries.weight[None].expand(b, -1, -1)
        for layer in self.decoder:
            q, gates, attention = layer(q, current, pos, current_mask, memory, memory_pos,
                                        memory_mask, history_valid, self.variant, visualize, self.mask_positions)
        result = {"logits": self.classifier(q), "boxes": self.box_head(q).sigmoid()}
        if self.task == "instance_segmentation":
            if self.mask_positions:
                ph, pw = pixels.shape[-2:]
                pixel_pos = spatial_position(ph, pw, self.dim, pixels.device, pixels.dtype)
                pixels = pixels + pixel_pos.transpose(1, 2).reshape(1, self.dim, ph, pw)
            result["mask_logits"] = torch.einsum("bqd,bdhw->bqhw", self.mask_head(q), pixels) / math.sqrt(self.dim)
            if self.mask_box_prior:
                result["mask_logits"] = result["mask_logits"] + box_mask_prior(result["boxes"], pixels.shape[-2:])
        if visualize:
            result["gates"] = gates.squeeze(-1)
            result["attention"] = (attention.reshape(b, q.shape[1], 2, gh, gw)
                                   if attention is not None else q.new_zeros(b, q.shape[1], 2, gh, gw))
        return result

    def active_parameter_count(self):
        return sum(p.numel() for n, p in self.named_parameters()
                   if not (self.variant == "single" and ("temporal_attention" in n or "gate." in n or "time_position" in n))
                   and not (self.variant == "temporal" and "gate." in n))


def build_model(config, variant, pretrained=None):
    if config.get("architecture") == "vehicle_roi_v2":
        from .roi_model import VehicleROI
        if variant != "single" or config.get("task") != "instance_segmentation":
            raise ValueError("vehicle_roi_v2 requires single-frame instance segmentation")
        model = VehicleROI(pretrained=config.get("pretrained", True) if pretrained is None else pretrained,
                           mask_projection=config.get("mask_projection", "dense"),
                           mask_objective=config.get("mask_objective", "global"),
                           foreground_supervision=config.get("foreground_supervision", "feature"),
                           **config.get("model", {}))
        if "classes" in config and len(config["classes"]) != model.num_classes:
            raise ValueError("Class list length differs from model.num_classes")
        return model
    if config.get("architecture", "jalo_v1") != "jalo_v1":
        raise ValueError("Unknown model architecture")
    model = JALO(variant=variant, pretrained=config.get("pretrained", True) if pretrained is None else pretrained,
                 **config.get("model", {}))
    if config.get("task", "detection") != model.task:
        raise ValueError("Configuration task differs from model.task")
    if "classes" in config and len(config["classes"]) != model.num_classes:
        raise ValueError("Class list length differs from model.num_classes")
    return model


def box_mask_prior(boxes, size, strength=10.):
    """Soft, differentiable spatial support; does not generate or fill a mask.

    Zero inside a predicted box; suppresses pixel logits progressively outside it.
    Both boxes and pixel appearance still receive the genuine mask loss gradient.
    """
    h, w = size
    x = (torch.arange(w, device=boxes.device, dtype=boxes.dtype) + .5) / w
    y = (torch.arange(h, device=boxes.device, dtype=boxes.dtype) + .5) / h
    cx, cy, bw, bh = boxes.unbind(-1)
    dx = (x[None, None, None, :] - cx[..., None, None]).abs() / (bw[..., None, None] * .5).clamp(min=1e-3)
    dy = (y[None, None, :, None] - cy[..., None, None]).abs() / (bh[..., None, None] * .5).clamp(min=1e-3)
    return -strength * torch.relu(torch.maximum(dx, dy) - 1)
