import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .geometry import cxcywh_to_xyxy, generalized_iou


class SetCriterion(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        weights = torch.ones(num_classes + 1)
        weights[-1] = .1
        self.register_buffer("class_weights", weights)
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        logits, boxes = outputs["logits"], outputs["boxes"]
        labels = torch.full(logits.shape[:2], self.num_classes, device=logits.device, dtype=torch.long)
        predicted, actual = [], []
        mask_bce, mask_dice = [], []
        for index, target in enumerate(targets):
            gt_boxes = target["boxes"].to(boxes.device)
            gt_labels = target["labels"].to(boxes.device)
            ignore = target.get("ignore_model_boxes")
            if ignore is not None and len(ignore):
                with torch.no_grad():
                    pb = cxcywh_to_xyxy(boxes[index])
                    ib = cxcywh_to_xyxy(ignore.to(boxes.device))
                    lo = torch.maximum(pb[:, None, :2], ib[None, :, :2])
                    hi = torch.minimum(pb[:, None, 2:], ib[None, :, 2:])
                    intersection = (hi - lo).clamp(min=0).prod(-1)
                    area = (pb[:, 2:] - pb[:, :2]).clamp(min=0).prod(-1).clamp(min=1e-8)
                    labels[index, (intersection / area[:, None]).amax(1) > .5] = -100
            if len(gt_labels) > boxes.shape[1]:
                raise ValueError("More ground-truth objects than queries; increase model.queries")
            if len(gt_labels) == 0:
                continue
            with torch.no_grad():
                # Matching on CPU avoids backend-specific assignment/cdist limitations.
                cost = (-logits[index].softmax(-1)[:, gt_labels]).cpu()
                pb, gb = boxes[index].detach().cpu(), gt_boxes.cpu()
                cost += 5 * torch.cdist(pb, gb, p=1)
                cost -= 2 * generalized_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb))
                if "mask_logits" in outputs:
                    pm, gm, valid = mask_tensors(outputs["mask_logits"][index], target)
                    cost += 2 * pair_mask_cost(pm, gm, valid).cpu()
                rows, cols = linear_sum_assignment(cost.numpy())
                rows = torch.as_tensor(rows, device=boxes.device)
                cols = torch.as_tensor(cols, device=boxes.device)
            labels[index, rows] = gt_labels[cols]
            predicted.append(boxes[index, rows])
            actual.append(gt_boxes[cols])
            if "mask_logits" in outputs:
                if outputs.get('mask_objective')=='local_roi':
                    bce,dice=local_roi_mask_losses(outputs['roi_logits'][index,rows],boxes[index,rows],target,cols)
                else:
                    pm, gm, valid = mask_tensors(outputs["mask_logits"][index], target)
                    bce, dice = mask_losses(pm[rows], gm[cols], valid)
                mask_bce.extend(bce.unbind())
                mask_dice.extend(dice.unbind())
        ce = (F.cross_entropy(logits.transpose(1, 2), labels, weight=self.class_weights)
              if (labels != -100).any() else logits.sum() * 0)
        if predicted:
            pb, gb = torch.cat(predicted), torch.cat(actual)
            l1 = F.l1_loss(pb, gb, reduction="sum") / len(pb)
            giou = (1 - generalized_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)).diag()).mean()
        else:
            l1 = giou = boxes.sum() * 0
        result = {"total": ce + 5 * l1 + 2 * giou, "classification": ce, "l1": l1, "giou": giou}
        if "mask_logits" in outputs:
            zero = outputs["mask_logits"].sum() * 0
            result["mask_bce"] = torch.stack(mask_bce).mean() if mask_bce else zero
            result["mask_dice"] = torch.stack(mask_dice).mean() if mask_dice else zero
            result["total"] = result["total"] + 2 * result["mask_bce"] + 2 * result["mask_dice"]
        return result


def local_roi_mask_losses(logits,boxes,target,matched_indices):
    """Supervise all local pixels of each matched predicted ROI, excluding ignore.

    The target crop is detached from box regression. Box L1/GIoU and auxiliary
    detection losses still supervise coverage outside a too-small predicted ROI.
    """
    from .roi_model import roi_grid
    grid=roi_grid(boxes.detach(),logits.shape[-1])
    truth=target['masks'].to(logits.device).float()[matched_indices]
    truth=F.grid_sample(truth[:,None],grid,mode='bilinear',align_corners=False)[:,0]
    valid=target['mask_valid'].to(logits.device).float()[None,None].expand(len(logits),-1,-1,-1)
    valid=(F.grid_sample(valid,grid,mode='bilinear',align_corners=False)[:,0]>.999).to(logits.dtype)
    count=valid.flatten(1).sum(1).clamp(min=1)
    bce=(F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*valid).flatten(1).sum(1)/count
    probability,truth=(logits.sigmoid()*valid).flatten(1),(truth*valid).flatten(1)
    dice=1-(2*(probability*truth).sum(1)+1)/(probability.sum(1)+truth.sum(1)+1)
    return bce,dice


def mask_tensors(logits, target):
    size = logits.shape[-2:]
    masks = target["masks"].to(logits.device).float()
    masks = F.interpolate(masks[:, None], size=size, mode="nearest")[:, 0] if len(masks) else logits.new_empty((0, *size))
    # A stride-4 cell touching ignore/padding must not contribute to the objective.
    invalid = (~target["mask_valid"]).to(logits.device).float()[None, None]
    valid = F.interpolate(invalid, size=size, mode="area")[0, 0] == 0
    return logits, masks, valid


def mask_losses(logits, targets, valid):
    valid = valid.to(logits.dtype)
    count = valid.sum().clamp(min=1)
    bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid).flatten(1).sum(1) / count
    prob, truth = (logits.sigmoid() * valid).flatten(1), (targets * valid).flatten(1)
    dice = 1 - (2 * (prob * truth).sum(1) + 1) / (prob.sum(1) + truth.sum(1) + 1)
    return bce, dice


def pair_mask_cost(logits, targets, valid, max_points=2048):
    # Deterministic shared points keep assignment memory bounded without affecting RNG.
    indices = valid.flatten().nonzero().flatten()
    if not len(indices):
        return logits.new_zeros((len(logits), len(targets)))
    if len(indices) > max_points:
        indices = indices[torch.linspace(0, len(indices)-1, max_points, device=indices.device).long()]
    x, y = logits.flatten(1)[:, indices], targets.flatten(1)[:, indices]
    bce = F.softplus(x).mean(1)[:, None] - (x @ y.T) / len(indices)
    prob = x.sigmoid()
    dice = 1 - (2 * (prob @ y.T) + 1) / (prob.sum(1)[:, None] + y.sum(1)[None] + 1)
    return bce + dice
