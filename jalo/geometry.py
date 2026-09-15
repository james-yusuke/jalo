from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


def cxcywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_cxcywh(boxes):
    return torch.cat(((boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]), dim=-1)


def box_iou(a, b):
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    lo = torch.maximum(a[:, None, :2], b[None, :, :2])
    hi = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (hi - lo).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b[None] - intersection
    return intersection / union.clamp(min=1e-8), union


def generalized_iou(a, b):
    iou, union = box_iou(a, b)
    lo = torch.minimum(a[:, None, :2], b[None, :, :2])
    hi = torch.maximum(a[:, None, 2:], b[None, :, 2:])
    enclosing = (hi - lo).clamp(min=0).prod(-1).clamp(min=1e-8)
    return iou - (enclosing - union) / enclosing


@dataclass
class Letterbox:
    original_h: int
    original_w: int
    height: int
    width: int
    resized_h: int
    resized_w: int

    @classmethod
    def create(cls, image, size):
        h, w = image.shape[:2]
        scale = min(size[0] / h, size[1] / w)
        return cls(h, w, *size, max(1, round(h * scale)), max(1, round(w * scale)))

    def image(self, rgb):
        resized = np.asarray(Image.fromarray(rgb).resize(
            (self.resized_w, self.resized_h), Image.Resampling.BILINEAR)).copy()
        out = torch.zeros(3, self.height, self.width)
        normalized = torch.from_numpy(resized).permute(2, 0, 1).float() / 255
        normalized = (normalized - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor(
            [.229, .224, .225])[:, None, None]
        out[:, :self.resized_h, :self.resized_w] = normalized
        padding = torch.ones(self.height, self.width, dtype=torch.bool)
        padding[:self.resized_h, :self.resized_w] = False
        return out, padding

    def boxes_to_model(self, xyxy):
        scale = xyxy.new_tensor([self.resized_w / self.original_w,
                                self.resized_h / self.original_h] * 2)
        size = xyxy.new_tensor([self.width, self.height] * 2)
        return xyxy_to_cxcywh(xyxy * scale / size)

    def boxes_to_original(self, cxcywh):
        scale = cxcywh.new_tensor([self.width * self.original_w / self.resized_w,
                                  self.height * self.original_h / self.resized_h] * 2)
        boxes = cxcywh_to_xyxy(cxcywh) * scale
        boxes[:, 0::2].clamp_(0, self.original_w)
        boxes[:, 1::2].clamp_(0, self.original_h)
        return boxes

    def masks_to_original(self, logits, threshold=.5):
        """Upsample logits, remove top-left letterbox padding, then restore source pixels."""
        from torch.nn import functional as F
        if len(logits) == 0:
            return torch.empty((0, self.original_h, self.original_w), dtype=torch.bool)
        restored = []
        for chunk in logits.detach().cpu().split(8):
            full = F.interpolate(chunk[:, None], size=(self.height, self.width), mode="bilinear", align_corners=False)
            crop = full[:, :, :self.resized_h, :self.resized_w]
            original = F.interpolate(crop, size=(self.original_h, self.original_w), mode="bilinear", align_corners=False)
            restored.append(original[:, 0].sigmoid() >= threshold)
        return torch.cat(restored)

    def roi_masks_to_original(self, logits, boxes, threshold=.5):
        """Sample local mask logits directly at source-pixel centers.

        Keep the unclipped ROI for coordinate mapping, then restrict work to its
        visible image intersection. No intermediate stride-4 canvas discards
        the 56x56 mask's fine detail, and no box is used as a replacement mask.
        """
        import math
        from torch.nn import functional as F
        logits,boxes=logits.detach().cpu(),boxes.detach().cpu()
        if len(logits)!=len(boxes):raise ValueError('One local mask is required per ROI')
        if not 0 < threshold < 1:raise ValueError('Mask threshold must be between zero and one')
        result=torch.zeros(len(logits),self.original_h,self.original_w,dtype=torch.bool)
        scale=boxes.new_tensor([self.width*self.original_w/self.resized_w,
                                self.height*self.original_h/self.resized_h]*2)
        original=cxcywh_to_xyxy(boxes)*scale
        cutoff=math.log(threshold/(1-threshold))
        for i,(mask,box) in enumerate(zip(logits,original)):
            left,top,right,bottom=box.tolist()
            if not all(math.isfinite(v) for v in (left,top,right,bottom)):
                raise ValueError('Nonfinite ROI coordinates')
            x0,y0=max(0,math.floor(left)),max(0,math.floor(top))
            x1,y1=min(self.original_w,math.ceil(right)),min(self.original_h,math.ceil(bottom))
            if right<=left or bottom<=top or x1<=x0 or y1<=y0:continue
            gx=((torch.arange(x0,x1,dtype=mask.dtype)+.5)-left)/(right-left)*2-1
            gy=((torch.arange(y0,y1,dtype=mask.dtype)+.5)-top)/(bottom-top)*2-1
            yy,xx=torch.meshgrid(gy,gx,indexing='ij')
            grid=torch.stack((xx,yy),-1)[None]
            sampled=F.grid_sample(mask[None,None],grid,mode='bilinear',
                                  padding_mode='border',align_corners=False)[0,0]
            result[i,y0:y1,x0:x1]=(sampled>=cutoff) & (xx.abs()<=1) & (yy.abs()<=1)
        return result
