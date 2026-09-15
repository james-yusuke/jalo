from __future__ import annotations

import contextlib
import io

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .data import CLASSES
from .geometry import box_iou


def decode(outputs, transforms, threshold=0., mask_threshold=.5, foreground_only=False):
    predictions = []
    for i, transform in enumerate(transforms):
        probabilities = outputs["logits"][i].detach().cpu().softmax(-1)
        scores, classes = probabilities[:, :-1].max(-1)
        boxes = transform.boxes_to_original(outputs["boxes"][i].detach().cpu())
        # Background-dominant queries are retained for AP ranking, but their foreground score is low.
        keep = (scores >= threshold) & ((boxes[:, 2:] - boxes[:, :2]) > 0).all(1)
        if foreground_only:
            keep &= probabilities.argmax(-1) < probabilities.shape[-1] - 1
        predictions.append({"boxes": boxes[keep], "scores": scores[keep], "labels": classes[keep],
                            "query_indices": torch.arange(len(boxes))[keep]})
        if "mask_logits" in outputs:
            predictions[-1]["masks"] = restore_masks(outputs,i,predictions[-1]['query_indices'],transform,mask_threshold)
    return predictions


def restore_masks(outputs, index, query_indices, transform, threshold=.5):
    if outputs.get('mask_projection')=='native_roi':
        masks=transform.roi_masks_to_original(outputs['roi_logits'][index,query_indices],
                                              outputs['boxes'][index,query_indices],threshold)
    else:masks=transform.masks_to_original(outputs['mask_logits'][index,query_indices],threshold)
    if 'foreground_logits' in outputs:
        masks &= transform.masks_to_original(outputs['foreground_logits'][index],.5)[0][None]
    return masks


def match_success(prediction, target, threshold=.3):
    valid = prediction["scores"] >= threshold
    boxes, labels = prediction["boxes"][valid], prediction["labels"][valid]
    gt_boxes, gt_labels = target["original_boxes"], target["labels"]
    success = np.zeros(len(gt_boxes), dtype=bool)
    if len(boxes) and len(gt_boxes):
        iou = box_iou(boxes, gt_boxes)[0].numpy()
        compatible = (labels[:, None] == gt_labels[None, :]).numpy() & (iou >= .5)
        costs = np.where(compatible, -100. - iou, 0.)
        rows, cols = linear_sum_assignment(costs)
        success[cols[compatible[rows, cols]]] = True
    return success


class DetectionMetrics:
    def __init__(self, classes=CLASSES):
        self.classes = tuple(classes)
        self.images, self.annotations, self.predictions = [], [], []
        self.track_observations = {}
        self.occluded_count = self.occluded_hits = self.gt_count = self.pred_count = 0

    def update(self, prediction, target):
        image_id = target["image_id"]
        tr = target["transform"]
        self.images.append({"id": image_id, "height": tr.original_h, "width": tr.original_w})
        for box, label in zip(target["original_boxes"].tolist(), target["labels"].tolist()):
            self._annotation(image_id, box, label, False)
            self.gt_count += 1
        for box in target.get("ignore_boxes", torch.empty(0, 4)).tolist():
            for label in range(len(self.classes)):
                self._annotation(image_id, box, label, True)
        for box, score, label in zip(prediction["boxes"].tolist(), prediction["scores"].tolist(), prediction["labels"].tolist()):
            x1, y1, x2, y2 = box
            self.predictions.append({"image_id": image_id, "category_id": label + 1,
                                     "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            self.pred_count += 1
        success = match_success(prediction, target)
        for i, track in enumerate(target["track_ids"]):
            key = (target["video"], track, int(target["labels"][i]))
            self.track_observations.setdefault(key, []).append((target["frame_index"], bool(success[i])))
            if target["occluded"][i]:
                self.occluded_count += 1
                self.occluded_hits += int(success[i])

    def _annotation(self, image_id, box, label, crowd):
        x1, y1, x2, y2 = box
        self.annotations.append({"id": len(self.annotations) + 1, "image_id": image_id,
                                 "category_id": label + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                                 "area": (x2 - x1) * (y2 - y1), "iscrowd": int(crowd)})

    def compute(self):
        result = {"frames": len(self.images), "ground_truth_count": self.gt_count,
                  "prediction_count": self.pred_count, "mAP": None, "AP50": None, "AP_small": None}
        if self.gt_count:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            with contextlib.redirect_stdout(io.StringIO()):
                gt = COCO()
                gt.dataset = {"images": self.images, "annotations": self.annotations,
                              "categories": [{"id": i + 1, "name": name} for i, name in enumerate(self.classes)], "info": {}}
                gt.createIndex()
                if self.predictions:
                    dt = gt.loadRes(self.predictions)
                else:
                    dt = COCO()
                    dt.dataset = {**gt.dataset, "annotations": []}
                    dt.createIndex()
                evaluator = COCOeval(gt, dt, "bbox")
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            for name, index in (("mAP", 0), ("AP50", 1), ("AP_small", 3)):
                value = float(evaluator.stats[index])
                result[name] = value if value >= 0 else None
        transitions = switches = 0
        for observations in self.track_observations.values():
            observations.sort()
            for (previous_index, previous_hit), (current_index, current_hit) in zip(observations, observations[1:]):
                if current_index == previous_index + 1:
                    transitions += 1
                    switches += previous_hit != current_hit
        result.update(occluded_count=self.occluded_count, occluded_hits=self.occluded_hits,
                      occluded_recall=self.occluded_hits / self.occluded_count if self.occluded_count else None,
                      track_transition_count=transitions, detection_switch_count=switches,
                      detection_switch_rate=switches / transitions if transitions else None)
        return result


def mask_rle(mask):
    from pycocotools import mask as mask_api
    rle = mask_api.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


class InstanceMetrics(DetectionMetrics):
    def __init__(self, classes):
        super().__init__(classes)
        self.mask_annotations, self.mask_predictions = [], []

    def update(self, prediction, target):
        from pycocotools import mask as mask_api
        start = len(self.annotations)
        super().update(prediction, {**target, "ignore_boxes": torch.empty(0, 4)})
        # COCO uses annotated instance area (not rectangle area) for size strata.
        areas = target.get("original_areas", [int(np.asarray(m).sum()) for m in target["original_masks"]])
        for annotation, area in zip(self.annotations[start:], areas):
            annotation["area"] = area
        for crowd_index, (mask, label) in enumerate(target.get("crowd_masks", [])):
            x, y, w, h = mask_api.toBbox(mask_rle(mask)).tolist()
            self._annotation(target["image_id"], [x, y, x+w, y+h], label, True)
            self.annotations[-1]["area"] = target.get("crowd_areas", [int(np.asarray(m).sum()) for m, _ in target.get("crowd_masks", [])])[crowd_index]
        for mask, label, area in zip(target["original_masks"], target["labels"].tolist(), areas):
            self._mask_annotation(mask, label, target["image_id"], False, area)
        for crowd_index, (mask, label) in enumerate(target.get("crowd_masks", [])):
            self._mask_annotation(mask, label, target["image_id"], True, target.get("crowd_areas", [None]*len(target["crowd_masks"]))[crowd_index])
        for mask, score, label in zip(prediction["masks"], prediction["scores"].tolist(), prediction["labels"].tolist()):
            self.mask_predictions.append({"image_id": target["image_id"], "category_id": label+1,
                                          "segmentation": mask_rle(mask), "score": score})

    def _mask_annotation(self, mask, label, image_id, crowd, area=None):
        from pycocotools import mask as mask_api
        rle = mask_rle(mask)
        self.mask_annotations.append({"id": len(self.mask_annotations)+1, "image_id": image_id,
            "category_id": label+1, "segmentation": rle, "area": float(mask_api.area(rle)) if area is None else area,
            "bbox": mask_api.toBbox(rle).tolist(), "iscrowd": int(crowd)})

    def compute(self):
        result = super().compute()
        result.update(mask_mAP=None, mask_AP50=None, mask_AP_small=None)
        if not self.gt_count:
            return result
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = {"images": self.images, "annotations": self.mask_annotations,
                "categories": [{"id": i+1, "name": n} for i,n in enumerate(self.classes)], "info": {}}
            gt.createIndex()
            if self.mask_predictions:
                dt = gt.loadRes(self.mask_predictions)
            else:
                dt = COCO()
                dt.dataset = {**gt.dataset, "annotations": []}
                dt.createIndex()
            ev = COCOeval(gt, dt, "segm")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        for name, index in (("mask_mAP",0), ("mask_AP50",1), ("mask_AP_small",3)):
            result[name] = float(ev.stats[index]) if ev.stats[index] >= 0 else None
        return result
