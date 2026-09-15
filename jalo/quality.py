"""Painting quality against explicit reference polygons, never inferred pseudo-GT."""
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class VehicleQuality:
    def __init__(self, classes=('car', 'truck', 'bus'), pipeline_stage='single-frame confidence-ordered masks before temporal tracker'):
        self.classes=classes;self.pipeline_stage=pipeline_stage
        self.frames=self.painted=self.correct=self.cabin_painted=self.cabin=0
        self.background=self.background_painted=0
        self.targets=self.hits=self.small_targets=self.small_hits=self.predictions=0
        self.by_class=defaultdict(lambda: {'count_ge32':0,'hits_ge32':0,'small_count':0,'small_hits':0})

    def update(self,prediction,target):
        transform=target['transform']
        ignore=np.asarray(target.get('original_ignore_mask',np.zeros((transform.original_h,transform.original_w),bool)),bool)
        valid=~ignore;shape=valid.shape
        union=np.zeros(shape,bool)
        gt=[_numpy(m).astype(bool)&valid for m in target['original_masks']]
        gt_labels=_numpy(target['labels'])
        gt_boxes=_numpy(target['original_boxes'])
        usable=np.array([m.any() for m in gt],dtype=bool)
        gt=[m for m,keep in zip(gt,usable) if keep]
        gt_labels=gt_labels[usable];gt_boxes=gt_boxes[usable]
        for m in gt:union |= m
        masks=[];labels=[];occupied=np.zeros(shape,bool)
        order=np.argsort(-_numpy(prediction['scores']),kind='stable')
        prediction_labels=_numpy(prediction['labels'])
        for i in order:
            raw=_numpy(prediction['masks'][i]).astype(bool)
            if raw.shape!=shape:raise ValueError('Quality masks must be in original image coordinates')
            visible=raw & ~occupied & valid
            if visible.any():masks.append(visible);labels.append(prediction_labels[i])
            occupied |= raw
        painted=occupied & valid
        cabin=np.asarray(target.get('interior_mask',np.zeros(shape,bool)),bool) & valid
        background=valid & ~union
        self.frames+=1;self.painted+=int(painted.sum());self.correct+=int((painted&union).sum())
        self.background+=int(background.sum());self.background_painted+=int((painted&background).sum())
        self.cabin+=int(cabin.sum());self.cabin_painted+=int((painted&cabin).sum());self.predictions+=len(masks)
        hits=np.zeros(len(gt),bool)
        if masks and gt:
            ious=np.array([[(a&b).sum()/max((a|b).sum(),1) for b in gt] for a in masks])
            eligible=(ious>=.5)&(np.asarray(labels)[:,None]==gt_labels[None])
            # Maximum cardinality first, IoU second: a merged prediction cannot hit two cars.
            rows,cols=linear_sum_assignment(np.where(eligible,-(len(gt)+1)-ious,0))
            hits[cols[eligible[rows,cols]]]=True
        for hit,box,label in zip(hits,gt_boxes,gt_labels):
            counts=self.by_class[self.classes[int(label)]]
            if max(box[2]-box[0],box[3]-box[1])>=32:
                self.targets+=1;self.hits+=int(hit);counts['count_ge32']+=1;counts['hits_ge32']+=int(hit)
            else:
                self.small_targets+=1;self.small_hits+=int(hit);counts['small_count']+=1;counts['small_hits']+=int(hit)

    def compute(self):
        precision=self.correct/self.painted if self.painted else None
        recall=self.hits/self.targets if self.targets else None
        cabin=self.cabin_painted/self.cabin if self.cabin else None
        background=self.background_painted/self.background if self.background else None
        passed=(precision is not None and recall is not None and background is not None and
                precision>=.9 and recall>=.8 and background<=.005 and (cabin is None or cabin<=.005))
        per_class={}
        for name in self.classes:
            counts=dict(self.by_class[name])
            per_class[name]={**counts,'recall_ge32':counts['hits_ge32']/counts['count_ge32'] if counts['count_ge32'] else None,
                             'small_recall':counts['small_hits']/counts['small_count'] if counts['small_count'] else None}
        return {'frames':self.frames,'painted_pixels':self.painted,'correct_painted_pixels':self.correct,
            'pixel_precision':precision,'vehicle_count_ge32':self.targets,'vehicle_hits_ge32':self.hits,'vehicle_recall_ge32':recall,
            'small_vehicle_count':self.small_targets,'small_vehicle_hits':self.small_hits,
            'small_vehicle_recall':self.small_hits/self.small_targets if self.small_targets else None,
            'background_pixels':self.background,'background_painted_pixels':self.background_painted,'background_false_paint_rate':background,
            'interior_pixels':self.cabin,'interior_painted_pixels':self.cabin_painted,'interior_false_paint_rate':cabin,
            'prediction_count':self.predictions,'quality_pass':bool(passed),'per_class':per_class,
            'reference_type':'AI-created, visually reviewed polygons','pipeline_stage':self.pipeline_stage}


def calibration_rank(metrics, mask_ap=0.):
    p=metrics['pixel_precision'] or 0.;r=metrics['vehicle_recall_ge32'] or 0.
    c=metrics['interior_false_paint_rate'];b=metrics.get('background_false_paint_rate')
    constrained=p>=.9 and b is not None and b<=.005 and (c is None or c<=.005) and metrics['painted_pixels']>0
    f1=2*p*r/max(p+r,1e-9)
    return (int(constrained),r if constrained else f1,mask_ap or 0.)
