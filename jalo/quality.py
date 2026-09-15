"""Painting quality against explicit reference polygons, never inferred pseudo-GT."""
import numpy as np
from scipy.optimize import linear_sum_assignment


class VehicleQuality:
    def __init__(self):
        self.frames=self.painted=self.correct=self.cabin_painted=self.cabin=0
        self.targets=self.hits=self.small_targets=self.small_hits=self.predictions=0

    def update(self,prediction,target):
        ignore=np.asarray(target.get('original_ignore_mask',np.zeros((target['transform'].original_h,target['transform'].original_w),bool)))
        valid=~ignore;shape=valid.shape
        union=np.zeros(shape,bool)
        gt=[np.asarray(m,bool)&valid for m in target['original_masks']]
        for m in gt:union |= m
        masks=[];occupied=np.zeros(shape,bool)
        order=np.argsort(-prediction['scores'].numpy(),kind='stable')
        for i in order:
            raw=prediction['masks'][i].numpy().astype(bool)
            visible=raw & ~occupied & valid
            if visible.any():masks.append(visible)
            occupied |= raw
        painted=occupied & valid
        cabin=np.asarray(target.get('interior_mask',np.zeros(shape,bool))) & valid
        self.frames+=1;self.painted+=int(painted.sum());self.correct+=int((painted&union).sum())
        self.cabin+=int(cabin.sum());self.cabin_painted+=int((painted&cabin).sum());self.predictions+=len(masks)
        hits=np.zeros(len(gt),bool)
        if masks and gt:
            ious=np.array([[(a&b).sum()/max((a|b).sum(),1) for b in gt] for a in masks])
            eligible=ious>=.5;rows,cols=linear_sum_assignment(np.where(eligible,-100-ious,0))
            hits[cols[eligible[rows,cols]]]=True
        for hit,box in zip(hits,target['original_boxes'].numpy()):
            if max(box[2]-box[0],box[3]-box[1])>=32:self.targets+=1;self.hits+=int(hit)
            else:self.small_targets+=1;self.small_hits+=int(hit)

    def compute(self):
        precision=self.correct/self.painted if self.painted else None
        recall=self.hits/self.targets if self.targets else None
        cabin=self.cabin_painted/self.cabin if self.cabin else None
        passed=(precision is not None and recall is not None and cabin is not None and precision>=.9 and recall>=.8 and cabin<=.005)
        return {'frames':self.frames,'painted_pixels':self.painted,'correct_painted_pixels':self.correct,
            'pixel_precision':precision,'vehicle_count_ge32':self.targets,'vehicle_hits_ge32':self.hits,'vehicle_recall_ge32':recall,
            'small_vehicle_count':self.small_targets,'small_vehicle_hits':self.small_hits,
            'small_vehicle_recall':self.small_hits/self.small_targets if self.small_targets else None,
            'interior_pixels':self.cabin,'interior_painted_pixels':self.cabin_painted,'interior_false_paint_rate':cabin,
            'prediction_count':self.predictions,'quality_pass':bool(passed),'reference_type':'AI-created, visually reviewed polygons',
            'pipeline_stage':'single-frame confidence-ordered masks before temporal tracker'}


def calibration_rank(metrics, mask_ap=0.):
    p=metrics['pixel_precision'] or 0.;r=metrics['vehicle_recall_ge32'] or 0.
    c=metrics['interior_false_paint_rate'];constrained=p>=.9 and c is not None and c<=.005 and metrics['painted_pixels']>0
    f1=2*p*r/max(p+r,1e-9)
    return (int(constrained),r if constrained else f1,mask_ap or 0.)
