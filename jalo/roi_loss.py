"""Dense center supervision, ROI masks, foreground negatives, and set prediction."""
import torch
from torch import nn
from torch.nn import functional as F
from .loss import SetCriterion, mask_losses
from .roi_model import roi_grid


def dense_targets(outputs, targets):
    logits=outputs['center_logits'];b,c,h,w=logits.shape
    heat=torch.zeros_like(logits);geometry=[];indices=[]
    valid=~outputs['dense_padding'];unions=[]
    yy,xx=torch.meshgrid(torch.arange(h,device=logits.device),torch.arange(w,device=logits.device),indexing='ij')
    for batch,t in enumerate(targets):
        invalid=F.interpolate((~t['mask_valid']).float()[None,None].to(logits.device),size=(h,w),mode='area')[0,0]>0
        valid[batch] &= ~invalid
        masks=t['masks'].to(logits.device).float()
        union=masks.amax(0) if len(masks) else logits.new_zeros(t['mask_valid'].shape)
        unions.append(F.interpolate(union[None,None],size=(h,w),mode='nearest')[0,0])
        occupied=set()
        # Regression collisions are deterministic; larger objects retain their center.
        boxes=t['boxes'].to(logits.device);labels=t['labels'].tolist()
        order=sorted(range(len(boxes)),key=lambda j:float(boxes[j,2:].prod()),reverse=True)
        for j in order:
            box=boxes[j];cx=float(box[0])*w;cy=float(box[1])*h
            x=min(w-1,max(0,int(cx)));y=min(h-1,max(0,int(cy)))
            radius=max(1.,min(float(box[2])*w,float(box[3])*h)/6.)
            gaussian=torch.exp(-((xx-x)**2+(yy-y)**2)/(2*radius**2))
            heat[batch,labels[j]]=torch.maximum(heat[batch,labels[j]],gaussian)
            if (y,x) not in occupied and valid[batch,y,x]:
                occupied.add((y,x));indices.append((batch,y,x))
                geometry.append(torch.stack((box[0]*w-x,box[1]*h-y,box[2],box[3])))
    return heat,valid,torch.stack(unions),indices,geometry


class ROICriterion(nn.Module):
    def __init__(self,classes=3):
        super().__init__();self.set_loss=SetCriterion(classes)

    def forward(self,outputs,targets,phase='joint'):
        heat,valid,union,indices,geometry=dense_targets(outputs,targets)
        prob=outputs['center_logits'].sigmoid().clamp(1e-5,1-1e-5)
        positive=(heat==1).to(prob.dtype);negative=(1-positive)*(1-heat)**4
        center=(-positive*(1-prob)**2*prob.log()-negative*prob**2*(1-prob).log())*valid[:,None]
        center=center.sum()/positive.mul(valid[:,None]).sum().clamp(min=1)
        if indices:
            chosen=torch.stack([outputs['proposal_geometry'][b,:,y,x] for b,y,x in indices])
            dense_box=F.l1_loss(chosen,torch.stack(geometry),reduction='sum')/len(chosen)
        else:dense_box=outputs['proposal_geometry'].sum()*0
        fg_bce,fg_dice=foreground_losses(outputs['foreground_logits'],targets,union,valid,
                                         outputs.get('foreground_supervision','feature'))
        if phase=='joint':
            result=self.set_loss(outputs,targets)
            auxiliary=outputs['logits'].sum()*0
            for aux in outputs['aux_outputs']:auxiliary=auxiliary+self.set_loss(aux,targets)['total']
            result['auxiliary']=auxiliary;result['total']=result['total']+auxiliary
        else:
            zero=outputs['center_logits'].sum()*0
            result={'total':zero,'classification':zero,'l1':zero,'giou':zero,'mask_bce':zero,'mask_dice':zero}
            bs,ds=[],[]
            for logits,t in zip(outputs['training_roi_logits'],targets):
                if not len(logits):continue
                boxes=t['boxes'].to(logits.device)
                grid=roi_grid(boxes,logits.shape[-1])
                truth=F.grid_sample(t['masks'][:,None].float().to(logits.device),grid,align_corners=False)[:,0]
                valid_roi=F.grid_sample(t['mask_valid'][None,None].float().to(logits.device).expand(len(boxes),-1,-1,-1),grid,
                                      align_corners=False)[:,0]>.999
                # Every ROI has its own valid mask; apply the shared loss independently.
                for pm,gm,vm in zip(logits,truth,valid_roi):
                    a,b=mask_losses(pm[None],gm[None],vm);bs.append(a[0]);ds.append(b[0])
            if bs:
                result['mask_bce']=torch.stack(bs).mean();result['mask_dice']=torch.stack(ds).mean()
                result['total']=2*(result['mask_bce']+result['mask_dice'])
        result.update(center_focal=center,dense_box=dense_box,foreground_bce=fg_bce,foreground_dice=fg_dice)
        result['total']=result['total']+center+5*dense_box+fg_bce+fg_dice
        return result


def foreground_losses(logits,targets,union,valid,supervision='feature'):
    if supervision=='input':
        # Supervise the same pixel-centered interpolation used for rendering.
        # A thin positive region must not disappear through label subsampling.
        size=targets[0]['mask_valid'].shape
        logits=F.interpolate(logits,size=size,mode='bilinear',align_corners=False)
        valid=torch.stack([t['mask_valid'].to(logits.device) for t in targets])
        union=torch.stack([t['masks'].to(logits.device).float().amax(0) if len(t['masks'])
                           else logits.new_zeros(size) for t in targets])
    elif supervision!='feature':raise ValueError('Unknown foreground supervision')
    fg=logits[:,0];bce=F.binary_cross_entropy_with_logits(fg,union,reduction='none')
    bce=(bce*valid).sum()/valid.sum().clamp(min=1)
    probability,truth=fg.sigmoid()*valid,union*valid
    dice=(1-(2*(probability*truth).flatten(1).sum(1)+1)/
          (probability.flatten(1).sum(1)+truth.flatten(1).sum(1)+1)).mean()
    return bce,dice
