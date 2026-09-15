"""Image-conditioned object queries and shared local instance masks.

Only torchvision's ImageNet ResNet-18 backbone is initialized with external weights.
No annotation, image identity, time, cabin polygon, or fixed ROI enters inference.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import Backbone, spatial_position
from .geometry import cxcywh_to_xyxy


def inverse_sigmoid(x):
    return torch.logit(x.clamp(1e-4,1-1e-4))


def roi_grid(boxes, size):
    """Normalized image coordinates; pixel centers and align_corners=False throughout."""
    points=(torch.arange(size,device=boxes.device,dtype=boxes.dtype)+.5)/size
    y,x=torch.meshgrid(points,points,indexing='ij')
    lo=boxes[:,:2]-boxes[:,2:]/2
    return torch.stack((lo[:,0,None,None]+x*boxes[:,2,None,None],
                        lo[:,1,None,None]+y*boxes[:,3,None,None]),-1)*2-1


def sample_rois(features, boxes, size=28, chunk=16):
    """One image's features, arbitrary number of differentiable ROI boxes."""
    if not len(boxes):return features.new_empty((0,features.shape[1],size,size))
    return torch.cat([F.grid_sample(features.expand(len(part),-1,-1,-1),roi_grid(part,size),
                        mode='bilinear',padding_mode='zeros',align_corners=False) for part in boxes.split(chunk)])


def paste_roi_logits(logits, boxes, size):
    """Paste genuine local logits; pixels outside the ROI have negligible probability."""
    if not len(boxes):return logits.new_empty((0,*size))
    h,w=size;y,x=torch.meshgrid((torch.arange(h,device=boxes.device)+.5)/h,
                               (torch.arange(w,device=boxes.device)+.5)/w,indexing='ij')
    cx,cy,bw,bh=boxes.unbind(-1)
    gx=(x[None]-cx[:,None,None])/bw[:,None,None].clamp(min=1e-4)*2
    gy=(y[None]-cy[:,None,None])/bh[:,None,None].clamp(min=1e-4)*2
    grid=torch.stack((gx,gy),-1)
    result=F.grid_sample(logits[:,None],grid,align_corners=False,padding_mode='border')[:,0]
    return result.masked_fill((gx.abs()>1)|(gy.abs()>1),-20.)


def local_attention_mask(boxes, padding, expansion=1.5):
    b,q,_=boxes.shape;h,w=padding.shape[-2:]
    y,x=torch.meshgrid((torch.arange(h,device=boxes.device)+.5)/h,
                       (torch.arange(w,device=boxes.device)+.5)/w,indexing='ij')
    points=torch.stack((x.flatten(),y.flatten()),-1)
    distance=(points[None,None]-boxes[:,:,:2,None].transpose(-1,-2)).abs()
    inside=(distance<=boxes[:,:,None,2:]*expansion/2).all(-1)
    valid=~padding.flatten(1)
    if not valid.any(1).all():raise ValueError('Current frame contains no valid image pixels')
    allowed=inside & valid[:,None]
    # No all-masked softmax; nearest VALID token, never an arbitrary padded token.
    nearest=distance.square().sum(-1).masked_fill(~valid[:,None],float('inf')).argmin(-1)
    fallback=F.one_hot(nearest,h*w).bool()
    return ~(allowed | (fallback & ~allowed.any(-1,keepdim=True)))


class LocalDecoder(nn.Module):
    def __init__(self,dim,heads,classes):
        super().__init__();self.heads=heads
        self.self_attention=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.norms=nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.ffn=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
        self.classifier=nn.Linear(dim,classes+1)
        self.box=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,4))
        nn.init.zeros_(self.box[-1].weight);nn.init.zeros_(self.box[-1].bias)

    def forward(self,q,boxes,features,padding):
        b,d,h,w=features.shape
        q=self.norms[0](q+self.self_attention(q,q,q,need_weights=False)[0])
        memory=features.flatten(2).transpose(1,2)
        position=spatial_position(h,w,d,features.device,features.dtype)
        mask=local_attention_mask(boxes.detach(),padding).repeat_interleave(self.heads,0)
        q=self.norms[1](q+self.cross_attention(q,memory+position,memory,attn_mask=mask,need_weights=False)[0])
        q=self.norms[2](q+self.ffn(q))
        refined=(inverse_sigmoid(boxes)+self.box(q)).sigmoid()
        return q,refined,self.classifier(q)


class VehicleROI(nn.Module):
    architecture='vehicle_roi_v2'
    task='instance_segmentation'
    variant='single'
    def __init__(self,num_classes=3,dim=192,heads=6,layers=3,queries=100,pretrained=True,task='instance_segmentation',mask_projection='dense',mask_objective='global',foreground_supervision='feature'):
        super().__init__()
        if task!=self.task or layers not in (1,2,3) or dim%8 or dim%heads:
            raise ValueError('Invalid ROI architecture configuration')
        self.num_classes,self.dim,self.queries=num_classes,dim,queries
        if mask_projection not in ('dense','native_roi'):raise ValueError('Unknown mask projection')
        self.mask_projection=mask_projection
        if mask_objective not in ('global','local_roi'):raise ValueError('Unknown mask objective')
        self.mask_objective=mask_objective
        if foreground_supervision not in ('feature','input'):raise ValueError('Unknown foreground supervision')
        self.foreground_supervision=foreground_supervision
        self.backbone=Backbone(dim,pretrained,with_masks=True)
        self.center=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GELU(),nn.Conv2d(dim,num_classes,1))
        self.geometry=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GELU(),nn.Conv2d(dim,4,1))
        self.foreground=nn.Sequential(nn.Conv2d(dim,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1))
        self.query_position=nn.Sequential(nn.Linear(4,dim),nn.GELU(),nn.Linear(dim,dim))
        self.query_class=nn.Embedding(num_classes,dim)
        self.decoder=nn.ModuleList(LocalDecoder(dim,heads,num_classes) for _ in range(layers))
        self.mask_features=nn.Sequential(nn.Conv2d(dim,64,1),nn.GroupNorm(8,64),nn.GELU())
        self.mask_head=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(64,1,1))
        nn.init.constant_(self.center[-1].bias,-4.6)
        nn.init.zeros_(self.geometry[-1].weight)
        with torch.no_grad():self.geometry[-1].bias.copy_(torch.tensor([0.,0.,-2.2,-2.2]))
        self.phase='joint'

    def roi_masks(self,features,boxes):
        return [self.mask_head(sample_rois(features[i:i+1],box))[:,0] if len(box)
                else features.new_empty((0,56,56)) for i,box in enumerate(boxes)]

    def forward(self,images,padding,history_valid,time_deltas,visualize=False,mask_query_threshold=None):
        if images.ndim!=5 or images.shape[1:3]!=(3,3):raise ValueError('Expected current+two history slots')
        if mask_query_threshold is not None and (self.training or not 0 <= mask_query_threshold <= 1):
            raise ValueError('Mask query filtering is inference-only and requires a threshold in [0, 1]')
        pyramid=self.backbone(images[:,0],return_pyramid=True)
        pixels=pyramid[4];b,d,h,w=pixels.shape
        invalid=F.interpolate(padding[:,0,None].float(),size=(h,w),mode='nearest')[:,0].bool()
        if invalid.flatten(1).all(1).any():raise ValueError('Current frame contains no valid image pixels')
        center_logits=self.center(pixels);geometry=self.geometry(pixels).sigmoid()
        foreground=self.foreground(pixels).masked_fill(invalid[:,None],-20.)
        mask_features=self.mask_features(pixels)
        common={'center_logits':center_logits,'proposal_geometry':geometry,'foreground_logits':foreground,
                'mask_features':mask_features,'dense_padding':invalid}
        scores=center_logits.sigmoid().masked_fill(invalid[:,None],-1.)
        peaks=(scores==F.max_pool2d(scores,3,1,1)) & ~invalid[:,None]
        peak_scores=scores.masked_fill(~peaks,-1).flatten(1)
        count=min(self.queries,peak_scores.shape[1]);_,indices=peak_scores.topk(count,dim=1)
        labels=indices//(h*w);indices=indices%(h*w)
        gathered=geometry.flatten(2).gather(2,indices[:,None].expand(-1,4,-1)).transpose(1,2)
        offsets=gathered[:,:,:2]
        coords=torch.stack((indices%w,indices//w),-1).to(pixels.dtype)
        centers=(coords+offsets)/pixels.new_tensor([w,h])
        boxes=torch.cat((centers,gathered[:,:,2:].clamp(min=1e-3)),dim=-1)
        q=pixels.flatten(2).gather(2,indices[:,None].expand(-1,d,-1)).transpose(1,2)
        q=q+self.query_position(boxes.detach())+self.query_class(labels)
        aux=[]
        for layer,stride in zip(self.decoder,(32,16,8)):
            feature=pyramid[stride]
            pad=F.interpolate(padding[:,0,None].float(),size=feature.shape[-2:],mode='nearest')[:,0].bool()
            q,boxes,logits=layer(q,boxes,feature,pad)
            aux.append({'logits':logits,'boxes':boxes})
        if mask_query_threshold is None:
            # Training and AP retain every query. Each ROI uses independent convolutions/GroupNorm.
            local=self.roi_masks(mask_features,[box for box in boxes])
            masks=torch.stack([paste_roi_logits(mask,box,(h,w)) for mask,box in zip(local,boxes)])
        else:
            # The renderer never draws these rejected queries; omit only their mask computation.
            probabilities=logits.softmax(-1)
            keep=(probabilities[:,:,:-1].amax(-1)>=mask_query_threshold) & (logits.argmax(-1)<self.num_classes)
            selected=self.roi_masks(mask_features,[box[k] for box,k in zip(boxes,keep)])
            local=[];canvases=[]
            for mask,box,k in zip(selected,boxes,keep):
                roi=mask_features.new_full((len(box),56,56),-20.)
                canvas=mask_features.new_full((len(box),h,w),-20.)
                roi[k]=mask;canvas[k]=paste_roi_logits(mask,box[k],(h,w))
                local.append(roi);canvases.append(canvas)
            masks=torch.stack(canvases)
        return {**common,'logits':logits,'boxes':boxes,'roi_logits':torch.stack(local),'mask_logits':masks,
                'mask_projection':self.mask_projection,'mask_objective':self.mask_objective,
                'foreground_supervision':self.foreground_supervision,'aux_outputs':aux[:-1]}

    def active_parameter_count(self):
        return sum(p.numel() for p in self.parameters())
