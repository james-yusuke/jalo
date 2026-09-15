"""Fixed, auditable video annotation splits and equal-domain training sampling."""
import json
from pathlib import Path
from collections import Counter, defaultdict
from torch.utils.data import Dataset
from PIL import Image
from .coco_data import CocoVehicles, VEHICLES, annotation_mask
from .data import contained_path
from .runtime import digest, write_json

INTERVALS={'train':[(20,120),(200,300),(420,520)],'val':[(150,170),(330,390)],'test':[(570,630),(690,750)]}


def prepare_video(root, source, annotations, overwrite=False):
    root=Path(root).resolve();source=Path(source).resolve();annotations=Path(annotations).resolve()
    destination=root/'manifest.json'
    if destination.exists() and not overwrite:raise FileExistsError(destination)
    labels=json.loads(annotations.read_text());info=labels['info'];source_sha=digest(source)
    if info['source_sha256']!=source_sha:raise ValueError('Annotation source video SHA differs')
    if info['intervals']!={k:[list(v) for v in a] for k,a in INTERVALS.items()}:raise ValueError('Video intervals differ from the fixed experiment')
    classes={c['id']:c['name'] for c in labels['categories']}
    if set(classes.values())!=set(VEHICLES):raise ValueError('Expected car/truck/bus annotations')
    expected={second:split for split,intervals in INTERVALS.items() for lo,hi in intervals for second in range(lo,hi,3)}
    if len(labels['images'])!=len(expected) or {im['second'] for im in labels['images']}!=set(expected):raise ValueError('Missing or duplicate fixed video frames')
    review={r['image_id']:r for r in labels['review']};by_image=defaultdict(list)
    image_ids={im['id'] for im in labels['images']}
    if len({a['id'] for a in labels['annotations']})!=len(labels['annotations']):raise ValueError('Duplicate annotation identity')
    if any(a['image_id'] not in image_ids or a['category_id'] not in classes for a in labels['annotations']):
        raise ValueError('Annotation references unknown image/class')
    for a in labels['annotations']:by_image[a['image_id']].append(a)
    splits={k:[] for k in INTERVALS};identities=set();frame_indices=set()
    for im in labels['images']:
        if im['id'] in identities or im['frame_index'] in frame_indices:raise ValueError('Video frame leakage')
        identities.add(im['id']);frame_indices.add(im['frame_index'])
        if im['split']!=expected[im['second']] or not review.get(im['id'],{}).get('reviewed'):raise ValueError('Unreviewed annotation or split leakage')
        if (im['frame_index']!=round(im['second']*info['fps']) or
                abs(im['source_timestamp_seconds']-im['second'])>2/info['fps']):
            raise ValueError('Frame identity/time differs from the fixed extraction')
        p=contained_path(root,im['file_name'])
        if digest(p)!=im['sha256']:raise ValueError('Annotated image changed')
        with Image.open(p) as image:
            if image.size!=(im['width'],im['height']):raise ValueError('Annotation dimensions disagree')
        anns=[]
        for a in by_image[im['id']]:
            mask=annotation_mask(a['segmentation'],im['height'],im['width'])
            if not mask.any():raise ValueError('Empty vehicle polygon')
            anns.append({'source_id':a['id'],'label':VEHICLES.index(classes[a['category_id']]),'bbox':a['bbox'],
                         'segmentation':a['segmentation'],'iscrowd':0,'area':int(mask.sum())})
        splits[im['split']].append({**im,'path':im['file_name'],'annotations':anns})
    manifest={'format_version':3,'dataset':'video_vehicle_instances','task':'instance_segmentation','classes':list(VEHICLES),
        'seed':0,'source':str(source),'source_sha256':source_sha,'annotations_sha256':digest(annotations),'intervals':info['intervals'],
        'annotation_provenance':info['annotation_provenance'],'splits':splits}
    for split,images in splits.items():
        manifest[split+'_distribution']=dict(Counter(VEHICLES[a['label']] for im in images for a in im['annotations']))
    write_json(destination,manifest);return destination


class EqualDomainVehicles(Dataset):
    """Exactly equal COCO/video samples per epoch, both from train only."""
    def __init__(self,coco,video):
        if coco.classes!=video.classes:raise ValueError('Mixed training classes disagree')
        self.coco,self.video=coco,video;self.classes=coco.classes
    def __len__(self):return 2*max(len(self.coco),len(self.video))
    def __getitem__(self,index):
        dataset=self.coco if index%2==0 else self.video
        sample=dataset[(index//2)%len(dataset)]
        sample['target']['domain']='coco' if index%2==0 else 'video'
        return sample


def adaptation_dataset(config,split,training=False):
    flip=config['train'].get('flip_probability',.5) if training else 0.
    video=CocoVehicles(config['data_root'],config['manifest'],split,config['image_size'],flip)
    if not training:return video
    coco=CocoVehicles(config['coco_root'],config['coco_manifest'],'train',config['image_size'],flip)
    return EqualDomainVehicles(coco,video)


def annotation_identity(image, annotations):
    """Bind a visual review to exact pixels, polygons, classes and split."""
    import hashlib
    payload={'image':image,'annotations':sorted(annotations,key=lambda a:a['id'])}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def prepare_videos(root, sources, annotations, overwrite=False):
    """Validate whole-video splits and reviewed COCO polygons without running a model.

    Source spec: sources[{id,path,sha256,split,intervals,stride_seconds,fps,
    title,author,license,license_url,source_url}]. Review records bind image_id,
    annotation_sha256 and an existing source/overlay review image SHA256.
    """
    import math
    import numpy as np
    root=Path(root).resolve();sources=Path(sources).resolve();annotations=Path(annotations).resolve()
    destination=root/'manifest.json'
    if destination.exists() and not overwrite:raise FileExistsError(destination)
    spec=json.loads(sources.read_text());labels=json.loads(annotations.read_text())
    if labels['info'].get('sources_sha256')!=digest(sources):raise ValueError('Annotation sources changed')
    if spec.get('split_unit')!='source_video':raise ValueError('Expected whole-video splits')
    by_source={};hashes=set();expected={}
    for source in spec['sources']:
        sid=source['id'];split=source['split'];sha=source['sha256']
        if sid in by_source or sha in hashes:raise ValueError('Duplicate source video / split leakage')
        if split not in ('train','val','test'):raise ValueError('Unknown video split')
        for key in ('title','author','license','license_url','source_url'):
            if not source.get(key):raise ValueError(f'Missing source attribution: {key}')
        path=Path(source['path'])
        if not path.is_absolute():path=sources.parent/path
        if digest(path)!=sha:raise ValueError(f'Source video changed: {sid}')
        if not math.isfinite(source['fps']) or source['fps']<=0:raise ValueError('Invalid extraction FPS')
        stride=source['stride_seconds']
        if not isinstance(stride,int) or stride<=0:raise ValueError('Invalid extraction interval')
        for start,end in source['intervals']:
            if start<0 or end<=start:raise ValueError('Invalid source interval')
            for second in range(start,end,stride):
                key=(sid,second)
                if key in expected:raise ValueError('Overlapping extraction intervals')
                expected[key]=split
        by_source[sid]={**source,'path':str(path.resolve())};hashes.add(sha)
    classes={c['id']:c['name'] for c in labels['categories']}
    if len(classes)!=3 or set(classes.values())!=set(VEHICLES):raise ValueError('Expected car/truck/bus categories')
    images=labels['images'];ids={im['id'] for im in images}
    if len(ids)!=len(images):raise ValueError('Duplicate image identity')
    keys=[(im['source_id'],im['second']) for im in images]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('Missing or duplicate fixed source frames')
    review={r['image_id']:r for r in labels['review']}
    if len(review)!=len(labels['review']) or set(review)!=ids:raise ValueError('Missing or duplicate annotation review')
    by_image=defaultdict(list);ann_ids=set()
    for a in labels['annotations']:
        if a['id'] in ann_ids or a['image_id'] not in ids or a['category_id'] not in classes:
            raise ValueError('Invalid annotation identity/class')
        ann_ids.add(a['id']);by_image[a['image_id']].append(a)
    splits={k:[] for k in ('train','val','test')};pixel_splits={}
    for im in images:
        source=by_source[im['source_id']];split=expected[(im['source_id'],im['second'])]
        if im['split']!=split:raise ValueError('Source video leaks across splits')
        if im['frame_index']!=round(im['second']*source['fps']) or abs(im['source_timestamp_seconds']-im['second'])>1/source['fps']:
            raise ValueError('Frame extraction timestamp mismatch')
        path=contained_path(root,im['file_name'])
        if digest(path)!=im['sha256']:raise ValueError('Annotated source image changed')
        with Image.open(path) as image:
            if image.size!=(im['width'],im['height']):raise ValueError('Image dimensions changed')
            import hashlib
            pixel_sha=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        if pixel_sha in pixel_splits and pixel_splits[pixel_sha]!=split:raise ValueError('Identical image pixels across splits')
        pixel_splits[pixel_sha]=split
        r=review[im['id']]
        if (not r.get('reviewed') or r.get('annotator')!='AI assistant' or
                r.get('annotation_sha256')!=annotation_identity(im,by_image[im['id']])):
            raise ValueError('Missing or stale visual annotation review')
        if digest(contained_path(root,r['overlay_file']))!=r['overlay_sha256']:raise ValueError('Reviewed overlay changed')
        h,w=im['height'],im['width'];anns=[]
        for key in ('ignore_polygons','interior_polygons'):
            if im.get(key):annotation_mask(im[key],h,w)
        if 'interior_present' not in im:raise ValueError('Explicit cabin review is required')
        if im['interior_present'] and not im.get('interior_polygons'):raise ValueError('Visible cabin requires reference polygons')
        for a in by_image[im['id']]:
            mask=annotation_mask(a['segmentation'],h,w);ys,xs=np.nonzero(mask)
            if not len(xs):raise ValueError('Empty reference vehicle')
            if a.get('iscrowd',0):raise ValueError('Use ignore polygons for ambiguous vehicle regions')
            box=np.asarray(a['bbox'],float)
            tight=np.array([xs.min(),ys.min(),xs.max()+1-xs.min(),ys.max()+1-ys.min()])
            if box.shape!=(4,) or not np.isfinite(box).all() or (np.abs(box-tight)>1).any():
                raise ValueError('Vehicle box must enclose its visible mask tightly')
            anns.append({'source_id':a['id'],'label':VEHICLES.index(classes[a['category_id']]),
                         'bbox':a['bbox'],'segmentation':a['segmentation'],'iscrowd':0,'area':int(mask.sum())})
        splits[split].append({**im,'path':im['file_name'],'annotations':anns})
    for values in splits.values():values.sort(key=lambda im:(im['source_id'],im['frame_index']))
    if any(not values for values in splits.values()):raise ValueError('Every split must contain reviewed frames')
    manifest={'format_version':4,'dataset':'video_vehicle_instances','task':'instance_segmentation',
        'classes':list(VEHICLES),'seed':spec.get('seed',0),'split_unit':'source_video','sources':list(by_source.values()),
        'sources_sha256':digest(sources),'annotations_sha256':digest(annotations),
        'annotation_provenance':labels['info']['annotation_provenance'],'splits':splits}
    for split,values in splits.items():
        manifest[split+'_distribution']=dict(Counter(VEHICLES[a['label']] for im in values for a in im['annotations']))
    write_json(destination,manifest);return destination
