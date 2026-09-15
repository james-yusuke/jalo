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
