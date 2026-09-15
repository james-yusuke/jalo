"""COCO ground-truth masks, with deterministic subsets and source provenance."""
from __future__ import annotations

import concurrent.futures
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_api
from torch.utils.data import Dataset

from .data import contained_path
from .geometry import Letterbox
from .runtime import digest, write_json

VEHICLES = ("car", "truck", "bus")
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as f:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
            temporary.replace(destination)
            return
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare_coco(root, archive=None, seed=0, counts=(1800, 200, 250, 50), overwrite=False):
    root = Path(root).resolve()
    output = root / "manifest.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to regenerate the same seeded subset")
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(archive) if archive else root / "annotations_trainval2017.zip"
    if not archive.exists():
        download(ANNOTATIONS_URL, archive)
    manifest = {"format_version": 2, "dataset": "coco2017_vehicle_instances", "task": "instance_segmentation",
                "seed": seed, "classes": list(VEHICLES), "source": ANNOTATIONS_URL,
                "annotations_sha256": digest(archive), "splits": {}}
    jobs = []
    with zipfile.ZipFile(archive) as z:
        for split, positive_count, negative_count in (("train", counts[0], counts[1]), ("val", counts[2], counts[3])):
            source = json.loads(z.read(f"annotations/instances_{split}2017.json"))
            ids = {c["id"]: VEHICLES.index(c["name"]) for c in source["categories"] if c["name"] in VEHICLES}
            by_image = defaultdict(list)
            for annotation in source["annotations"]:
                if annotation["category_id"] in ids:
                    by_image[annotation["image_id"]].append(annotation)
            images = {im["id"]: im for im in source["images"]}
            positives = sorted(i for i, anns in by_image.items() if any(not a.get("iscrowd", 0) for a in anns))
            negatives = sorted(i for i in images if i not in by_image)
            rng = random.Random(seed)
            selected = sorted(rng.sample(positives, positive_count) + rng.sample(negatives, negative_count))
            entries = []
            for image_id in selected:
                im = images[image_id]
                filename = Path(im["file_name"]).name
                relative = f"images/{split}2017/{filename}"
                url = f"http://images.cocodataset.org/{split}2017/{filename}"
                annotations = [{"source_id": a["id"], "label": ids[a["category_id"]], "bbox": a["bbox"],
                                "segmentation": a["segmentation"], "iscrowd": a.get("iscrowd", 0), "area": a["area"]}
                               for a in by_image[image_id]]
                entries.append({"id": image_id, "path": relative, "width": im["width"], "height": im["height"],
                                "source_url": url, "license_id": im.get("license"), "annotations": annotations})
                jobs.append((url, root / relative))
            manifest["splits"][split] = entries
            manifest[f"{split}_distribution"] = dict(Counter(VEHICLES[a["label"]] for im in entries for a in im["annotations"] if not a["iscrowd"]))
            manifest[f"{split}_licenses"] = source.get("licenses", [])
    if {i["id"] for i in manifest["splits"]["train"]} & {i["id"] for i in manifest["splits"]["val"]}:
        raise ValueError("COCO train and val image identities overlap")
    print(f"Downloading/checking {len(jobs)} selected COCO images", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, url, path) for url, path in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if i % 100 == 0:
                print(f"Images: {i}/{len(jobs)}", flush=True)
    for entries in manifest["splits"].values():
        for im in entries:
            p = contained_path(root, im["path"])
            with Image.open(p) as image:
                if image.size != (im["width"], im["height"]):
                    raise ValueError(f"Wrong image dimensions: {p}")
                image.verify()
            im["sha256"] = digest(p)
    write_json(output, manifest)
    return output


def annotation_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        rle = mask_api.merge(mask_api.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = mask_api.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return mask_api.decode(rle).astype(bool)


def transform_mask(mask, transform):
    resized = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(
        (transform.resized_w, transform.resized_h), Image.Resampling.NEAREST))
    out = torch.zeros(transform.height, transform.width, dtype=torch.bool)
    out[:transform.resized_h, :transform.resized_w] = torch.from_numpy(resized.copy()).bool()
    return out


class CocoVehicles(Dataset):
    classes = VEHICLES
    def __init__(self, root, manifest, split, image_size, flip_probability=0):
        self.root = Path(root).resolve()
        self.manifest = json.loads(Path(manifest).read_text())
        self.classes = tuple(self.manifest["classes"])
        if self.classes != VEHICLES:
            raise ValueError("Unexpected vehicle class order")
        all_ids = [{im["id"] for im in items} for items in self.manifest["splits"].values()]
        if any(a & b for i, a in enumerate(all_ids) for b in all_ids[i + 1:]):
            raise ValueError("Dataset split leakage")
        self.images = self.manifest["splits"][split]
        self.image_size, self.flip_probability = image_size, flip_probability
        if not self.images:
            raise ValueError("Empty COCO subset")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = self.images[index]
        rgb = np.asarray(Image.open(contained_path(self.root, im["path"])).convert("RGB"))
        h, w = rgb.shape[:2]
        boxes, labels, masks, original_masks, ignored = [], [], [], [], []
        areas = []
        crowd_areas = []
        flip = random.random() < self.flip_probability
        if flip:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        transform = Letterbox.create(rgb, self.image_size)
        image, padding = transform.image(rgb)
        crowd_masks = []
        ignore_mask = np.zeros((h, w), bool)
        if im.get("ignore_polygons"):
            ignore_mask = annotation_mask(im["ignore_polygons"], h, w)
            if flip: ignore_mask = np.ascontiguousarray(ignore_mask[:, ::-1])
            for label in range(len(self.classes)):
                crowd_masks.append((ignore_mask.copy(), label))
                crowd_areas.append(int(ignore_mask.sum()))
            # Keep each ambiguous region separate for classification ignore IoA.
            for polygon in im['ignore_polygons']:
                xy=np.asarray(polygon).reshape(-1,2)
                lo,hi=xy.min(0),xy.max(0)
                box=[lo[0],lo[1],hi[0],hi[1]]
                if flip:box[0],box[2]=w-box[2],w-box[0]
                ignored.append(box)
        for a in im["annotations"]:
            mask = annotation_mask(a["segmentation"], h, w)
            if flip:
                mask = np.ascontiguousarray(mask[:, ::-1])
            x, y, bw, bh = a["bbox"]
            box = [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
            if flip:
                box[0], box[2] = w - box[2], w - box[0]
            if a["iscrowd"]:
                crowd_masks.append((mask, a["label"]))
                crowd_areas.append(a.get("area", int(mask.sum())))
                ignore_mask |= mask
                ignored.append(box)
                continue
            if not mask.any() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            labels.append(a["label"])
            masks.append(transform_mask(mask, transform))
            original_masks.append(mask)
            areas.append(a.get("area", int(mask.sum())))
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        ignored = torch.tensor(ignored, dtype=torch.float32).reshape(-1, 4)
        mask_tensor = torch.stack(masks) if masks else torch.zeros(0, *self.image_size, dtype=torch.bool)
        valid_pixels = ~padding & ~transform_mask(ignore_mask, transform)
        target = {"boxes": transform.boxes_to_model(boxes), "labels": torch.tensor(labels, dtype=torch.long),
                  "masks": mask_tensor, "mask_valid": valid_pixels, "original_masks": original_masks, "original_areas": areas, "crowd_areas": crowd_areas,
                  "original_boxes": boxes, "ignore_boxes": ignored, "ignore_model_boxes": transform.boxes_to_model(ignored),
                  "original_ignore_mask": ignore_mask, "crowd_masks": crowd_masks, "track_ids": [], "occluded": [], "video": str(im["id"]),
                  "frame_index": 0, "image_id": im["id"], "transform": transform, "time": 0.}
        if "interior_exterior_windows" in im:
            interior = ~annotation_mask(im["interior_exterior_windows"], h, w)
            if im.get('interior_polygons'):interior |= annotation_mask(im['interior_polygons'],h,w)
            if flip: interior = np.ascontiguousarray(interior[:, ::-1])
            target.update(interior_mask=interior, video=self.manifest.get('source_sha256','video'),
                          frame_index=im['frame_index'], time=im['source_timestamp_seconds'])
        elif im.get('interior_polygons'):
            interior=annotation_mask(im['interior_polygons'],h,w)
            target['interior_mask']=np.ascontiguousarray(interior[:,::-1]) if flip else interior
        if 'source_id' in im:
            target.update(video=im['source_id'],frame_index=im['frame_index'],time=im['source_timestamp_seconds'])
        # Empty history is explicit: still images must not be represented as genuine video clips.
        return {"images": image[None].expand(3, -1, -1, -1), "padding": padding[None].expand(3, -1, -1),
                "history_valid": torch.tensor([False, False]), "time_deltas": torch.tensor([.2, .4]), "target": target}
