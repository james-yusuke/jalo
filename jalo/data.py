from __future__ import annotations

import bisect
import functools
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import Letterbox
from .runtime import digest, write_json

CLASSES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
DOWNLOAD_GUIDE = """BDD100K MOT 2020 is required (not the 100K still-image detection set).
Official instructions: https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst
Distribution: https://dl.cv.ethz.ch/bdd100k/data/
Obtain MOT 2020 Images (train/val) and MOT 2020 Labels under the dataset's license.
Expected layout under --root:
  images/track/{train,val}/VIDEO/FRAME.jpg
  labels/box_track_20/{train,val}/VIDEO.json
No data or model metrics are substituted if the official distribution is unavailable.
"""


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {relative}")
    return path


def read_frames(path):
    value = json.loads(Path(path).read_text())
    frames = value.get("frames") if isinstance(value, dict) else value
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No Scalabel frames: {path}")
    for frame in frames:
        if "frameIndex" not in frame:
            frame["frameIndex"] = int(Path(frame["name"]).stem.rsplit("-", 1)[-1]) - 1
    frames = sorted(frames, key=lambda x: x["frameIndex"])
    indices = [f["frameIndex"] for f in frames]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate frameIndex: {path}")
    return frames


def prepare(root, small=True, seed=0, train_videos=4, val_videos=2, max_frames=64, overwrite=False):
    root = Path(root).resolve()
    output = root / ("manifest_small.json" if small else "manifest_full.json")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to regenerate")
    manifest = {"format_version": 1, "dataset": "bdd100k_mot2020", "fps": 5,
                "seed": seed, "classes": list(CLASSES), "splits": {}}
    all_names = {}
    for split, limit in (("train", train_videos), ("val", val_videos)):
        files = sorted((root / "labels" / "box_track_20" / split).glob("*.json"))
        if not files:
            raise FileNotFoundError(f"Missing MOT labels in {root}:\n{DOWNLOAD_GUIDE}")
        all_names[split] = {p.stem for p in files}
        if small:
            if len(files) < limit:
                raise ValueError(f"{split}: need {limit} videos, found {len(files)}")
            files = sorted(random.Random(seed).sample(files, limit))
        videos = []
        for label_file in files:
            frames = read_frames(label_file)
            if small:
                frames = frames[:max_frames]
            names, indices = [], []
            for f in frames:
                if f.get("videoName", label_file.stem) != label_file.stem:
                    raise ValueError(f"videoName differs from label filename: {label_file}")
                relative = Path("images/track") / split / label_file.stem / f["name"]
                path = contained_path(root, relative)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing frame: {path}\n{DOWNLOAD_GUIDE}")
                with Image.open(path) as im:
                    im.verify()
                names.append(str(relative))
                indices.append(f["frameIndex"])
            videos.append({"name": label_file.stem, "labels": str(label_file.relative_to(root)),
                           "labels_sha256": digest(label_file), "frames": names, "indices": indices})
        manifest["splits"][split] = videos
    if all_names["train"] & all_names["val"]:
        raise ValueError("Video identities overlap between official train and val")
    write_json(output, manifest)
    return output


class DrivingClips(Dataset):
    def __init__(self, root, manifest, split, image_size=(256, 448), history_seconds=(.2, .4),
                 flip_probability=0):
        self.root = Path(root).resolve()
        self.manifest = json.loads(Path(manifest).read_text())
        if tuple(self.manifest["classes"]) != CLASSES:
            raise ValueError("Manifest class order differs from model class order")
        split_names = [{v["name"] for v in values} for values in self.manifest["splits"].values()]
        if any(a & b for i, a in enumerate(split_names) for b in split_names[i + 1:]):
            raise ValueError("Manifest splits overlap")
        self.videos = self.manifest["splits"][split]
        self.image_size, self.history_seconds = tuple(image_size), tuple(history_seconds)
        self.flip_probability = flip_probability
        self.fps = float(self.manifest["fps"])
        if self.fps <= 0 or any(s <= 0 for s in history_seconds):
            raise ValueError("Frame rate and history offsets must be positive")
        self.samples = [(vi, fi) for vi, video in enumerate(self.videos) for fi in range(len(video["frames"]))]
        if not self.samples:
            raise ValueError(f"Empty {split} split")
        for video in self.videos:
            if digest(contained_path(self.root, video["labels"])) != video["labels_sha256"]:
                raise ValueError(f"Labels changed since prepare: {video['labels']}")
            if video["indices"] != sorted(set(video["indices"])):
                raise ValueError("Frame indices must be strictly increasing")

    @functools.lru_cache(maxsize=4)
    def labels(self, video_index):
        video = self.videos[video_index]
        return {f["frameIndex"]: f for f in read_frames(contained_path(self.root, video["labels"]))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        vi, fi = self.samples[index]
        video = self.videos[vi]
        indices = video["indices"]
        current_index = indices[fi]
        chosen, valid, deltas = [fi], [], []
        for offset in self.history_seconds:
            past = bisect.bisect_right(indices, current_index - offset * self.fps + 1e-6) - 1
            ok = past >= 0 and past < fi
            chosen.append(past if ok else fi)
            valid.append(ok)
            deltas.append((current_index - indices[past]) / self.fps if ok else offset)
        rgbs = [np.asarray(Image.open(contained_path(self.root, video["frames"][i])).convert("RGB")) for i in chosen]
        if any(rgb.shape != rgbs[0].shape for rgb in rgbs):
            raise ValueError(f"Frame size changes inside clip: {video['name']}")
        frame = self.labels(vi)[current_index]
        boxes, labels, tracks, occluded, ignored = [], [], [], [], []
        ih, iw = rgbs[0].shape[:2]
        for label in frame.get("labels") or []:
            category = label.get("category")
            if category == "person":
                category = "pedestrian"
            box = label.get("box2d")
            if box is None:
                continue
            # Scalabel uses inclusive right/bottom coordinates; internal boxes are half-open.
            xyxy = [max(0, box["x1"]), max(0, box["y1"]),
                    min(iw, box["x2"] + 1), min(ih, box["y2"] + 1)]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            attrs = label.get("attributes") or {}
            if attrs.get("crowd", False) or attrs.get("ignore", False):
                ignored.append(xyxy)
                continue
            if category not in CLASSES:
                continue
            boxes.append(xyxy)
            labels.append(CLASSES.index(category))
            tracks.append(str(label["id"]))
            occluded.append(bool(attrs.get("occluded", False)))
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        ignore_boxes = torch.tensor(ignored, dtype=torch.float32).reshape(-1, 4)
        flip = random.random() < self.flip_probability
        if flip:
            rgbs = [np.ascontiguousarray(rgb[:, ::-1]) for rgb in rgbs]
            for values in (boxes, ignore_boxes):
                values[:, [0, 2]] = iw - values[:, [2, 0]]
        transform = Letterbox.create(rgbs[0], self.image_size)
        processed = [transform.image(rgb) for rgb in rgbs]
        target = {"boxes": transform.boxes_to_model(boxes), "labels": torch.tensor(labels, dtype=torch.long),
                  "original_boxes": boxes, "ignore_boxes": ignore_boxes,
                  "ignore_model_boxes": transform.boxes_to_model(ignore_boxes),
                  "track_ids": tracks, "occluded": occluded, "video": video["name"],
                  "frame_index": current_index, "image_id": index + 1, "transform": transform,
                  "time": current_index / self.fps}
        return {"images": torch.stack([p[0] for p in processed]),
                "padding": torch.stack([p[1] for p in processed]),
                "history_valid": torch.tensor(valid), "time_deltas": torch.tensor(deltas), "target": target}


def collate(samples):
    return {key: [s[key] for s in samples] if key == "target" else torch.stack([s[key] for s in samples])
            for key in samples[0]}


def model_inputs(batch, device):
    return {k: batch[k].to(device) for k in ("images", "padding", "history_valid", "time_deltas")}
