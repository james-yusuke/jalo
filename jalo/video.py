from __future__ import annotations

import bisect
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import CLASSES
from .engine import checkpoint_model
from .encoding import H264Writer
from .geometry import Letterbox
from .metrics import decode
from .runtime import select_device, synchronize


class CausalHistory:
    def __init__(self, offsets=(.2, .4)):
        self.offsets = offsets
        self.frames = deque()

    def reset(self):
        self.frames.clear()

    def clip(self, rgb, timestamp, size):
        if self.frames and timestamp <= self.frames[-1][0]:
            raise ValueError("Video timestamps must increase; reset history at video boundaries")
        transform = Letterbox.create(rgb, size)
        image, padding = transform.image(rgb)
        times = [item[0] for item in self.frames]
        images, masks, valid, deltas = [image], [padding], [], []
        for offset in self.offsets:
            past = bisect.bisect_right(times, timestamp - offset + 1e-7) - 1
            ok = past >= 0
            valid.append(ok)
            if ok:
                pt, pi, pm = self.frames[past]
                images.append(pi)
                masks.append(pm)
                deltas.append(timestamp - pt)
            else:
                images.append(image)
                masks.append(padding)
                deltas.append(offset)
        self.frames.append((timestamp, image, padding))
        while len(self.frames) > 2 and self.frames[1][0] <= timestamp - max(self.offsets):
            self.frames.popleft()
        return {"images": torch.stack(images)[None], "padding": torch.stack(masks)[None],
                "history_valid": torch.tensor([valid]), "time_deltas": torch.tensor([deltas])}, transform


class FlowEstimator:
    def __init__(self, width=320):
        self.width, self.previous, self.previous_time = width, None, None

    def update(self, bgr, timestamp):
        h, w = bgr.shape[:2]
        fh = max(8, round(h * min(1, self.width / w)))
        fw = min(w, self.width)
        gray = cv2.cvtColor(cv2.resize(bgr, (fw, fh)), cv2.COLOR_BGR2GRAY)
        flow = np.zeros((fh, fw, 2), np.float32)
        motion = None
        if self.previous is not None:
            dt = timestamp - self.previous_time
            if dt <= 0:
                raise ValueError("Flow requires increasing timestamps")
            if gray.shape != self.previous.shape:
                raise ValueError("Video resolution changed")
            flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            rate = flow * np.array([w / fw, h / fh], np.float32) / dt
            motion = float(np.median(np.linalg.norm(rate, axis=2)))
        self.previous, self.previous_time = gray, timestamp
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((fh, fw, 3), np.uint8)
        hsv[..., 0] = (angle * 90 / np.pi).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(magnitude * 24, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        vectors = cv2.resize(bgr, (fw, fh))
        for y in range(8, fh, 16):
            for x in range(8, fw, 16):
                dx, dy = flow[y, x]
                cv2.arrowedLine(vectors, (x, y), (round(x + dx * 3), round(y + dy * 3)), (80, 240, 120), 1, tipLength=.3)
        return color, vectors, motion


class ClasswiseTracker:
    def __init__(self, fps, classes=CLASSES):
        self.classes = tuple(classes)
        import supervision as sv
        from trackers import ByteTrackTracker
        self.sv = sv
        self.trackers = [ByteTrackTracker(frame_rate=fps, track_activation_threshold=.3,
                        high_conf_det_threshold=.3, minimum_consecutive_frames=1) for _ in self.classes]
        self.ids = {}
        self.next_id = 1

    def update(self, prediction, timestamp):
        boxes, scores, labels = [prediction[k].numpy() for k in ("boxes", "scores", "labels")]
        tracks = []
        for category, tracker in enumerate(self.trackers):
            keep = labels == category
            detections = self.sv.Detections(xyxy=boxes[keep], confidence=scores[keep], class_id=labels[keep],
                data={"prediction_index": np.flatnonzero(keep)})
            result = tracker.update(detections, timestamp=timestamp)
            if result.tracker_id is None or len(result) == 0:
                continue
            for box, score, track_id, prediction_index in zip(result.xyxy, result.confidence, result.tracker_id, result.data["prediction_index"]):
                if track_id < 0:
                    continue
                key = (category, int(track_id))
                if key not in self.ids:
                    self.ids[key] = self.next_id
                    self.next_id += 1
                tracks.append({"box": box.tolist(), "score": float(score), "class_id": category,
                               "class_name": self.classes[category], "track_id": self.ids[key],
                               "prediction_index": int(prediction_index)})
        return tracks


def text(image, string, xy=(14, 28), size=.6, color=(230, 234, 240)):
    cv2.putText(image, string, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def fit(image, width, height, title):
    out = np.full((height, width, 3), (24, 20, 19), np.uint8)
    scale = min(width / image.shape[1], (height - 34) / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    x, y = (width - resized.shape[1]) // 2, 34 + (height - 34 - resized.shape[0]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    text(out, title, size=.5)
    return out


def graph(values, width=426, height=240):
    out = np.full((height, width, 3), (28, 24, 22), np.uint8)
    text(out, "Motion: median image flow (px/s)", size=.47)
    finite = [v for v in values if v is not None]
    if not finite:
        text(out, "Waiting for consecutive frames", (15, 90), .45)
        return out
    maximum = max(1., max(finite))
    for i in range(4):
        y = 55 + round(i * 150 / 3)
        cv2.line(out, (45, y), (width - 12, y), (55, 51, 48), 1)
    text(out, f"max {maximum:.1f}   now {finite[-1]:.1f}", (15, height - 10), .42)
    points = [(45 + round(i * (width - 60) / max(1, len(finite) - 1)), 205 - round(v * 150 / maximum)) for i, v in enumerate(finite)]
    if len(points) > 1:
        cv2.polylines(out, [np.array(points, np.int32)], False, (255, 160, 95), 2, cv2.LINE_AA)
    return out


def attention_panel(outputs, prediction, valid, offsets):
    out = np.full((240, 426, 3), (28, 24, 22), np.uint8)
    text(out, "Temporal Attention (last layer)", size=.48)
    if not len(prediction["scores"]):
        text(out, "No detection above threshold", (14, 100), .45)
        return out, None
    top = int(prediction["scores"].argmax())
    query = int(prediction["query_indices"][top])
    maps = outputs["attention"][0, query].detach().cpu().numpy()
    gate = float(outputs["gates"][0, query].item())
    peak = max(float(maps.max()), 1e-9)
    for i in range(2):
        heat = cv2.applyColorMap(np.uint8(np.clip(maps[i] / peak * 255, 0, 255)), cv2.COLORMAP_INFERNO)
        heat = cv2.resize(heat, (194, 126), interpolation=cv2.INTER_NEAREST)
        x = 14 + i * 206
        out[70:196, x:x + 194] = heat
        text(out, f"-{offsets[i]:.2f}s" if valid[i] else "No history", (x, 58), .42)
    text(out, f"{CLASSES[int(prediction['labels'][top])]} / query {query} / gate {gate:.3f}", (14, 221), .42)
    return out, {"query_index": query, "gate": gate, "weights": maps.tolist()}


@torch.no_grad()
def demo(input_path, checkpoint_path, output="outputs/demo.mp4", device="auto", preview=True,
         max_frames=None, overwrite=False, threshold=.3):
    device = select_device(device)
    output = Path(output)
    json_path = output.with_suffix(".jsonl")
    if Path(input_path).resolve() == output.resolve():
        raise ValueError("Input and output must be different files")
    if not overwrite and (output.exists() or json_path.exists()):
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have .mp4 extension")
    capture = cv2.VideoCapture(str(input_path))
    writer = log = None
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot read video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Input video has no valid FPS")
        model, checkpoint = checkpoint_model(checkpoint_path, device)
        if model.task != "detection":
            raise ValueError("Dashboard requires a detection checkpoint; use --render masks for instance_segmentation")
        config = checkpoint["config"]
        history = CausalHistory(config["model"]["history_seconds"])
        flow = FlowEstimator()
        tracker = ClasswiseTracker(fps)
        traces = defaultdict(lambda: deque(maxlen=30))
        last_seen = {}
        motion_history = deque(maxlen=150)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = H264Writer(output, fps, (1280, 720))
        log = json_path.open("w")
        previous_timestamp = None
        while max_frames is None or count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamp = pts if np.isfinite(pts) and (previous_timestamp is None or pts > previous_timestamp) else count / fps
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                timestamp = previous_timestamp + 1 / fps
            previous_timestamp = timestamp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs, transform = history.clip(rgb, timestamp, config["image_size"])
            synchronize(device)
            started = time.perf_counter()
            outputs = model(**{k: v.to(device) for k, v in inputs.items()}, visualize=True)
            synchronize(device)
            inference_seconds = time.perf_counter() - started
            # Low-confidence detections are retained for ByteTrack's second association pass.
            all_predictions = decode(outputs, [transform], threshold=.1)[0]
            prediction = {k: v[all_predictions["scores"] >= threshold] for k, v in all_predictions.items()}
            tracks = tracker.update(all_predictions, timestamp)
            annotated, trajectory = frame.copy(), frame.copy()
            # Show raw detections even before a tracker confirms its first observation.
            for box, score, category in zip(prediction["boxes"].tolist(), prediction["scores"].tolist(), prediction["labels"].tolist()):
                x1, y1, x2, y2 = map(round, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 115, 70), 1)
                text(annotated, f"{CLASSES[category]} {score:.2f}", (x1, max(16, y1 - 5)), .42)
            for track in tracks:
                x1, y1, x2, y2 = map(round, track["box"])
                track_id = track["track_id"]
                traces[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
                last_seen[track_id] = count
                color = (int((track_id * 67) % 170 + 80), 210, int((track_id * 43) % 160 + 80))
                text(annotated, f"ID {track_id}", (x1, min(frame.shape[0] - 5, y2 + 15)), .45, color)
                if len(traces[track_id]) > 1:
                    cv2.polylines(trajectory, [np.array(traces[track_id], np.int32)], False, color, 2)
            for track_id in list(traces):
                if count - last_seen[track_id] > 60:
                    del traces[track_id], last_seen[track_id]
            colors, vectors, motion = flow.update(frame, timestamp)
            motion_history.append(motion)
            attention, inspected = attention_panel(outputs, prediction, inputs["history_valid"][0].tolist(), inputs["time_deltas"][0].tolist())
            canvas = np.zeros((720, 1280, 3), np.uint8)
            canvas[:480, :854] = fit(annotated, 854, 480, f"JALO | {model.variant} | {device} | t={timestamp:.2f}s | inference {1/inference_seconds:.1f} FPS")
            canvas[:240, 854:] = attention
            canvas[240:480, 854:] = fit(trajectory, 426, 240, "Tracking trajectories")
            canvas[480:, :426] = graph(motion_history)
            canvas[480:, 426:852] = fit(colors, 426, 240, "Optical flow: color")
            canvas[480:, 852:] = fit(vectors, 428, 240, "Optical flow: vectors")
            writer.write(canvas)
            record = {"frame": count, "timestamp_seconds": timestamp, "variant": model.variant,
                      "device": str(device), "inference_seconds": inference_seconds,
                      "motion_px_per_second": motion, "detections": {k: v.tolist() for k, v in prediction.items()},
                      "tracks": tracks, "attention": inspected, "history_valid": inputs["history_valid"][0].tolist()}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            count += 1
            if preview:
                cv2.imshow("JALO", canvas)
                key = cv2.waitKey(1) & 0xff
                if key == 32:
                    key = cv2.waitKey(0) & 0xff
                if key in (ord("q"), 27):
                    break
        if count == 0:
            raise ValueError("Input video has no decodable frames")
    finally:
        capture.release()
        try:
            if writer is not None:
                writer.release()
        finally:
            if log is not None:
                log.close()
            if preview:
                cv2.destroyAllWindows()
    return {"video": str(output), "jsonl": str(json_path), "frames": count, "device": str(device)}
