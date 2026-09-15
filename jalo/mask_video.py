"""Render genuine query masks; never substitute boxes or track predictions for masks."""
from __future__ import annotations
import colorsys
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch

from .encoding import H264Writer, TimedVideoReader
from .engine import checkpoint_model
from .geometry import Letterbox
from .metrics import decode, mask_rle, restore_masks
from .runtime import select_device, write_json, digest
from .video import ClasswiseTracker


def track_color(track_id):
    rgb = colorsys.hsv_to_rgb((int(track_id) * .61803398875) % 1., .72, 1.)
    return tuple(round(v * 255) for v in rgb[::-1])


def paint_masks(frame, instances, alpha=.45):
    """Confidence wins overlaps. Return exactly the visible, once-colored regions."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    result = frame.copy()
    occupied = np.zeros(frame.shape[:2], bool)
    rendered = []
    for instance in sorted(instances, key=lambda i: i["score"], reverse=True):
        mask = np.asarray(instance["mask"], dtype=bool)
        if mask.shape != occupied.shape:
            raise ValueError("Mask must have original image dimensions")
        visible = mask & ~occupied
        if not visible.any():
            continue
        occupied |= visible
        color = np.array(track_color(instance["track_id"]), dtype=np.float32)
        # A per-channel lookup gives the same rounding as the original blend,
        # without gathering millions of RGB pixels into temporary float arrays.
        lookup = np.rint(np.arange(256, dtype=np.uint8)[:, None] * (1-alpha)
                         + color * alpha).astype(np.uint8).reshape(256, 1, 3)
        cv2.copyTo(cv2.LUT(frame, lookup), visible.view(np.uint8), result)
        rendered.append({k:v for k,v in instance.items() if k != "mask"} | {
            "color_bgr": color.astype(int).tolist(), "mask_rle": mask_rle(visible),
            "predicted_mask_rle": mask_rle(mask), "visible_pixels": int(visible.sum())})
    return result, rendered


def mux_audio(video, source, destination, start, duration):
    command = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(video), "-ss", str(start), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-af", "apad", "-t", str(duration),
               "-movflags", "+faststart", str(destination)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def tracked_instances(outputs, transform, tracker, timestamp, classes, threshold=.3,
                      mask_threshold=.5, foreground_only=True):
    """Shared by video rendering and sequential evaluation, with identical ByteTrack input."""
    prediction=decode({k:v for k,v in outputs.items() if k!='mask_logits'},[transform],.1,
                      foreground_only=foreground_only)[0]
    tracks=tracker.update(prediction,timestamp)
    winners=outputs['logits'][0].argmax(-1).detach().cpu()
    selected=[t for t in tracks if t['score']>=threshold and (not foreground_only or
        int(winners[int(prediction['query_indices'][t['prediction_index']])])<len(classes))]
    indices=[int(prediction['query_indices'][t['prediction_index']]) for t in selected]
    masks=restore_masks(outputs,0,indices,transform,mask_threshold)
    return [{'mask':mask.numpy(),'class_name':track['class_name'],'class_id':track['class_id'],
             'score':track['score'],'track_id':track['track_id'],
             'foreground_winner':int(winners[index])<len(classes)}
            for track,mask,index in zip(selected,masks,indices)]


def validate_full_export(checkpoint, final_test, settings):
    """Full exports must use the settings whose quality was actually measured."""
    identity = final_test.get('identity', {})
    profile = checkpoint.get('render_settings', {})
    if (not checkpoint.get('selection', {}).get('quality_pass') or
        identity.get('checkpoint_sha256') != checkpoint['loaded_sha256'] or
        identity.get('manifest_sha256') != checkpoint.get('manifest_sha256') or
        identity.get('render_settings') != profile or
        not (final_test.get('result') or {}).get('selected', {}).get('quality_pass')):
        raise ValueError('Full video export requires passing validation and the frozen final test. Use --duration for a diagnostic preview.')
    if any(settings[key] != profile.get(key) for key in settings):
        raise ValueError('Full video export requires the validated render settings. Explicit overrides remain available for a diagnostic --duration preview.')


@torch.no_grad()
def demo_masks(input_path, checkpoint_path, output, device="auto", preview=True, max_frames=None,
               overwrite=False, threshold=None, mask_threshold=None, alpha=.45, max_edge=None,
               start_seconds=0., duration=None, foreground_only=None):
    device = select_device(device)
    model, checkpoint = checkpoint_model(checkpoint_path, device)
    profile = checkpoint.get('render_settings', {})
    threshold = profile.get('threshold', .3) if threshold is None else threshold
    mask_threshold = profile.get('mask_threshold', .5) if mask_threshold is None else mask_threshold
    explicit_size = max_edge is not None
    max_edge = max(profile.get('image_size', [640])) if max_edge is None else max_edge
    foreground_only = profile.get('foreground_only', False) if foreground_only is None else foreground_only
    if checkpoint['config'].get('architecture') == 'vehicle_roi_v2': foreground_only = True
    if not (0 <= threshold <= 1 and 0 < mask_threshold < 1 and 0 <= alpha <= 1):
        raise ValueError("Invalid probability threshold or alpha")
    if max_edge < 32 or max_edge > 1280 or start_seconds < 0 or (duration is not None and duration <= 0):
        raise ValueError("max-edge must be 32..1280 and times must be valid")
    if model.task != "instance_segmentation":
        raise ValueError("Checkpoint has no mask head. --render masks requires an instance_segmentation checkpoint trained on real masks.")
    output = Path(output).resolve()
    source = Path(input_path).resolve()
    json_path = output.with_suffix('.jsonl')
    if output == source or output.suffix.lower() != '.mp4':
        raise ValueError("Use a separate .mp4 output path")
    if not overwrite and (output.exists() or json_path.exists()):
        raise FileExistsError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = TimedVideoReader(source, start_seconds, duration)
    fps, width, height = capture.fps, capture.width, capture.height
    scale = min(1., max_edge / max(width,height))
    size = [math.ceil(height*scale/32)*32, math.ceil(width*scale/32)*32]
    # Fit within the strict long-edge limit, including padding.
    size = [min(max_edge//32*32, value) for value in size]
    if not explicit_size and 'image_size' in profile:
        size = profile['image_size']
    if checkpoint['config'].get('architecture') == 'vehicle_roi_v2' and start_seconds == 0 and duration is None and (
            max_frames is None or max_frames >= capture.frame_count):
        try:
            settings={'threshold': threshold, 'mask_threshold': mask_threshold,
                'foreground_only': foreground_only, 'image_size': size, 'alpha': alpha}
            if 'mask_projection' in profile:settings['mask_projection']=model.mask_projection
            if checkpoint.get('inference_only') or checkpoint.get('quality_certificate'):
                from .certification import validate_certificate
                validate_certificate(checkpoint,settings)
            else:
                lock = Path(checkpoint['config']['data_root']) / 'final_test_lock.json'
                final = json.loads(lock.read_text()) if lock.exists() else {}
                validate_full_export(checkpoint, final, settings)
        except ValueError:
            capture.release()
            raise
    tracker = ClasswiseTracker(fps, checkpoint["classes"])
    limit = max_frames
    if duration is not None:
        limit = min(limit, round(duration*fps)) if limit is not None else round(duration*fps)
    started = time.perf_counter()
    count = colored_frames = colored_instances = 0
    reached_end = False
    source_probe = capture.probe
    with tempfile.TemporaryDirectory(prefix='.jalo-masks-', dir=output.parent) as temp:
        temp = Path(temp)
        writer = H264Writer(temp/'silent.mp4', fps, (width,height))
        try:
            with (temp/'frames.jsonl').open('w') as log:
                while limit is None or count < limit:
                    ok, frame = capture.read()
                    if not ok:
                        reached_end = True
                        break
                    source_time = start_seconds + count / fps
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    transform = Letterbox.create(rgb, size)
                    image, padding = transform.image(rgb)
                    inputs = {"images":image[None,None].expand(1,3,-1,-1,-1).to(device),
                        "padding":padding[None,None].expand(1,3,-1,-1).to(device),
                        "history_valid":torch.zeros(1,2,dtype=torch.bool,device=device),
                        "time_deltas":torch.tensor([[.2,.4]],device=device)}
                    outputs = (model(**inputs, mask_query_threshold=threshold)
                               if checkpoint['config'].get('architecture') == 'vehicle_roi_v2' else model(**inputs))
                    instances=tracked_instances(outputs,transform,tracker,count/fps,checkpoint['classes'],
                                                threshold,mask_threshold,foreground_only)
                    colored, rendered = paint_masks(frame, instances, alpha)
                    writer.write(colored)
                    log.write(json.dumps({"frame":count,"timestamp_seconds":count/fps,
                        "source_timestamp_seconds":source_time, "instances":rendered},allow_nan=False)+'\n')
                    count += 1
                    colored_frames += bool(rendered)
                    colored_instances += len(rendered)
                    if count == 1 or count % 300 == 0:
                        print(f"Masks: {count} frames, {count/max(time.perf_counter()-started,1e-6):.2f} processing FPS, colored frames={colored_frames}",flush=True)
                    if preview:
                        cv2.imshow('JALO masks',colored)
                        key=cv2.waitKey(1)&255
                        if key==32:
                            key=cv2.waitKey(0)&255
                        if key in (27,ord('q')):
                            break
        finally:
            capture.release()
            writer.release()
            if preview:
                cv2.destroyAllWindows()
        if not count:
            raise ValueError("No decodable frames in the requested interval")
        media_duration = count / fps
        if reached_end:
            media_duration = max(media_duration, float(source_probe.get("format", {}).get("duration", media_duration + start_seconds)) - start_seconds)
        mux_audio(temp/'silent.mp4',source,temp/'final.mp4',start_seconds,media_duration)
        os.replace(temp/'final.mp4', output)
        os.replace(temp/'frames.jsonl',json_path)
    result = {"input":str(source), "video":str(output),"jsonl":str(json_path),"checkpoint":str(Path(checkpoint_path).resolve()),
        "checkpoint_step":checkpoint["step"], "checkpoint_sha256":checkpoint["loaded_sha256"], "classes":checkpoint["classes"],
        "manifest_sha256":checkpoint.get("manifest_sha256"), "source_sha256":digest(source),
        "frames":count,"fps":fps,"duration_seconds":media_duration,"video_frame_duration_seconds":count/fps,"source_start_seconds":start_seconds,
        "colored_frames":colored_frames,"colored_instances":colored_instances,"alpha":alpha,
        "input_timing":"ffmpeg_fps_resampling", "source_timestamp_basis":"resampled_timeline_seconds",
        "class_threshold":threshold,"mask_threshold":mask_threshold,"background_winners_excluded":foreground_only,"inference_size":size,
        "architecture":checkpoint['config'].get('architecture','jalo_v1'),
        "mask_projection":getattr(model,'mask_projection','dense_image'),
        "mask_objective":getattr(model,'mask_objective','global'),
        "foreground_supervision":getattr(model,'foreground_supervision','feature'),
        "foreground_mask_intersection":'foreground_logits' in outputs,
        "output_size":[height,width],"elapsed_seconds":time.perf_counter()-started,"device":str(device)}
    write_json(output.with_suffix('.metadata.json'),result)
    return result
