"""Browser-compatible H.264 output with the MP4 index at the start of the file."""
from __future__ import annotations

import os
import json
import math
from fractions import Fraction
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class TimedVideoReader:
    """Decode by presentation time, resampling VFR inputs onto a constant-rate timeline."""
    def __init__(self, source, start=0., duration=None):
        self.source = Path(source)
        self.start, self.duration = start, duration
        self.process = None
        self.errors = None
        self.closed = False
        self.frame = 0
        self.probe = json.loads(subprocess.check_output([
            shutil.which('ffprobe') or 'ffprobe', '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(self.source)]))
        stream = next((s for s in self.probe['streams'] if s['codec_type'] == 'video'), None)
        if stream is None:
            raise ValueError('Input has no video stream')
        self.fps = 0.
        for key in ('avg_frame_rate', 'r_frame_rate'):
            try:
                rate = float(Fraction(stream.get(key, '0/1')))
            except (ValueError, ZeroDivisionError):
                continue
            if math.isfinite(rate) and rate > 0:
                self.fps = rate
                break
        self.width, self.height = stream['width'], stream['height']
        rotation = next((s['rotation'] for s in stream.get('side_data_list', []) if 'rotation' in s), 0)
        if abs(round(float(rotation))) % 180 == 90:
            self.width, self.height = self.height, self.width
        if not math.isfinite(self.fps) or self.fps <= 0 or self.width % 2 or self.height % 2:
            raise ValueError('Input needs valid FPS and even dimensions for yuv420p')
        self.source_duration = float(self.probe.get('format', {}).get('duration', 0))
        self.frame_count = round(self.source_duration * self.fps)

    def read(self):
        if self.closed:
            return False, None
        if self.process is None:
            self.errors = tempfile.TemporaryFile()
            command = [shutil.which('ffmpeg') or 'ffmpeg', '-hide_banner', '-loglevel', 'error',
                       '-i', str(self.source), '-map', '0:v:0', '-an', '-sn']
            if self.duration is not None:
                command += ['-t', str(self.duration)]
            # Resample before trimming: input seeking would discard a sparse VFR
            # frame that remains visible across the requested start time.
            timing = f'fps=fps={self.fps}:start_time=0:round=near,trim=start={self.start},setpts=PTS-STARTPTS'
            command += ['-vf', timing,
                        '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
            try:
                self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.errors)
            except BaseException:
                self.errors.close()
                raise
        data = self.process.stdout.read(self.width * self.height * 3)
        if not data:
            code = self.process.wait(timeout=30)
            if code:
                self.errors.seek(0)
                detail = self.errors.read(8192).decode('utf-8', errors='replace')
                self.release()
                raise RuntimeError(f'FFmpeg video decoding failed: {detail}')
            return False, None
        if len(data) != self.width * self.height * 3:
            self.release()
            raise RuntimeError('FFmpeg returned an incomplete video frame')
        self.frame += 1
        return True, np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)

    def release(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.stdout.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.errors.close()


class H264Writer:
    def __init__(self, output, fps, size):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("H.264 video output requires FFmpeg with libx264. Install FFmpeg and add it to PATH (macOS: brew install ffmpeg).")
        self.output = Path(output)
        self.width, self.height = size
        self.frames = 0
        self.closed = False
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.output.stem}-", suffix=".mp4", dir=self.output.parent)
        os.close(fd)
        self.pending = Path(name)
        self.errors = tempfile.TemporaryFile()
        command = [executable, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{self.width}x{self.height}",
                   "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.pending)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.errors)
        except BaseException:
            self.errors.close()
            self.pending.unlink(missing_ok=True)
            raise

    def write(self, frame):
        if self.closed:
            raise RuntimeError("Video writer is already closed")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("Video frame must be uint8 BGR with the configured dimensions")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.frames += 1
        except BrokenPipeError:
            self.release()  # Produces FFmpeg's useful error message, rather than a pipe traceback.
            raise RuntimeError("FFmpeg stopped accepting video frames") from None

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("FFmpeg did not finalize the MP4 within 60 seconds") from None
            if code:
                self.errors.seek(0)
                detail = self.errors.read(8192).decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"H.264 encoding failed: {detail}")
            if self.frames:
                os.replace(self.pending, self.output)
        finally:
            self.pending.unlink(missing_ok=True)
            self.errors.close()
