"""Browser-compatible H.264 output with the MP4 index at the start of the file."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


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
