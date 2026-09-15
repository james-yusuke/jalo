from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def select_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device: {requested}")
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        requested = next(d for d in ("cuda", "mps", "cpu") if available[d])
    if not available[requested]:
        raise RuntimeError(f"Requested {requested} is unavailable in this PyTorch/runtime.")
    return torch.device(requested)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_bytes(device):
    # resource is Unix-only; CUDA hosts may run Windows. Missing telemetry is not zero.
    peak_rss = None
    if platform.system() != "Windows":
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    values = {"process_peak_rss_bytes": peak_rss}
    if device.type == "cuda":
        values["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        values["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        values["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return values


def environment():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "opencv-python", "trackers",
                 "supervision", "pycocotools", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available()}


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)
