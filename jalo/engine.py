from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from .data import CLASSES, DrivingClips, collate, model_inputs
from .loss import SetCriterion
from .metrics import DetectionMetrics, InstanceMetrics, decode
from .model import build_model
from .runtime import (digest, environment, memory_bytes, restore_rng, rng_state, seed_all,
                      select_device, synchronize, write_json)


class SampleStream:
    """Resumable data order; its RNG is independent of model size and initialization."""
    def __init__(self, size, seed, epoch=0, position=0):
        self.size, self.seed, self.epoch, self.position = size, seed, epoch, position
        self.order = None

    def next(self):
        if self.position == self.size:
            self.epoch += 1
            self.position = 0
            self.order = None
        if self.order is None:
            self.order = list(range(self.size))
            random.Random(self.seed + self.epoch).shuffle(self.order)
        index = self.order[self.position]
        self.position += 1
        return index

    def state(self):
        return {"epoch": self.epoch, "position": self.position}


def dataset_for(config, split, training=False):
    if config.get("architecture") == "vehicle_roi_v2":
        from .video_data import adaptation_dataset
        return adaptation_dataset(config, split, training)
    if config.get("task") == "instance_segmentation":
        from .coco_data import CocoVehicles
        return CocoVehicles(config["data_root"], config["manifest"], split, config["image_size"],
                            config["train"].get("flip_probability", .5) if training else 0)
    return DrivingClips(config["data_root"], config["manifest"], split, config["image_size"],
                        config["model"]["history_seconds"],
                        config["train"].get("flip_probability", .5) if training else 0)


def load_checkpoint(path):
    # These files include optimizer/RNG Python objects. Only load local, trusted experiment files.
    import hashlib
    with open(path, "rb") as file:
        checkpoint = torch.load(file, map_location="cpu", weights_only=False)
        file.seek(0)
        checkpoint["loaded_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    version = checkpoint.get("format_version")
    expected = checkpoint.get("config", {}).get("classes", list(CLASSES))
    if version not in (1, 2, 3) or checkpoint.get("classes") != expected or (version == 1 and expected != list(CLASSES)):
        raise ValueError("Unsupported checkpoint format or class mapping")
    return checkpoint


def checkpoint_model(path, device):
    checkpoint = load_checkpoint(path)
    if checkpoint["step"] < 1:
        raise ValueError("Demo/evaluation requires a checkpoint that has completed training updates")
    model = build_model(checkpoint["config"], checkpoint["variant"], pretrained=False)
    if checkpoint.get('architecture') and checkpoint['architecture'] != getattr(model,'architecture','jalo_v1'):
        raise ValueError('Checkpoint architecture metadata disagrees with its config')
    if checkpoint.get("task", "detection") != model.task or len(checkpoint["classes"]) != model.num_classes:
        raise ValueError("Checkpoint task/class metadata disagrees with its architecture")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def export_checkpoint(source, destination):
    """Keep weights and inference settings; omit optimizer, RNG and local paths."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = load_checkpoint(source)
    if checkpoint['step'] < 1:
        raise ValueError('Only trained checkpoints can be exported')
    config = checkpoint['config']
    clean_config = {key: copy.deepcopy(config[key]) for key in
                    ('task', 'architecture', 'classes', 'model', 'image_size', 'seed',
                     'mask_projection','mask_objective','foreground_supervision') if key in config}
    clean_config.update(pretrained=False, device='auto')
    # Do not copy arbitrary run metadata or embedded training data into a Release.
    exported = {key: checkpoint[key] for key in
                ('task', 'architecture', 'classes', 'variant', 'step', 'manifest_sha256') if key in checkpoint}
    exported.update(format_version=2, inference_only=True, config=clean_config,
                    model={key: value.detach().cpu().clone() for key, value in checkpoint['model'].items()},
                    source_checkpoint_sha256=checkpoint['loaded_sha256'])
    if checkpoint.get('render_settings'):
        exported['render_settings'] = copy.deepcopy(checkpoint['render_settings'])
    if checkpoint.get('quality_certificate'):
        exported['quality_certificate']=copy.deepcopy(checkpoint['quality_certificate'])
    elif config.get('architecture')=='vehicle_roi_v2' and config.get('data_root'):
        lock=Path(config['data_root'])/'final_test_lock.json'
        if lock.exists():
            from .certification import certificate
            exported['quality_certificate']=certificate(checkpoint,json.loads(lock.read_text()))
    if checkpoint.get('initialization'):
        exported['initialization']=copy.deepcopy(checkpoint['initialization'])
    for key in ('annotations_sha256','coco_manifest_sha256'):
        if key in checkpoint:exported[key]=checkpoint[key]
    for key in ('preliminary','reference_coverage','evaluation_eligible','experiment_seconds','experiment_updates'):
        if key in checkpoint:exported[key]=copy.deepcopy(checkpoint[key])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(exported, temporary)
    # Portable exports contain tensors and basic Python types only.
    torch.load(temporary, map_location='cpu', weights_only=True)
    os.replace(temporary, destination)
    return {'checkpoint': str(destination), 'sha256': digest(destination),
            'source_checkpoint_sha256': checkpoint['loaded_sha256'], 'step': checkpoint['step'],
            'inference_only': True, 'bytes': destination.stat().st_size}


@torch.no_grad()
def evaluate_model(model, dataset, device):
    was_training = model.training
    model.eval()
    if getattr(model, 'architecture', None) == 'vehicle_roi_v2':
        from .roi_loss import ROICriterion
        criterion = ROICriterion(model.num_classes).to(device)
    else:
        criterion = SetCriterion(model.num_classes).to(device)
    metrics = (InstanceMetrics(dataset.classes) if model.task == "instance_segmentation"
               else DetectionMetrics())
    total_loss = inference_seconds = 0.
    started = time.perf_counter()
    for index in range(len(dataset)):
        batch = collate([dataset[index]])
        inputs = model_inputs(batch, device)
        synchronize(device)
        then = time.perf_counter()
        outputs = model(**inputs)
        synchronize(device)
        inference_seconds += time.perf_counter() - then
        total_loss += float(criterion(outputs, batch["target"])["total"].item())
        prediction = decode(outputs, [t["transform"] for t in batch["target"]])[0]
        metrics.update(prediction, batch["target"][0])
    results = metrics.compute()
    results.update(loss=total_loss / len(dataset), inference_seconds=inference_seconds,
                   inference_fps=len(dataset) / max(inference_seconds, 1e-9),
                   evaluation_seconds=time.perf_counter() - started,
                   active_parameters=model.active_parameter_count(),
                   instantiated_parameters=sum(p.numel() for p in model.parameters()),
                   device=str(device), **memory_bytes(device))
    model.train(was_training)
    return results


def evaluate_checkpoint(path, split="val", device="auto", config_override=None, output=None):
    device = select_device(device)
    model, checkpoint = checkpoint_model(path, device)
    config = checkpoint["config"] if config_override is None else config_override
    if checkpoint.get('inference_only') and config_override is None:
        raise ValueError('Release weights do not contain local data paths; pass --config for evaluation')
    if config.get("architecture") == "vehicle_roi_v2":
        from .adapt import evaluate_adaptation_checkpoint
        return evaluate_adaptation_checkpoint(model, checkpoint, path, config, split, device, output)
    results = evaluate_model(model, dataset_for(config, split), device)
    results.update(checkpoint=str(path), variant=checkpoint["variant"], split=split,
                   manifest_sha256=digest(config["manifest"]))
    if output:
        write_json(output, results)
        with Path(output).with_suffix(".csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(results))
            writer.writeheader()
            writer.writerow(results)
    return results


def _save(path, model, optimizer, scheduler, stream, step, config, best_score, train_seconds, manifest,
          phase, planned_steps):
    state = {"format_version": 2, "task": model.task, "classes": list(config.get("classes", CLASSES)), "config": config,
             "variant": model.variant, "step": step, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "stream": stream.state(), "rng": rng_state(), "best_score": best_score,
             "train_seconds": train_seconds, "manifest_sha256": manifest,
             "environment": environment(), "phase": phase, "planned_steps": planned_steps}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(config, variant="single", run_dir=None, initialize=None, resume=None, bootstrap=False):
    if config.get("architecture") == "vehicle_roi_v2":
        from .adapt import train_adaptation
        if resume and load_checkpoint(resume).get('inference_only'):
            raise ValueError('Release weights have no optimizer state; a full training checkpoint is required to resume')
        if variant != "single" or (initialize and resume):
            raise ValueError("Vehicle adaptation uses single; choose initialize OR resume")
        return train_adaptation(config, run_dir, resume, initialize)
    config = copy.deepcopy(config)
    seed_all(config["seed"])
    device = select_device(config.get("device", "auto"))
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint and checkpoint.get('inference_only'):
        raise ValueError('Release weights have no optimizer state; use --initialize instead of --resume')
    run_dir = Path(run_dir or Path(config["output_root"]) / variant)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Run directory is not empty: {run_dir}; use a new directory or --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data, val_data = dataset_for(config, "train", True), dataset_for(config, "val")
    manifest = digest(config["manifest"])
    if checkpoint:
        if checkpoint["variant"] != variant or checkpoint["manifest_sha256"] != manifest:
            raise ValueError("Resume variant or manifest differs from saved run")
        saved = copy.deepcopy(checkpoint["config"])
        for key in ("device", "data_root", "manifest", "output_root"):
            saved[key] = config[key]
        saved["train"]["max_seconds"] = config["train"].get("max_seconds", saved["train"].get("max_seconds"))
        if "max_seconds" not in config["train"]:
            saved["train"].pop("max_seconds", None)
        if saved != config:
            raise ValueError("Resume configuration differs; only paths/device may change")
    model = build_model(config, variant, pretrained=False if initialize or resume else None).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif initialize:
        initial = load_checkpoint(initialize)
        source_model, target_model = copy.deepcopy(initial["config"]["model"]), copy.deepcopy(config["model"])
        # The optional analytic mask support has no weights. Explicit --initialize
        # may enable it for a new, separately logged mask-training phase.
        if source_model.get("task") == target_model.get("task") == "instance_segmentation":
            source_model.pop("mask_box_prior", None)
            target_model.pop("mask_box_prior", None)
        if initial["manifest_sha256"] != manifest or source_model != target_model or initial["classes"] != list(config.get("classes", CLASSES)):
            raise ValueError("Initialization must use the same manifest, classes and parameter architecture")
        model.load_state_dict(initial["model"], strict=True)
        write_json(run_dir / "initialization.json", {"checkpoint": str(initialize),
            "sha256": initial["loaded_sha256"], "step": initial["step"], "source_config": initial["config"]})
    model.train()
    tc = config["train"]
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("backbone.") and not any(k in name for k in ("project", "fuse", "pixel", "refine")) else other).append(parameter)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": tc["backbone_lr"]},
                                  {"params": other, "lr": tc["lr"]}], weight_decay=tc["weight_decay"])
    phase = checkpoint["phase"] if checkpoint else ("bootstrap" if bootstrap else "branch")
    effective_batch = tc["batch_size"] * tc["accumulation"]
    steps = tc.get(phase + "_steps")
    if steps is None:
        steps = math.ceil(len(train_data) * tc[phase + "_epochs"] / effective_batch)
    if steps <= 0 or effective_batch <= 0:
        raise ValueError("Training steps and batch size must be positive")
    if checkpoint and steps != checkpoint["planned_steps"]:
        raise ValueError("Resume training budget differs from the saved scheduler budget")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    stream = SampleStream(len(train_data), config["seed"])
    step, best_score, elapsed = 0, -math.inf, 0.
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        stream = SampleStream(len(train_data), config["seed"], **checkpoint["stream"])
        step, best_score = checkpoint["step"], checkpoint["best_score"]
        elapsed = checkpoint["train_seconds"]
        restore_rng(checkpoint["rng"])
    else:
        # Architecture changes must not perturb the augmentation or sample-order RNG.
        seed_all(config["seed"] + 10000)
    # Resuming from best.pt into a fresh directory must preserve that historical best.
    if checkpoint and best_score > -math.inf and not (run_dir / "best.pt").exists():
        if Path(resume).name != "best.pt":
            raise ValueError("Resume from last.pt in its original run directory, or initialize a new run")
        import shutil
        shutil.copy2(resume, run_dir / "best.pt")
        saved_metrics = Path(resume).parent / "best_metrics.json"
        if saved_metrics.exists():
            shutil.copy2(saved_metrics, run_dir / "best_metrics.json")
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment())
    (run_dir / "manifest.json").write_bytes(Path(config["manifest"]).read_bytes())
    criterion = SetCriterion(model.num_classes).to(device)
    print(f"Training {variant}: device={device}, updates={step}/{steps}, samples={len(train_data)}", flush=True)
    log_path = run_dir / "train.jsonl"
    # Discard logs after the checkpoint when resuming, avoiding duplicate step numbers.
    if resume and log_path.exists():
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        log_path.write_text("".join(json.dumps(e) + "\n" for e in entries if e["step"] <= step))
    started = time.perf_counter()
    budget = tc.get("max_seconds", math.inf)
    if elapsed >= budget:
        print("Saved run already reached its time budget; increase budget explicitly to continue.", flush=True)
        return run_dir / "best.pt"
    time_limit = False
    for step in range(step + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for _ in range(tc["accumulation"]):
            batch = collate([train_data[stream.next()] for _ in range(tc["batch_size"])])
            outputs = model(**model_inputs(batch, device))
            parts = criterion(outputs, batch["target"])
            if not torch.isfinite(parts["total"]):
                raise FloatingPointError(f"Non-finite training loss at update {step}")
            (parts["total"] / tc["accumulation"]).backward()
            for key, value in parts.items():
                losses[key] = losses.get(key, 0.) + value.detach().item() / tc["accumulation"]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_grad"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        synchronize(device)
        training_seconds = elapsed + time.perf_counter() - started
        entry = {"step": step, **losses, "gradient_norm": float(grad_norm), "elapsed_seconds": training_seconds}
        with log_path.open("a") as log:
            log.write(json.dumps(entry, allow_nan=False) + "\n")
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"{variant} {step}/{steps} loss={losses['total']:.4f} elapsed={training_seconds:.1f}s", flush=True)
        time_limit = training_seconds >= budget
        if step % tc["validate_every"] == 0 or step == steps or time_limit:
            state = rng_state()
            result = evaluate_model(model, val_data, device)
            restore_rng(state)
            result.update(step=step, variant=variant, training_seconds=training_seconds)
            write_json(run_dir / "latest_metrics.json", result)
            with (run_dir / "evaluations.jsonl").open("a") as log:
                log.write(json.dumps(result, allow_nan=False) + "\n")
            metric = "mask_mAP" if model.task == "instance_segmentation" else "mAP"
            score = result[metric] if result[metric] is not None else -result["loss"]
            if score > best_score:
                best_score = score
                _save(run_dir / "best.pt", model, optimizer, scheduler, stream, step, config,
                      best_score, elapsed + time.perf_counter() - started, manifest, phase, steps)
                write_json(run_dir / "best_metrics.json", result)
            print(f"Validation {variant}: AP50={result['AP50']}, mAP={result['mAP']}, mask_mAP={result.get('mask_mAP')}", flush=True)
        time_limit = elapsed + time.perf_counter() - started >= budget
        if step % tc["checkpoint_every"] == 0 or step == steps or time_limit:
            _save(run_dir / "last.pt", model, optimizer, scheduler, stream, step, config,
                  best_score, elapsed + time.perf_counter() - started, manifest, phase, steps)
        if time_limit:
            print(f"Time budget reached at update {step}; resumable last.pt saved.", flush=True)
            break
    write_json(run_dir / "status.json", {"step": step, "planned_steps": steps,
               "stop_reason": "time_budget" if time_limit else "updates_complete",
               "elapsed_seconds": elapsed + time.perf_counter() - started})
    return run_dir / "best.pt"


def experiment(config, resume=False):
    if config.get('architecture') == 'vehicle_roi_v2':
        raise ValueError('Vehicle ROI adaptation is single-frame only; use train with its matching configuration')
    base = Path(config["output_root"])
    seeds = config.get("seeds", [config["seed"]])
    comparisons = []
    def stage(cfg, variant, directory, initialize=None, bootstrap=False):
        last = directory / "last.pt"
        if resume and last.exists():
            checkpoint = load_checkpoint(last)
            if checkpoint["step"] >= checkpoint["planned_steps"]:
                # Even skipped runs must still correspond to the requested data and configuration.
                if checkpoint["config"] != cfg or checkpoint["manifest_sha256"] != digest(cfg["manifest"]):
                    raise ValueError(f"Completed stage configuration changed: {directory}")
                return directory / "best.pt"
            return train(cfg, variant, directory, resume=last, bootstrap=bootstrap)
        return train(cfg, variant, directory, initialize=initialize, bootstrap=bootstrap)
    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg["seed"] = seed
        root = base / f"seed_{seed}" if len(seeds) > 1 else base
        bootstrap = stage(cfg, "single", root / "bootstrap", bootstrap=True)
        write_json(root / "experiment.json", {"initial_checkpoint": str(bootstrap),
                   "initial_sha256": digest(bootstrap), "seed": seed, "manifest_sha256": digest(cfg["manifest"])})
        for variant in ("single", "temporal", "gated"):
            best = stage(cfg, variant, root / variant, initialize=bootstrap)
            results = json.loads((best.parent / "best_metrics.json").read_text())
            results.update(seed=seed, checkpoint=str(best))
            comparisons.append(results)
            write_json(base / "comparison.json", comparisons)
            with (base / "comparison.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
    return comparisons
