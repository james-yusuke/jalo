from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="JALO: causal temporal attention research")
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Validate BDD100K, COCO or reviewed video instances and create a reproducible manifest")
    prep.add_argument("--dataset", choices=["bdd100k", "coco-instances", "video-instances"], default="bdd100k")
    prep.add_argument("--input", help="Source video for video-instances; annotations must already be reviewed")
    prep.add_argument("--root", help="Dataset root; defaults depend on --dataset")
    prep.add_argument("--annotations", help="COCO trainval ZIP, or reviewed video COCO JSON for video-instances")
    prep.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto",
                      help="Accepted for CLI consistency; data preparation runs on CPU")
    prep.add_argument("--full", action="store_true")
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--instructions", action="store_true")
    prep.add_argument("--overwrite", action="store_true")
    for name in ("train", "experiment"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/mac_small.yaml")
        p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"])
        if name == "experiment":
            p.add_argument("--resume", action="store_true")
        if name == "train":
            p.add_argument("--variant", choices=["single", "temporal", "gated"], default="single")
            p.add_argument("--run-dir")
            group = p.add_mutually_exclusive_group()
            group.add_argument("--initialize", help="Common single-frame checkpoint; resets optimizer and RNG")
            group.add_argument("--resume", help="Trusted local checkpoint; restores complete training state")
            p.add_argument("--bootstrap", action="store_true", help="Use bootstrap training budget")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--split", choices=["train", "val", "test"], default="val")
    ev.add_argument("--config", help="Optional dataset path configuration")
    ev.add_argument("--output", default="outputs/evaluation.json")
    ev.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    d = sub.add_parser("demo")
    d.add_argument("--input", required=True)
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--output", default="outputs/demo.mp4")
    d.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    d.add_argument("--no-preview", action="store_true")
    d.add_argument("--max-frames", type=int)
    d.add_argument("--overwrite", action="store_true")
    d.add_argument("--render", choices=["dashboard", "masks"], default="dashboard")
    d.add_argument("--start-seconds", type=float, default=0.)
    d.add_argument("--duration", type=float)
    d.add_argument("--threshold", type=float)
    d.add_argument("--mask-threshold", type=float)
    d.add_argument("--alpha", type=float, default=.45)
    d.add_argument("--max-edge", type=int)
    d.add_argument("--foreground-only", action="store_true", default=None, help="Additionally exclude queries whose background class wins")
    sub.add_parser("doctor", help="Print dependency and GPU availability")
    export = sub.add_parser("export", help="Create a portable inference checkpoint for a Release")
    export.add_argument("--checkpoint", required=True, help="Trusted local training checkpoint")
    export.add_argument("--output", required=True)
    return root


def main():
    # Keep downloaded weights and visualization caches local to the project by default.
    os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    args = parser().parse_args()
    try:
        from .runtime import environment, load_config
        if args.command == "doctor":
            result = environment()
        elif args.command == "export":
            from .engine import export_checkpoint
            result = export_checkpoint(args.checkpoint, args.output)
        elif args.command == "prepare":
            if args.dataset == "video-instances":
                from .video_data import prepare_video
                if args.instructions:
                    print('video-instances: supply --input source.webm and --annotations reviewed COCO JSON. '
                          'The annotation JSON must contain info, images, annotations, categories and review; '
                          'the reference interval specification is jalo.video_data.INTERVALS.')
                    return
                if not args.input or not args.annotations:
                    raise ValueError("video-instances requires --input source.webm and --annotations reviewed.json")
                print(json.dumps({"manifest":str(prepare_video(args.root or 'data/video_instances',args.input,args.annotations,args.overwrite))}))
                return
            args.root = args.root or ("data/coco_vehicle" if args.dataset == "coco-instances" else "data/bdd100k")
            from .data import DOWNLOAD_GUIDE, prepare
            if args.dataset == "coco-instances":
                from .coco_data import prepare_coco, ANNOTATIONS_URL
                if args.instructions:
                    print("COCO 2017: " + ANNOTATIONS_URL + "\nOnly selected images are downloaded; see https://cocodataset.org/#download")
                    return
                result = {"manifest": str(prepare_coco(args.root, args.annotations, args.seed, overwrite=args.overwrite))}
                print(json.dumps(result, ensure_ascii=False))
                return
            if args.instructions:
                print(DOWNLOAD_GUIDE)
                return
            result = {"manifest": str(prepare(args.root, small=not args.full, seed=args.seed, overwrite=args.overwrite))}
        elif args.command in {"train", "experiment"}:
            from .engine import experiment, train
            config = load_config(args.config)
            if args.device:
                config["device"] = args.device
            if args.command == "experiment":
                result = experiment(config, args.resume)
            else:
                result = {"checkpoint": str(train(config, args.variant, args.run_dir, args.initialize,
                                                  args.resume, args.bootstrap or (args.variant == "single" and not args.initialize)))}
        elif args.command == "evaluate":
            from .engine import evaluate_checkpoint
            result = evaluate_checkpoint(args.checkpoint, args.split, args.device,
                                         load_config(args.config) if args.config else None, args.output)
        elif args.command == "demo":
            from .video import demo
            if args.max_frames is not None and args.max_frames <= 0:
                raise ValueError("--max-frames must be positive")
            if args.render == "masks":
                from .mask_video import demo_masks
                result = demo_masks(args.input, args.checkpoint, args.output, args.device,
                    not args.no_preview, args.max_frames, args.overwrite, args.threshold,
                    args.mask_threshold, args.alpha, args.max_edge, args.start_seconds, args.duration, args.foreground_only)
            else:
                result = demo(args.input, args.checkpoint, args.output, args.device,
                              not args.no_preview, args.max_frames, args.overwrite, .3 if args.threshold is None else args.threshold)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as error:
        print(f"JALO: {error}", file=sys.stderr)
        raise SystemExit(2) from error
