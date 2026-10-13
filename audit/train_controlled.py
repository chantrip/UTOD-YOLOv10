#!/usr/bin/env python3
"""Controlled CIoU/WIoU training; never silently use a benchmark test set for selection.

Examples (run on HPC, from a checkout of the audit branch):
    python audit/train_controlled.py --data data/uodd.yaml --model yolov10n.yaml --box-loss ciou --name uodd_base_ciou_s42
    python audit/train_controlled.py --data data/uodd.yaml --model yolov10n.yaml --box-loss wiou --name uodd_base_wiou_s42

A dataset with no distinct test partition is refused by default. For *exploratory
development only*, pass --development-only; any resulting metrics are validation
metrics, NOT an independent held-out test. No new empirical results are supplied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolved_sources(raw, root: Path) -> list[Path]:
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else raw
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("Expected a string or list of dataset split paths")
    return [(Path(s) if Path(s).is_absolute() else root / s).resolve() for s in items]


def inspect_data_yaml(filename: Path) -> dict:
    filename = filename.resolve()
    cfg = yaml.safe_load(filename.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{filename}: dataset YAML must be a mapping")
    raw_root = Path(str(cfg.get("path") or filename.parent))
    root = (raw_root if raw_root.is_absolute() else filename.parent / raw_root).resolve()
    paths = {s: _resolved_sources(cfg.get(s), root) for s in ("train", "val", "test")}
    if not paths["train"] or not paths["val"]:
        raise ValueError("Both train and val partitions must be explicitly declared")
    if set(paths["train"]) & set(paths["val"]):
        raise ValueError("Train and validation paths must be disjoint")
    if paths["test"] and ((set(paths["train"]) | set(paths["val"])) & set(paths["test"])):
        raise ValueError("Train, validation and test paths must be disjoint")
    val_named_test = any("test" in p.parts for p in paths["val"])
    reasons = []
    if not paths["test"]:
        reasons.append("No separate test: partition in dataset YAML")
    if val_named_test:
        reasons.append("Validation path points inside a directory named 'test'")
    return {
        "yaml": str(filename),
        "root": str(root),
        "splits": {k: [str(p) for p in v] for k, v in paths.items()},
        "development_only_reasons": reasons,
        "note": "Different directory names do not prove there is no duplicated image content.",
    }


def current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    p.add_argument("--model", type=str, required=True, help="Model YAML (e.g. yolov10n.yaml)")
    p.add_argument("--box-loss", choices=("ciou", "wiou"), required=True)
    p.add_argument("--name", required=True, help="Distinct experiment name")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--development-only", action="store_true",
                   help="Allow absent independent test partition; never claim final test scores")
    p.add_argument("--dry-run", action="store_true", help="Print configuration without training")
    args = p.parse_args()
    evidence = inspect_data_yaml(args.data)
    problems = evidence["development_only_reasons"]
    if problems and not args.development_only:
        p.error("; ".join(problems) + ". Create a valid val/test split or pass --development-only.")
    if problems:
        print("WARNING: DEVELOPMENT-ONLY EXPERIMENT — " + "; ".join(problems), file=sys.stderr)
    configuration = {
        "base_commit": current_commit(), "box_loss": args.box_loss,
        "model": args.model, "name": args.name, "seed": args.seed,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "development_only": bool(problems),
        "dataset": evidence,
    }
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        return

    # Import only after verification; no GPU/custom extension needed for --dry-run.
    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLOv10

    model = YOLOv10(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        pretrained=False,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        degrees=15,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.3,
        workers=0,
        seed=args.seed,
        box_loss=args.box_loss,
        name=args.name,
    )


if __name__ == "__main__":
    main()
