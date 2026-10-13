import argparse
import os
from pathlib import Path

import yaml
import random
import numpy as np
import torch

from ultralytics import YOLOv10

# Same dataset dict you used in training
DATASETS = {
    "tci": {
        "dir": "/home/dsai-st125287/project/TrashCan/instance_v",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/trashcan_i.yaml"
    },
    "tcm": {
        "dir": "/home/dsai-st125287/project/TrashCan/material_v",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/trashcan_m.yaml"
    },
    "ruod": {
        "dir": "/home/dsai-st125287/project/RUOD_YOLO",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/ruod.yaml"
    },
    "duo": {
        "dir": "/home/dsai-st125287/project/DUO",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/duo.yaml"
    },
    "uodd": {
        "dir": "/home/dsai-st125287/project/UODD_YOLO",
        "yaml": "/home/dsai-st125287/project/UTOD-YOLOv10/data/uodd.yaml"
    }
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds", type=str,
                        help="Dataset shortcode: tci, tcm, ruod, duo, uodd")
    parser.add_argument("--data", type=str,
                        help="Direct YAML path for audited generated train/val/test splits (overrides --ds)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained .pt weights (best.pt)")
    parser.add_argument("--name", type=str, required=True,
                        help="Name for the validation run")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Explicit evaluation partition; do not label val as final test")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Resolve dataset paths
    # ---------------------------------------------------------
    if args.data:
        YAML_PATH = args.data
    elif args.ds in DATASETS:
        YAML_PATH = DATASETS[args.ds]["yaml"]
    else:
        parser.error("Provide --data DATA_YAML or a valid --ds dataset shortcode")

    # Fail closed: never silently evaluate val data under a claimed test label.
    dataset_config = yaml.safe_load(Path(YAML_PATH).read_text(encoding="utf-8"))
    if not isinstance(dataset_config, dict) or not dataset_config.get(args.split):
        parser.error(f"Evaluation split {args.split!r} is not configured in {YAML_PATH}")

    print(f"Using dataset '{args.ds if args.ds else 'custom YAML'}'")
    print(f"YAML        = {YAML_PATH}")
    print(f"Weights     = {args.weights}")
    print(f"Run name    = {args.name}")
    print(f"Eval split  = {args.split}")

    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    model = YOLOv10(args.weights)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    print("\n=== Running YOLOv10 Validation ===")

    metrics = model.val(
        data=YAML_PATH,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,    # Explicit validation or held-out test partition
        plots=False,
        save_json=True,      # JSON predictions alone do NOT calculate size-specific AP
        name=args.name
    )

    print("\n✅ Validation complete.")

    # ---------------------------------------------------------
    # Print key metrics
    # ---------------------------------------------------------
    print("\n=== Key Metrics ===")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")

    # Ultralytics .maps is indexed by CLASS, not COCO box-area category.
    if hasattr(metrics.box, "maps"):
        print("\n=== Per-class AP@0.5:0.95 (NOT AP by object size) ===")
        names = metrics.names
        for class_id, value in enumerate(metrics.box.maps):
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            print(f"AP_class[{class_id}] {label}: {float(value):.4f}")

if __name__ == "__main__":
    main()
