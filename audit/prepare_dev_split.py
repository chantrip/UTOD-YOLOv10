#!/usr/bin/env python3
"""Create an independent development-validation subset from official training images.

This utility DOES NOT move/copy/edit the original images or labels. It produces:
  train.txt, val.txt, data_audit.yaml, split_provenance.json

For official two-way RUOD/DUO train/test layouts, use --val-is-test.
WARNING: Existing test-set reuse in historical model selection cannot be
retroactively undone by this tool. New training/evaluation runs must avoid
test-guided tuning. For adjacent video frames, supply --group-regex to keep
the same scene/sequence in one split.

Example (run on HPC after inspecting the source YAML):
  python audit/prepare_dev_split.py --data data/ruod.yaml \
      --val-is-test --outdir /tmp/ruod_dev_s42 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

from verify_splits import images_at, resolve_split


def create_split(
    data_yaml: Path,
    outdir: Path,
    seed: int,
    val_fraction: float,
    val_is_test: bool,
    group_regex: str | None = None,
    root_override: Path | None = None,
) -> dict:
    if not (0 < val_fraction < 0.5):
        raise ValueError("val_fraction must be between 0 and 0.5 (exclusive)")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not cfg.get("train"):
        raise ValueError("Dataset YAML must provide a training split")
    raw_root = Path(str(cfg.get("path", data_yaml.parent)))
    root = (root_override or (
        raw_root if raw_root.is_absolute() else data_yaml.resolve().parent / raw_root
    )).resolve()
    train_sources = cfg["train"] if isinstance(cfg["train"], list) else [cfg["train"]]
    train_images = sorted({p for entry in train_sources
                           for p in images_at(resolve_split(entry, root), root)})
    if len(train_images) < 3:
        raise ValueError("Need at least three accessible training images")
    if val_is_test:
        if cfg.get("test"):
            raise ValueError("--val-is-test requested, but test: already exists; review splits")
        official_test = cfg.get("val")
    else:
        official_test = cfg.get("test")
    if not official_test:
        raise ValueError("No official test partition identified; do not invent one")
    test_sources = official_test if isinstance(official_test, list) else [official_test]
    test_paths = [resolve_split(x, root) for x in test_sources]
    if not all(x.exists() for x in test_paths):
        raise FileNotFoundError("Official test partition unavailable: " + str(test_paths))
    test_images = {
        p for location in test_paths for p in images_at(location, root)
    }
    if not test_images:
        raise ValueError("Official test partition has no accessible images")
    if set(train_images) & test_images:
        raise ValueError("Train and official test partition contain overlapping image paths")

    by_group: dict[str, list[Path]] = {}
    rx = re.compile(group_regex) if group_regex else None
    for image in train_images:
        if rx:
            m = rx.search(image.stem)
            if m is None:
                raise ValueError(f"Grouping regex did not match {image.name}")
            key = m.group(1) if m.groups() else m.group()
        else:
            key = str(image)
        by_group.setdefault(key, []).append(image)

    groups = sorted(by_group)
    if len(groups) < 2:
        raise ValueError("Too few distinct groups to construct an independent validation subset")
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(train_images) * val_fraction))
    chosen = set()
    count = 0
    for group in groups[:-1]:
        if count >= target:
            break
        chosen.add(group)
        count += len(by_group[group])
    val_images = sorted(p for g in chosen for p in by_group[g])
    fit_images = sorted(p for g, ps in by_group.items() if g not in chosen for p in ps)
    if not fit_images or not val_images:
        raise ValueError("Could not form nonempty training and development-validation splits")

    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    for name, entries in (("train", fit_images), ("val", val_images)):
        (outdir / f"{name}.txt").write_text(
            "\n".join(str(p) for p in entries) + "\n", encoding="utf-8"
        )
    new_cfg = dict(cfg)
    new_cfg["path"] = str(root)
    new_cfg["train"] = str(outdir / "train.txt")
    new_cfg["val"] = str(outdir / "val.txt")
    new_cfg["test"] = [str(p) for p in test_paths] if len(test_paths) > 1 else str(test_paths[0])
    output_yaml = outdir / "data_audit.yaml"
    output_yaml.write_text(yaml.safe_dump(new_cfg, sort_keys=False), encoding="utf-8")

    overlap = {
        "train_val": len(set(fit_images) & set(val_images)),
        "train_test": len(set(fit_images) & test_images),
        "val_test": len(set(val_images) & test_images),
    }
    assert all(n == 0 for n in overlap.values())
    metadata = {
        "original_yaml": str(data_yaml.resolve()),
        "generated_yaml": str(output_yaml),
        "seed": seed, "val_fraction": val_fraction,
        "group_regex": group_regex,
        "caution": "Image-level splitting cannot guarantee independence of visually similar video frames. Test reuse from earlier experiments cannot be undone.",
        "original_train_count": len(train_images),
        "audit_train_count": len(fit_images),
        "audit_val_count": len(val_images),
        "official_test_count": len(test_images),
        "resolved_path_overlap": overlap,
        "manifests_sha256": {
            x: hashlib.sha256((outdir / f"{x}.txt").read_bytes()).hexdigest()
            for x in ("train", "val")
        },
    }
    (outdir / "split_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--val-is-test", action="store_true",
                   help="Source YAML's val points to the official benchmark test set")
    p.add_argument("--group-regex", help="Optional regex on stem; first capture group defines sequence ID")
    p.add_argument("--root", type=Path, help="Override source YAML dataset root")
    args = p.parse_args()
    report = create_split(
        args.data, args.outdir, args.seed, args.val_fraction,
        args.val_is_test, args.group_regex, args.root
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
