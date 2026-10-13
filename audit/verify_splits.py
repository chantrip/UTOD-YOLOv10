#!/usr/bin/env python3
"""Read-only YOLO dataset-split audit: duplicate content and split provenance.

Usage:
    python audit/verify_splits.py --yaml data/ruod.yaml --root /path/to/RUOD_YOLO
    python audit/verify_splits.py --yaml data/duo.yaml --root /path/to/DUO --output /tmp/duo_split_audit.json

Requires PyYAML. Does not modify images, annotations, or YAML files.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def images_at(location: Path, root: Path) -> list[Path]:
    if location.is_dir():
        return sorted(p.resolve() for p in location.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if location.is_file() and location.suffix.lower() in IMAGE_EXTS:
        return [location.resolve()]
    if location.is_file() and location.suffix.lower() == ".txt":
        items = []
        for raw in location.read_text(encoding="utf-8").splitlines():
            value = raw.strip()
            if not value:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                basedir = location.parent if value.startswith("./") else root
                candidate = basedir / candidate
            items.append(candidate.resolve())
        return items
    return []


def resolve_split(raw: str, root: Path) -> Path:
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def audit(yaml_path: Path, root_override: Path | None, skip_hash: bool) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Expected a YOLO dataset YAML mapping")
    if root_override is not None:
        root = root_override.resolve()
    else:
        raw_root = cfg.get("path", ".")
        path = Path(str(raw_root))
        root = path.resolve() if path.is_absolute() else (yaml_path.parent / path).resolve()

    report: dict[str, Any] = {
        "yaml": str(yaml_path.resolve()), "root": str(root),
        "hashing_enabled": not skip_hash, "splits": {}, "comparisons": {}, "warnings": [],
    }
    split_files: dict[str, list[Path]] = {}
    for split in SPLITS:
        sources = cfg.get(split)
        if not sources:
            report["splits"][split] = {"declared": False, "image_count": None}
            continue
        if isinstance(sources, str):
            sources = [sources]
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ValueError(f"Invalid {split} declaration: expected path or list of paths")
        candidates = [resolve_split(s, root) for s in sources]
        images = list(itertools.chain.from_iterable(images_at(p, root) for p in candidates))
        found = [p for p in candidates if p.exists()]
        missing_files = [p for p in images if not p.is_file()]
        split_files[split] = [p for p in images if p.is_file()]
        report["splits"][split] = {
            "declared": True, "sources": [str(p) for p in candidates],
            "present_sources": len(found), "image_count": len(images),
            "missing_image_count": len(missing_files),
            "missing_image_examples": [str(p) for p in missing_files[:10]],
        }
        if len(found) != len(candidates):
            report["warnings"].append(f"{split}: {len(candidates) - len(found)} declared source(s) unavailable")
        if split == "val" and any("test" in p.parts for p in candidates):
            report["warnings"].append("val: directory has a component literally named 'test'; verify final evaluation independence")
        if not images:
            report["warnings"].append(f"{split}: no images resolved")

    paths_hash: dict[Path, str] = {}
    if not skip_hash:
        unique = set(itertools.chain.from_iterable(split_files.values()))
        for p in sorted(unique):
            paths_hash[p] = digest(p)

    for left, right in itertools.combinations(split_files, 2):
        a, b = split_files[left], split_files[right]
        shared_paths = set(a) & set(b)
        duplicate_names = {p.name.casefold() for p in a} & {p.name.casefold() for p in b}
        if skip_hash:
            duplicate_hashes = None
        else:
            duplicate_hashes = {paths_hash[p] for p in a} & {paths_hash[p] for p in b}
        report["comparisons"][f"{left}_vs_{right}"] = {
            "shared_resolved_paths": len(shared_paths),
            "shared_path_examples": [str(p) for p in sorted(shared_paths)[:10]],
            "same_basename_count": len(duplicate_names),
            "same_basename_examples": sorted(duplicate_names)[:10],
            "identical_content_sha256_count": len(duplicate_hashes) if duplicate_hashes is not None else None,
            "identical_sha256_examples": sorted(duplicate_hashes)[:10] if duplicate_hashes is not None else [],
        }
        if shared_paths or duplicate_hashes:
            report["warnings"].append(f"{left}/{right}: identical files detected across splits; investigate overlap")
        elif duplicate_names:
            report["warnings"].append(f"{left}/{right}: repeated image names (not proof of duplicate content)")
    if "test" not in split_files:
        report["warnings"].append("No independent 'test:' split declared in dataset YAML")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", required=True, type=Path, help="Dataset YAML config")
    p.add_argument("--root", type=Path, help="Override absolute server path in dataset YAML")
    p.add_argument("--skip-hash", action="store_true", help="Skip SHA-256 duplicate-content checking")
    p.add_argument("--output", type=Path, help="Optional JSON report path")
    args = p.parse_args()
    result = audit(args.yaml, args.root, args.skip_hash)
    formatted = json.dumps(result, indent=2)
    print(formatted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
