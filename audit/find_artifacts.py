#!/usr/bin/env python3
"""Find YOLOv10 training artifacts on the original machine (read-only by default).

Examples:
  python audit/find_artifacts.py --root /home/dsai-st125287/project/UTOD-YOLOv10
  python audit/find_artifacts.py --root /home/dsai-st125287/project/UTOD-YOLOv10 --bundle /tmp/utod_run_metadata.zip

The optional metadata ZIP contains args.yaml, results.csv and small text logs only:
NO checkpoints, datasets, images, or private model weights are copied.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

DEFAULT_ROOT = Path("/home/dsai-st125287/project/UTOD-YOLOv10")
METADATA_NAMES = {"args.yaml", "results.csv"}
MAX_TEXT_FILE = 16 * 1024 * 1024


def search_base(root: Path) -> Path:
    root = root.expanduser().resolve()
    for relative in ("runs/detect", "runs"):
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    return root


def find_runs(root: Path) -> list[Path]:
    base = search_base(root)
    if not base.is_dir():
        return []
    found: set[Path] = set()
    for item in base.rglob("*"):
        if not item.is_file():
            continue
        if item.name in METADATA_NAMES:
            found.add(item.parent)
        elif item.name in ("best.pt", "last.pt") and item.parent.name == "weights":
            found.add(item.parent.parent)
    return sorted(found)


def important_args(path: Path) -> dict[str, str]:
    keys = {
        "model", "data", "epochs", "seed", "optimizer",
        "name", "batch", "imgsz", "pretrained", "box_loss",
    }
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([a-zA-Z_][a-zA-Z_0-9]*)\s*:\s*(.*?)\s*(?:#.*)?$", line)
        if match and match.group(1) in keys:
            values[match.group(1)] = match.group(2)
    return values


def small_text_artifacts(run: Path) -> list[Path]:
    result = [run / name for name in ("args.yaml", "results.csv")]
    for pattern in ("*.log", "*.txt"):
        result.extend(sorted(run.glob(pattern)))
    return [
        file for file in dict.fromkeys(result)
        if file.is_file() and not file.is_symlink() and file.stat().st_size <= MAX_TEXT_FILE
    ]


def scan(root: Path) -> list[dict]:
    rows = []
    for run in find_runs(root):
        weights = run / "weights"
        checkpoints = {
            file.name: {
                "path": str(file.resolve()), "bytes": file.stat().st_size,
            }
            for file in (weights / "best.pt", weights / "last.pt")
            if file.is_file()
        }
        rows.append({
            "run": str(run),
            "args": important_args(run / "args.yaml"),
            "has_args_yaml": (run / "args.yaml").is_file(),
            "has_results_csv": (run / "results.csv").is_file(),
            "checkpoints": checkpoints,
            "metadata_files": [str(p) for p in small_text_artifacts(run)],
        })
    return rows


def bundle_metadata(rows: list[dict], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as out:
        for number, row in enumerate(rows, 1):
            run = Path(row["run"])
            for file in small_text_artifacts(run):
                # Explicitly exclude model binaries, image files, TensorBoard data and datasets.
                out.write(file, arcname=f"run_{number:03d}_{run.name}/{file.name}")
                added += 1
        out.writestr("RUN_INDEX.json", json.dumps(rows, indent=2))
        out.writestr(
            "README.txt",
            "Training run metadata only: NO model checkpoints, datasets or images.\n"
            "Run names and args do not prove historical CIoU/WIoU code provenance.\n"
            "Never publish experiment artifacts or dataset paths without permission.\n",
        )
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bundle", type=Path,
                        help="Optional output ZIP of metadata only; no .pt checkpoints")
    parser.add_argument("--json", type=Path, help="Optional JSON index output")
    args = parser.parse_args()

    base = search_base(args.root)
    if not base.is_dir():
        parser.error(f"Directory unavailable: {base}. Run on the correct HPC account.")
    rows = scan(args.root)
    print(f"Search root: {base}\nTraining run folders found: {len(rows)}")
    for row in rows:
        print(f"\nRUN: {row['run']}")
        print("  args.yaml:", "found" if row["has_args_yaml"] else "missing")
        print("  results.csv:", "found" if row["has_results_csv"] else "missing")
        print("  checkpoints:", ", ".join(row["checkpoints"]) or "none")
        print("  model/seed/loss:", row["args"].get("model", "?"),
              row["args"].get("seed", "?"), row["args"].get("box_loss", "unrecorded"))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print("\nJSON index written:", args.json)
    if args.bundle:
        count = bundle_metadata(rows, args.bundle)
        print(f"\nMetadata ZIP: {args.bundle} ({count} small files; no checkpoints)")
    if not rows:
        print(
            "\nNo run artifacts found here. Search other HPC project directories, "
            "external tracking tools, or backup locations. GitHub does not "
            "contain the final underwater training runs."
        )


if __name__ == "__main__":
    main()
