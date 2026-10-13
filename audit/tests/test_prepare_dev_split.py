"""Synthetic-data tests for new reproducible development splits (no actual dataset required).

Run: python -m pytest -q audit/tests/test_prepare_dev_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "audit"))
from prepare_dev_split import create_split
from verify_splits import audit as audit_splits


def create_dataset(tmp_path: Path, grouped: bool = False) -> Path:
    train = tmp_path / "images/train"
    test = tmp_path / "images/test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    for i in range(20):
        stem = f"video{i // 5}_frame{i:03d}" if grouped else f"frame{i:03d}"
        (train / f"{stem}.png").write_bytes(f"TRAIN_IMAGE_{i}".encode())
    for i in range(4):
        (test / f"test_{i}.png").write_bytes(f"TEST_IMAGE_{i}".encode())
    y = tmp_path / "data.yaml"
    y.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train": "images/train", "val": "images/test",
        "names": ["object"],
    }), encoding="utf-8")
    return y


def test_split_preserves_official_test_and_is_deterministic(tmp_path):
    source = create_dataset(tmp_path)
    a = create_split(source, tmp_path / "out_a", 42, 0.15, True)
    b = create_split(source, tmp_path / "out_b", 42, 0.15, True)
    assert a["original_train_count"] == 20
    assert a["audit_train_count"] + a["audit_val_count"] == 20
    assert a["official_test_count"] == 4
    assert a["manifests_sha256"] == b["manifests_sha256"]
    assert all(count == 0 for count in a["resolved_path_overlap"].values())
    checked = audit_splits(tmp_path / "out_a/data_audit.yaml", None, skip_hash=False)
    assert all(z["identical_content_sha256_count"] == 0 for z in checked["comparisons"].values())
    assert not any("No independent" in warning for warning in checked["warnings"])


def test_grouped_sequences_do_not_cross_train_and_val(tmp_path):
    source = create_dataset(tmp_path, grouped=True)
    result = create_split(source, tmp_path / "grouped", 42, 0.15, True, group_regex=r"^(video\d+)_")
    train = (tmp_path / "grouped/train.txt").read_text().splitlines()
    val = (tmp_path / "grouped/val.txt").read_text().splitlines()
    train_groups = {Path(x).stem.split("_")[0] for x in train}
    val_groups = {Path(x).stem.split("_")[0] for x in val}
    assert not train_groups.intersection(val_groups)
    assert result["audit_val_count"] >= 3


def test_source_test_requires_explicit_remapping(tmp_path):
    source = create_dataset(tmp_path)
    with pytest.raises(ValueError, match="No official test"):
        create_split(source, tmp_path / "out", 42, 0.15, False)


def test_invalid_val_fraction_is_rejected(tmp_path):
    source = create_dataset(tmp_path)
    with pytest.raises(ValueError, match="val_fraction"):
        create_split(source, tmp_path / "out", 42, 0.9, True)
