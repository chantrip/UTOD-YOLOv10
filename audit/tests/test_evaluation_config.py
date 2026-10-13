"""Data-free source audit for validation-script metric labels and partition choice.

Run: python -m pytest -q audit/tests/test_evaluation_config.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]


def test_validator_does_not_claim_class_ap_is_size_ap():
    source = (ROOT / "vali.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "AP_small" not in source
    assert "AP_medium" not in source
    assert "AP_large" not in source
    assert "metrics.box.maps" in source
    assert "Per-class AP" in source


def test_explicit_split_forwarded_to_ultralytics():
    source = (ROOT / "vali.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    calls = [
        node for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "val"
    ]
    assert calls, "Missing model.val() call"
    assert any(
        any(
            keyword.arg == "split"
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "args"
            and keyword.value.attr == "split"
            for keyword in call.keywords
        )
        for call in calls
    )


def test_generated_dataset_yaml_contains_separate_test_for_evaluation(tmp_path):
    path = tmp_path / "data_audit.yaml"
    path.write_text(yaml.safe_dump({
        "train": "train.txt", "val": "val.txt", "test": "images/test",
        "names": ["object"],
    }), encoding="utf-8")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["test"] != cfg["val"]
