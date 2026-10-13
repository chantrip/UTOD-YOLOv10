"""Static tests: audit architecture factors must change only specified YAML locations.

Run: python -m pytest -q audit/tests/test_architecture_factors.py
No GPU, custom DCNv4 library, or original datasets are required.
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v10"


def model(name):
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def changes(source, target):
    assert set(source) == set(target)
    assert len(source["backbone"]) == len(target["backbone"])
    assert len(source["head"]) == len(target["head"])
    return {
        group: [i for i, (a, b) in enumerate(zip(source[group], target[group])) if a != b]
        for group in ("backbone", "head")
    }


def test_single_site_dc2f_ablation():
    base = model("yolov10n.yaml")
    p5_only = model("yolov10nP5DCN.yaml")
    assert changes(base, p5_only) == {"backbone": [8], "head": []}
    assert base["backbone"][8][2] == "C2f"
    assert p5_only["backbone"][8][2] == "DC2f"


def test_weighted_concats_match_final_locations():
    base = model("yolov10n.yaml")
    weighted_cat = model("yolov10nBi.yaml")
    assert changes(base, weighted_cat) == {"backbone": [], "head": [4, 7, 10]}
    assert [weighted_cat["head"][i][2] for i in (4, 7, 10)] == ["BiFPN"] * 3


def test_final_is_exact_factor_combination():
    base = model("yolov10n.yaml")
    p5_only = model("yolov10nP5DCN.yaml")
    weighted_cat = model("yolov10nBi.yaml")
    final = model("yolov10n-dwb.yaml")
    assert changes(p5_only, final) == {"backbone": [], "head": [4, 7, 10]}
    assert changes(weighted_cat, final) == {"backbone": [8], "head": []}
    assert changes(base, final) == {"backbone": [8], "head": [4, 7, 10]}


def test_prior_four_stage_dcn_is_not_p5_only():
    original_dcn = model("yolov10nDCN.yaml")
    baseline = model("yolov10n.yaml")
    assert changes(baseline, original_dcn) == {"backbone": [2, 4, 6, 8], "head": []}
