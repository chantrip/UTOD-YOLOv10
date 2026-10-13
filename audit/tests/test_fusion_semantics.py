"""Characterization tests for the checked-in BiFPN-named operator, not full BiFPN.

Run: python -m pytest -q audit/tests/test_fusion_semantics.py
These tests intentionally document original semantics without altering the model.
"""
import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def load_original_bifpn():
    path = ROOT / "ultralytics/nn/bifpn.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "BiFPN")
    ns = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    return ns["BiFPN"]


def test_bifpn_named_module_is_weighted_concat_not_weighted_sum():
    BiFPN = load_original_bifpn()
    fusion = BiFPN(dimension=1)
    left = torch.ones(1, 2, 3, 3)
    right = torch.full((1, 2, 3, 3), 2.0)
    result = fusion([left, right])
    assert tuple(result.shape) == (1, 4, 3, 3)  # 2+2 channels: concatenation
    expected = torch.cat([left / 3.0001, right / 3.0001], dim=1)
    assert torch.allclose(result, expected)


def test_third_weight_affects_normalization_not_a_third_input():
    BiFPN = load_original_bifpn()
    fusion = BiFPN(dimension=1)
    x = [torch.ones(1, 2, 2, 2), torch.ones(1, 3, 2, 2)]
    initial = fusion(x)
    assert len(fusion.w) == 3  # third weight exists even with only two inputs
    with torch.no_grad():
        fusion.w[2] = 2.0
    later = fusion(x)
    assert tuple(later.shape) == (1, 5, 2, 2)
    assert not torch.allclose(initial, later)  # indirect denominator effect
