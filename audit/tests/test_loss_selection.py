"""Data-free tests for the loss-selection audit.

The source classes/functions are extracted by AST from the actual repository files.
This avoids importing the full Ultralytics package or loading the custom DCNv4 CUDA
extension for simple unit tests; it does not replace an end-to-end GPU training test.

Run: python -m pytest -q audit/tests/test_loss_selection.py
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]


def extract(filename: str, *definitions: str, extra: dict | None = None) -> dict:
    """Execute exact unmodified class/func definitions from the checked-in source."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    wanted = [
        n for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in definitions
    ]
    assert {n.name for n in wanted} == set(definitions)
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
    if extra:
        namespace.update(extra)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.fixture
def metrics():
    return extract("ultralytics/utils/metrics.py", "WIoU_Scale", "bbox_iou")


@pytest.fixture
def losses(metrics):
    def fake_assigner(**kwargs):
        return SimpleNamespace(**kwargs)

    return extract(
        "ultralytics/utils/loss.py",
        "BboxLoss", "v8DetectionLoss", "v10DetectLoss",
        extra={"bbox_iou": metrics["bbox_iou"], "TaskAlignedAssigner": fake_assigner},
    )


def make_inputs():
    return (
        torch.zeros((1, 1, 4)),
        torch.tensor([[[0.1, 0.1, 0.8, 0.8]]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[[0.2, 0.2, 0.9, 0.9]]], dtype=torch.float32),
        torch.ones((1, 1, 1)),
        torch.tensor(1.0),
        torch.tensor([[True]]),
    )


def test_real_bbox_loss_selects_ciou_or_wiou(losses, metrics):
    cls = losses["BboxLoss"]
    box_args = make_inputs()
    mean = metrics["WIoU_Scale"]
    mean.iou_mean = 1.0
    mean._is_train = True

    ciou_loss, ciou_dfl = cls(reg_max=15, use_dfl=False, box_loss="ciou")(*box_args)
    assert torch.isfinite(ciou_loss) and ciou_loss.ndim == 0
    assert ciou_dfl.item() == 0
    assert mean.iou_mean == 1.0  # CIoU must NOT update global WIoU statistics

    wiou_loss, wiou_dfl = cls(reg_max=15, use_dfl=False, box_loss="wiou")(*box_args)
    assert torch.isfinite(wiou_loss) and wiou_loss.ndim == 0
    assert wiou_dfl.item() == 0
    assert mean.iou_mean != 1.0
    assert not torch.allclose(ciou_loss, wiou_loss)


def test_unsupported_loss_is_rejected(losses):
    with pytest.raises(ValueError, match="Unsupported box_loss"):
        losses["BboxLoss"](reg_max=15, box_loss="typo")


@pytest.mark.parametrize("chosen", ["ciou", "wiou"])
def test_both_yolov10_branches_receive_explicit_loss(losses, chosen):
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.args = SimpleNamespace(box_loss=chosen)
            self.model = [SimpleNamespace(reg_max=16, stride=torch.tensor([8, 16, 32]), nc=1, no=65)]

    criterion = losses["v10DetectLoss"](FakeModel())
    assert criterion.one2many.bbox_loss.box_loss == chosen
    assert criterion.one2one.bbox_loss.box_loss == chosen


def test_default_training_configuration_declares_loss():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text())
    assert cfg["box_loss"] == "wiou"


def test_controlled_runner_refuses_missing_test(tmp_path):
    yaml = pytest.importorskip("yaml")
    runner = extract  # avoid importing Ultralytics or torch extension from the runner
    import importlib.util
    filename = ROOT / "audit/train_controlled.py"
    spec = importlib.util.spec_from_file_location("audit_train_controlled", filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dataset = tmp_path / "ruod.yaml"
    dataset.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train": "images/train", "val": "images/test",
    }), encoding="utf-8")
    info = module.inspect_data_yaml(dataset)
    assert len(info["development_only_reasons"]) == 2

    safe = tmp_path / "uodd.yaml"
    safe.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train": "images/train",
        "val": "images/val", "test": "images/test",
    }), encoding="utf-8")
    assert module.inspect_data_yaml(safe)["development_only_reasons"] == []

    bad = tmp_path / "same.yaml"
    bad.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train": "images/train", "val": "images/train",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        module.inspect_data_yaml(bad)
