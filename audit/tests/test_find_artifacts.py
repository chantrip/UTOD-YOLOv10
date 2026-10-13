"""Read-only artifact discovery tests using synthetic HPC training runs."""
from pathlib import Path
import importlib.util
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def load_locator():
    filename = ROOT / "audit/find_artifacts.py"
    spec = importlib.util.spec_from_file_location("artifact_locator", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(root: Path, name: str):
    run = root / "runs" / "detect" / name
    (run / "weights").mkdir(parents=True)
    (run / "args.yaml").write_text(
        "model: yolov10n.yaml\nseed: 42\nname: " + name + "\n", encoding="utf-8"
    )
    (run / "results.csv").write_text("epoch,mAP50\n1,0.12\n", encoding="utf-8")
    (run / "weights/best.pt").write_bytes(b"TEST_BINARY_DO_NOT_COPY")
    (run / "weights/last.pt").write_bytes(b"TEST_LAST_BINARY")
    return run


def test_locator_detects_runs_and_train_args(tmp_path):
    mod = load_locator()
    first = make_run(tmp_path, "ruod_bs_v2")
    make_run(tmp_path, "ruod_dbw_v12")
    rows = mod.scan(tmp_path)
    assert len(rows) == 2
    assert rows[0]["has_args_yaml"] and rows[0]["has_results_csv"]
    assert rows[0]["args"]["model"] == "yolov10n.yaml"
    assert rows[0]["args"]["seed"] == "42"
    assert set(rows[0]["checkpoints"]) == {"best.pt", "last.pt"}
    assert (first / "weights/best.pt").read_bytes() == b"TEST_BINARY_DO_NOT_COPY"


def test_metadata_archive_excludes_weights(tmp_path):
    mod = load_locator()
    make_run(tmp_path, "ruod_bs_v2")
    rows = mod.scan(tmp_path)
    outfile = tmp_path / "metadata.zip"
    count = mod.bundle_metadata(rows, outfile)
    assert count == 2
    with zipfile.ZipFile(outfile) as archive:
        filenames = archive.namelist()
        assert any(n.endswith("/args.yaml") for n in filenames)
        assert any(n.endswith("/results.csv") for n in filenames)
        assert "RUN_INDEX.json" in filenames
        assert all(not n.endswith((".pt", ".png", ".jpg")) for n in filenames)
        idx = json.loads(archive.read("RUN_INDEX.json"))
        assert len(idx) == 1


def test_missing_directory_finds_no_runs(tmp_path):
    mod = load_locator()
    assert mod.find_runs(tmp_path / "does_not_exist") == []
