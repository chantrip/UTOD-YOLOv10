# CVC 2027 audit: CIoU baseline and dataset-split provenance

**Scope:** Static inspection of the fork at upstream base commit `6a4617b19ebd587bdb514405879a9db048d226fa`, earlier upstream commits, the thesis/method manuscript, and the checked-in training/validation code. No original datasets, logs, or model checkpoints have been examined. **Do not interpret the risks below as proof of invalid historical experiments.**

## Executive status

| Check | Source-supported observation | Status |
|---|---|---|
| Historical CIoU implementation | `BboxLoss.forward` used `CIoU=True` in commit `f682bb1891f99aed0a5ca36e1053da94f1bf4f6f` (2025-10-20). Commit `361ccdca9733ed845ae5eea697dcf6237fb987bb` (2025-10-23) replaced it with `WIoU=True`. | **Confirmed** |
| Loss selected by current baseline scripts | `trash_m_baseline.py`, `trash_i_baseline.py`, `ruod_baseline.py`, `duo_baseline.py`, etc. use baseline architecture `yolov10n.yaml`, but the checked-in shared `ultralytics/utils/loss.py` uses `WIoU=True` without a CLI loss selector. In YOLOv10, `v10DetectLoss` creates two `v8DetectionLoss` branches, both using `BboxLoss`. | **Confirmed for current code; historical training unverified** |
| RUOD split mapping | `data/ruod.yaml` sets `val: images/test` and has no active `test:` mapping. | **Confirmed** |
| DUO split mapping | `data/duo.yaml` sets `val: images/test` and has no active `test:` mapping. | **Confirmed** |
| Checkpoint-selection dependence | `BaseTrainer.get_dataset` loads `data['val']`; `BaseTrainer.validate` calculates fitness; `save_model` saves `best.pt` when fitness is best. The default config enables validation at each epoch (`val: True`, `val_period: 1`). Thus, **if those YAMLs and this training pipeline were used**, `images/test` affected best-checkpoint selection. | **Confirmed code path; exact run provenance unverified** |
| TrashCan-Material / TrashCan-Instance | Both YAMLs define `train` and `val`, but no active `test`. | **Confirmed** |
| UODD | Defines distinct `train`, `val`, `test` entries; however, checked-in `vali.py` explicitly evaluates `split='val'` for all datasets. | **Confirmed code paths; which split produced manuscript values unverified** |
| Cross-split duplicate images | No local image manifests or original dataset bytes available in checked-in source. | **Not tested** |

## Evidence links

- CIoU-era source: [loss.py at f682bb1](https://github.com/chantrip/UTOD-YOLOv10/blob/f682bb1891f99aed0a5ca36e1053da94f1bf4f6f/ultralytics/utils/loss.py#L63-L86)
- Exact loss switch: [commit 361ccdc](https://github.com/chantrip/UTOD-YOLOv10/commit/361ccdca9733ed845ae5eea697dcf6237fb987bb)
- Current WIoU loss: [loss.py](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/ultralytics/utils/loss.py#L63-L108) and [v10DetectLoss](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/ultralytics/utils/loss.py#L741-L751)
- [RUOD YAML](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/data/ruod.yaml) and [DUO YAML](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/data/duo.yaml)
- Checkpoint selection: [trainer.get_dataset](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/ultralytics/engine/trainer.py#L508-L516), [trainer.validate](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/ultralytics/engine/trainer.py#L546-L556), [trainer.save_model](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/ultralytics/engine/trainer.py#L479-L505), [default validation config](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/ultralytics/cfg/default.yaml#L46-L47)
- [vali.py](https://github.com/chantrip/UTOD-YOLOv10/blob/cvc-audit/vali.py#L82-L94) uses `split='val'`.

## Interpretation

The original CIoU code in Git history makes a legitimate CIoU baseline **possible**. It does **not** prove that every baseline in the paper was trained with it. Specifically, RUOD/DUO scripts were added in a later commit while the checked-in shared loss was WIoU; a script named `*_baseline.py` denotes the *architecture*, not an independently selected loss. The manuscript's CIoU-vs-WIoU comparison requires experiment-level provenance.

For RUOD and DUO, if `images/test` served for epoch-by-epoch checkpoint selection and also for the paper's final figures, the paper should not describe those figures as independent held-out test results. This is **selection-set reuse**, even if there is no duplicate image between train and test directories. Whether an untouched alternative test set is available is currently unknown.

For TrashCan, the repository shows train and validation, but not an independent test. For UODD, a separate test is configured but the generic checked-in validation script uses val. Results may be valid validation-set results; relabeling them as test results needs corroborating logs.

## Priority evidence request for Bidhan

For **each dataset** and **each claimed baseline/ablation**, request:

1. The actual experiment directory name, command line, `args.yaml`, `results.csv`, log or WandB run ID, timestamp, and the code commit/worktree snapshot (including `ultralytics/utils/loss.py`).
2. The `best.pt` file or securely shared checkpoint metadata: epoch, run arguments, training date, best fitness, and metric history. **Caution:** `best.pt` contains saved model objects; checkpoint metadata by itself does not reliably identify CIoU vs WIoU. Load only trusted checkpoints and do not run arbitrary untrusted serialized code.
3. Dataset YAML **as used at run time**, resolved absolute paths, number of images per split, exact image-ID manifests, and, if possible, file hashes.
4. Which split selected the best epoch, which split was used for final mAP, whether the same results were consulted during architecture/hyperparameter selection, and whether an independent untouched holdout exists.
5. For RUOD/DUO, explicitly explain why `images/test` was mapped to `val`. For UODD, show the `split='test'` evaluation logs if the reported values are claimed to be test results.

## Decision rule before submission

- **If baseline CIoU logs/checkpoints and independent evaluation are documented:** preserve original experiments and add concise reproducibility details, checkpoint-to-result mapping, and correct split terminology.
- **If a nominal CIoU baseline actually used WIoU:** the claimed WIoU-only ablation is confounded. Re-run the affected baseline and comparisons with explicit versioned loss selection; update all affected tables. Do not relabel a WIoU result as CIoU.
- **If a nominal final test partition was used to select checkpoints:** do not claim independent test performance on it. For a proper final comparison, establish a separate development validation set and an untouched test set before retraining/reselecting; ideally use a genuinely unseen external holdout if the original test set has already influenced model design.
- **If data or time prevent independent re-evaluation:** state exactly what is validation-set performance, do not claim an untouched holdout, and avoid unsubstantiated benchmark generalization conclusions.

**Next technical step:** Run a read-only image-manifest and duplicate-content audit against the real HPC dataset roots. Comparing names and SHA-256 hashes across splits can detect overlap, but **cannot by itself detect reuse of test labels for model selection**; the YAMLs plus training logs are essential.
