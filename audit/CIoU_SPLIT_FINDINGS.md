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


## Additional independent verification: five pairs of training scripts

A direct line-by-line comparison of the **checked-in source at the audit base commit** finds that each paired baseline and WIoU training script invokes the **same detector configuration**, **same dataset**, **same train() hyperparameters**, and **same shared loss code**:

| Paired scripts | Differences in checked-in source |
|---|---|
| `trash_m_baseline.py` / `trash_m_wiou.py` | Experiment `name` only |
| `trash_i_baseline.py` / `trash_i_wiou.py` | Header comment and experiment `name` only |
| `ruod_baseline.py` / `ruod_wiou.py` | Header comment and experiment `name` only |
| `duo_baseline.py` / `duo_wiou.py` | Header comment and experiment `name` only |
| `uodd_baseline.py` / `uodd_wiou.py` | Header comment and experiment `name` only |

**Inference (conditional on same code revision and environment):** Replaying each pair from the current checked-in repository **does not isolate the CIoU-vs-WIoU variable**. All ten scripts would use the globally hardcoded WIoU loss. Running them can yield different output metrics because of randomness, but **not because the scripts explicitly select different localization losses**.

This finding **does not establish which historical source snapshots were actually executed**. The TrashCan-Material baseline script already appeared in the Oct 20, 2025 history when CIoU was in force, whereas many other experiment scripts were only added to Git on April 7, 2026, after WIoU was checked in; local training versions might have differed. The thesis is dated April 2026 and does not identify commit versions for experiments.

The one-factor WIoU ablation is central to the paper: manuscript Table 2 reports WIoU-only mAP@0.5 values 0.609 vs 0.577 on TrashCan-Material and 0.658 vs 0.615 on TrashCan-Instance. These are **reported experimental observations**, not validated reproduced improvements, until the code-era/experiment provenance is established.

## Independent source cross-check on benchmark definitions

- The [original DUO authors' repository](https://github.com/chongweiliu/DUO) explicitly says missing annotated test sets and inconsistent self-created test splits were problems the benchmark sought to address. Reusing DUO's annotated test partition for `best.pt` selection would undermine the independent benchmark comparison; it should be treated as a *development/selection set* if used that way.
- [RUOD-R's documentation](https://github.com/RSSL-MTU/RUOD-R) states the original RUOD train/test image IDs and split definitions remain unchanged in the re-annotation. This confirms a recognized benchmark train/test distinction; it does not establish how the student's local files were constructed.

## Verification of the read-only split checker

The local Python implementation of `audit/verify_splits.py` was exercised with **three synthetic fixtures**, not the original datasets:

1. Identical image bytes under different filenames across train and val were detected by SHA-256.
2. A RUOD/DUO-style `val: images/test` mapping with no `test:` entry was flagged.
3. Distinct-image train/val/test partitions yielded zero hash overlap.

All three checks passed in the local test environment. **No real dataset counts, hash comparisons, loss-isolation reruns, or AP scores have been generated.** Note that hash comparison cannot detect visually duplicated-but-recompressed scenes, nor can it prove independence of model selection; run records remain essential.

## Strongest achievable conclusion from the available public evidence

- The present repository is **insufficient to independently reproduce the claimed CIoU-vs-WIoU baseline/ablation**.
- The present code **would use an intended benchmark test directory for checkpoint selection** for both RUOD and DUO if the checked-in configs were used unchanged.
- There is **not enough evidence to assert that the historical reported accuracy numbers are false**; old code, local patches, unpublished run settings, or differently defined file partitions remain possible.
- No actual benchmark re-evaluation is possible from this repository alone because the required datasets and model checkpoints are not included.

