# Reproducibility audit workspace (CVC 2027)

This directory is the independent audit workspace on **`chantrip/UTOD-YOLOv10:cvc-audit`**. Do not merge it into the original model or use the rewritten code as evidence that past experiments were valid. The fork's `main` and the upstream student repository are unchanged.

**Source base:** `6a4617b19ebd587bdb514405879a9db048d226fa`.

## Findings and scope

- [CIoU baseline / dataset split evidence](CIoU_SPLIT_FINDINGS.md) — checked-in loss selected WIoU; October 2025 history includes genuine CIoU. Five current baseline/WIoU script pairs do **not** select different losses. Actual historical run provenance is unknown.
- [Manuscript source-to-code corrections](MANUSCRIPT_TRACEABILITY.md) — equation, metric, fusion, statistical, and deployment-claim traceability.
- [Verified audit status](STATUS.md) — CI results and empirical blockers.
- RUOD and DUO source YAMLs use `val: images/test` and the training loop selects `best.pt` using validation fitness. That is **not an independent test**, if training and reporting used the checked-in configuration.
- [Source/checklist](CVC_AUDIT_PLAN.md) — original claims require traceability to run artifacts.
- **No actual datasets or model checkpoints have been evaluated by this independent audit.** Numerical mAP claims remain unverified.

## Locate original HPC training artifacts

If authorized to access Bidhan's HPC project account/storage, run:

```bash
python audit/find_artifacts.py --root /home/dsai-st125287/project/UTOD-YOLOv10

# Optional metadata-only bundle for independent review:
python audit/find_artifacts.py --root /home/dsai-st125287/project/UTOD-YOLOv10 \
  --bundle /tmp/utod_run_metadata.zip
```

The locator prints every run that has an `args.yaml`, `results.csv`, or `weights/best.pt`/`last.pt`, including saved model/seed when recorded. The ZIP deliberately contains **only small run metadata and text logs**; it excludes `.pt` checkpoints, datasets, and images. Share the ZIP privately, not by committing it to the public fork.

Expected examples from existing hardcoded paths are `runs/detect/ruod_bs_v2` and `runs/detect/ruod_dbw_v12`. Those paths were found in source code only, not verified on HPC.

**Important:** Earlier runs did not explicitly record `box_loss`; absence of this key in an old `args.yaml` does not establish CIoU versus WIoU. The code snapshot and/or run log is still needed.

## Python tools (read original data; do not modify it)

Requires `PyYAML`. For regression tests, install `pytest` and a compatible CPU build of `torch`. Full training requires the original project's PyTorch/CUDA/DCNv4 environment.

1. Check content overlap and train/val/test paths:

   ```bash
   python audit/verify_splits.py --yaml data/ruod.yaml --output audit_ruod_existing.json
   python audit/verify_splits.py --yaml data/duo.yaml --output audit_duo_existing.json
   ```

   A matching SHA-256 across two partitions indicates duplicated bytes. Different hashes do **not** prove frame-level/video-level independence. The script cannot detect whether a test set was used to select `best.pt`; that must be traced from training code/logs.

2. Generate a new development-validation split **from official training images**, retaining the recognized official test set:

   ```bash
   python audit/prepare_dev_split.py --data data/ruod.yaml --val-is-test \
     --outdir /home/dsai-st125287/project/audit-splits/ruod_s42 --seed 42

   python audit/prepare_dev_split.py --data data/duo.yaml --val-is-test \
     --outdir /home/dsai-st125287/project/audit-splits/duo_s42 --seed 42
   ```

   This produces `train.txt`, `val.txt`, `data_audit.yaml`, and `split_provenance.json`. It does not move or copy original files. Use `--group-regex '^(sequenceID)_'` **only if an actual meaningful group identifier is present in filenames**; otherwise inspect the videos/image metadata for near-duplicate scene grouping. A random image-level split alone is not sufficient to rule out correlated frames. When a test partition was already consulted to design the model, its historical status as an untouched holdout cannot be restored.

3. Verify generated splits:

   ```bash
   python audit/verify_splits.py --yaml /home/dsai-st125287/project/audit-splits/ruod_s42/data_audit.yaml \
      --output /home/dsai-st125287/project/audit-splits/ruod_s42/overlap_report.json
   ```

4. Inspect planned matched baseline vs WIoU run configurations **without running training**:

   ```bash
   DATA=/home/dsai-st125287/project/audit-splits/ruod_s42/data_audit.yaml
   python audit/train_controlled.py --data "$DATA" --model yolov10n.yaml \
      --box-loss ciou --name ruod_baseline_ciou_s42 --seed 42 --dry-run
   python audit/train_controlled.py --data "$DATA" --model yolov10n.yaml \
      --box-loss wiou --name ruod_baseline_wiou_s42 --seed 42 --dry-run
   ```

   Remove `--dry-run` only in a valid configured GPU environment with the actual data. The default training hyperparameters match the main source scripts (SGD, 200 epochs, batch 32, image size 640, same augmentation). `box_loss` is preserved in generated `args.yaml` and checkpoint training arguments through Ultralytics' configuration. **Do not compare this new training with numbers from the original paper as if protocols were identical**: new val/test splits can change checkpoint selection and reported AP. For fair new runs, repeat *both* baseline and proposed configurations under precisely the same corrected protocol.

5. Run data-free unit tests:

   ```bash
   python -m pytest -q audit/tests
   ```

   The regression-loss and fusion tests AST-load the actual source classes/functions without importing the entire Ultralytics package/DCNv4 CUDA extension. They check criterion selection for both YOLOv10 branches, numeric behavior, and that CIoU does not update WIoU running statistics. The partition tests use synthetic tiny image files to test deterministic generation, grouped sequences, and overlap checks. **These unit tests are not substitutes for a GPU end-to-end validation run.**

## Matched factorial ablations

See [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) for eight controlled configurations using four YAMLs and the explicit loss selector. The new `yolov10nP5DCN.yaml` changes **only the final backbone stage**, unlike the original four-stage DCN ablation. This is a protocol for future reruns, not newly measured accuracy.

To evaluate a separate final test partition after all model selection is frozen, pass the generated YAML directly: `python vali.py --data /home/dsai-st125287/project/audit-splits/ruod_s42/data_audit.yaml --weights /path/to/best.pt --name ruod_final_test --split test`. This requires an actual `test:` entry. Results from `--split val` remain development/model-selection metrics. Because the original benchmark test set may already have been consulted historically, new splitting does not retroactively make the old experiments independent.

## Commit-level change log

- Original loss selection was global WIoU. The audit branch adds `box_loss: wiou` to `ultralytics/cfg/default.yaml` and supports `box_loss=ciou` explicitly in `BboxLoss` through the existing `model.args` mechanism.
- `bbox_iou` now updates WIoU's running mean only for WIoU calls; the CIoU branch should leave its state untouched.
- The upstream scripts and models have **not** been rewritten; new experimental executions should use the controlled audit runner so each loss choice is recorded.
- All changes are specific to `cvc-audit`. Run `git diff main...cvc-audit` on your checkout to inspect them.

## Known remaining limitations

- The original HPC datasets, image manifests, trained checkpoints, and `results.csv`/run `args.yaml` files are not included in the publicly available repository.
- Historical CIoU vs WIoU experimental provenance cannot be recovered from script names alone.
- The repo's DCNv4-only architecture YAML places DC2f at more stages than the full final model. A well-controlled placement comparison should be designed before assigning each component a causal contribution.
- The script `vali.py` prints per-class AP values as `AP_small`, `AP_medium`, and `AP_large`; those are not object-size categories. That analysis must be corrected in later work.
- The code calls feature-weighted concatenation `BiFPN`; that terminology and architecture need a separate method audit.
