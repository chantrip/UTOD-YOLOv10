# CVC 2027 — Independent Reproducibility Audit

## Scope and provenance

- **Upstream repository:** [Bidhan-Bajracharya/UTOD-YOLOv10](https://github.com/Bidhan-Bajracharya/UTOD-YOLOv10)
- **Audit fork:** [chantrip/UTOD-YOLOv10](https://github.com/chantrip/UTOD-YOLOv10)
- **Base commit:** `6a4617b19ebd587bdb514405879a9db048d226fa`
- **Working branch:** `cvc-audit`
- **Purpose:** Reproducibility and implementation-to-manuscript alignment review for the Computer Vision Conference (CVC) submission.

This branch is an independent working area. Do not change upstream code or the fork's `main` branch. Preserve original behavior as a reference; isolate each experimental change and label unverified results clearly.

## Audit workstreams

1. **Model definition:** Confirm the exact architecture configured by `ultralytics/cfg/models/v10/yolov10n-dwb.yaml`, especially DC2f placement and the implementation/terminology of fusion modules in `ultralytics/nn/bifpn.py`.
2. **Regression-loss isolation:** Trace the full loss call path in `ultralytics/utils/loss.py` and `ultralytics/utils/metrics.py`. Verify baseline and ablation configurations, checkpoints, commits, and logs without assuming the current code describes historical training runs.
3. **Dataset protocol:** Reconstruct train/validation/test image IDs and selection procedure for TrashCan-Material, TrashCan-Instance, RUOD, DUO, and UODD. Preserve manifests; confirm that final test data was not used for checkpoint or hyperparameter selection.
4. **Evaluation metrics:** Verify mAP IoU ranges and AP-by-object-size computation against the actual validation implementation. Distinguish per-class AP from per-size AP.
5. **Efficiency:** Clearly separate synthetic GPU forward-pass latency from real-image end-to-end inference and genuine edge-hardware measurements. Use identical hardware/software settings for all comparisons.
6. **Experimental controls:** Compare matched backbones, losses, augmentations, schedules, checkpoints, and random seeds. Calculate variability and paired differences where supported.
7. **Manuscript traceability:** Link every paper table, figure, metric, and reproducibility statement to evidence artifacts and an exact Git revision.

## Proposed initial checks (no new experimental claims)

- Unit-test feature-fusion shape behavior and weight normalization.
- Add an explicit, testable way to select the localization loss without modifying the historical reference in place.
- Audit dataset YAML paths and image/label manifests before reusing published performance numbers.
- Verify metric calculations against known simple inputs.
- Record inference batch size, image size, precision, GPU, warmup, timed operations, and software versions.

## Evidence needed from the original training environment

- Exact code commits/configurations used for **each** baseline and ablation.
- Dataset split manifests or image IDs, plus dataset acquisition/annotation versions.
- Training logs, `best.pt` and/or `last.pt` checkpoints, seeds, selection criteria, and validation outputs.
- Dependency versions, CUDA/torch environment, and compiled DCNv4 build requirements.
- Scripts and raw outputs for every table in the manuscript.

## Working rules

- Do not invent missing measurements or silently overwrite published results.
- Treat gaps as **unverified**, not automatically as evidence of invalid experiments.
- Perform corrected trials in separate branches or commits; retain the base commit for comparison.
- Keep preliminary manuscript criticisms private until the authors have reviewed and verified them.
