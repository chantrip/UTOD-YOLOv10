# CVC audit status (2026-10-13)

## Verified code audit state

- Fork: [chantrip/UTOD-YOLOv10](https://github.com/chantrip/UTOD-YOLOv10)
- Audit branch: [cvc-audit](https://github.com/chantrip/UTOD-YOLOv10/tree/cvc-audit)
- Original student source base: `6a4617b19ebd587bdb514405879a9db048d226fa`.
- Data-free CI: [GitHub Actions run 38345901087](https://github.com/chantrip/UTOD-YOLOv10/actions/runs/38345901087) — **19 tests passed** at commit `338117d4f2e4f8a290024d1a5a63fc45d3ddf686`. No failed or skipped tests reported.
- Testing covered extracted real source-code loss selection and model branches, YAML architecture factor isolation, synthetic duplicate detection and split generation, and metric/split reporting safeguards. **No GPU training or inference has been run.**

## Result status

| Type of finding | Status |
|---|---|
| Historical CIoU era in commit history | Confirmed in Git |
| Five current baseline/WIoU script pairs fail to explicitly isolate loss | Confirmed by line-by-line comparison |
| Current RUOD and DUO validation path points into `images/test` | Confirmed in configs |
| Best-checkpoint selection is influenced by configured validation fitness | Confirmed in train-loop source |
| Corrected loss-selector code works on synthetic PyTorch inputs | Confirmed by passing CI tests |
| New P5-only architecture and all eight model factor combinations isolate intended YAML changes | Confirmed by passing CI tests |
| Deterministic split generator and SHA-256 overlap logic on synthetic tiny data | Confirmed by passing CI tests |
| Historical claimed mAP and CIoU/WIoU causal differences | **Not verified** — no original experiment artifacts |
| Hash/near-duplicate checks on real training/test image sets | **Not performed** — data inaccessible |
| GPU/DCNv4 compile, model construction and full detector training | **Not performed** — original environment unavailable |
| True held-out model evaluation and realistic latency | **Not performed** — data, checkpoints, and GPU/edge hardware unavailable |

## Deliverables

- [Independent forensic findings](CIoU_SPLIT_FINDINGS.md)
- [ECCV manuscript source-to-code correction map](MANUSCRIPT_TRACEABILITY.md)
- [Reproducible eight-condition ablation matrix](EXPERIMENT_MATRIX.md)
- [Operator and split audit guide](README.md)
- `train_controlled.py`: explicit `--box-loss ciou|wiou`, split preflight, `--dry-run`
- `prepare_dev_split.py`: seeded development split from official train, reserved official test
- `verify_splits.py`: read-only names, paths, and SHA-256 duplicate audit
- `../vali.py`: explicit test/val choice, direct `--data` support, per-class AP labels corrected

## Minimum artifacts required to extend beyond code audit

Read-only access to the **actual** source-of-truth dataset manifests and images, or the output JSONs of `verify_splits.py`, is needed to check overlaps and evaluation independence. To check the exact historical CIoU baseline, obtain per-run `args.yaml`/logs and the code snapshots used for each baseline and WIoU-only ablation, especially RUOD and DUO. For numerical replication, a compatible CUDA/DCNv4 environment and datasets/checkpoints are required.

Until those artifacts exist, do not treat the passing CI tests as a scientific replication of the paper's results.
