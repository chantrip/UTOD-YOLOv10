# Controlled 2×2×2 ablation matrix (future reruns; no new results)

The experiment changes only three specified factors, using matched training settings, split, seed, and evaluation protocol. **This is a proposed protocol, not a statement about the actual historical paper results.**

- Factor W: **CIoU** versus **WIoU-v3** via the explicit `--box-loss` selector.
- Factor D: ordinary `C2f` versus `DC2f` **only at P5/32**.
- Factor F: the existing `BiFPN`-named **weighted-concatenation operator** at the three selected neck merges versus standard `Concat`. The implementation is **not conventional full BiFPN**; retain the actual name/behavior in results until method nomenclature is corrected.

| ID | WIoU? | P5 DC2f? | Weighted concats? | Architecture YAML | `--box-loss` |
|---|---|---|---|---|---|
| E0 | No | No | No | `yolov10n.yaml` | `ciou` |
| E1 | Yes | No | No | `yolov10n.yaml` | `wiou` |
| E2 | No | Yes | No | `yolov10nP5DCN.yaml` | `ciou` |
| E3 | No | No | Yes | `yolov10nBi.yaml` | `ciou` |
| E4 | Yes | Yes | No | `yolov10nP5DCN.yaml` | `wiou` |
| E5 | Yes | No | Yes | `yolov10nBi.yaml` | `wiou` |
| E6 | No | Yes | Yes | `yolov10n-dwb.yaml` | `ciou` |
| E7 | Yes | Yes | Yes | `yolov10n-dwb.yaml` | `wiou` |

Every row represents **the same base YOLOv10n**, dataset partition, initial-weights policy (`pretrained=False`), image size, epochs, batch, augmentations, and seed. All eight are necessary to estimate potential loss/feature interactions without assuming contributions are additive. For a lower-cost validation, first inspect E0, E1, E6, and E7 on one dataset, and avoid declaring a component improvement before proper repeatability checks.

**Run commands (example after generating a valid 3-way RUOD split):**

```bash
DATA=/home/dsai-st125287/project/audit-splits/ruod_s42/data_audit.yaml

python audit/train_controlled.py --data "$DATA" --model yolov10n.yaml \
  --box-loss ciou --name ruod_E0_ciou_s42 --seed 42 --dry-run

python audit/train_controlled.py --data "$DATA" --model yolov10n.yaml \
  --box-loss wiou --name ruod_E1_wiou_s42 --seed 42 --dry-run

python audit/train_controlled.py --data "$DATA" --model yolov10nP5DCN.yaml \
  --box-loss ciou --name ruod_E2_p5dcn_s42 --seed 42 --dry-run

python audit/train_controlled.py --data "$DATA" --model yolov10nBi.yaml \
  --box-loss ciou --name ruod_E3_weightedcat_s42 --seed 42 --dry-run

python audit/train_controlled.py --data "$DATA" --model yolov10nP5DCN.yaml \
  --box-loss wiou --name ruod_E4_p5dcn_wiou_s42 --seed 42 --dry-run

python audit/train_controlled.py --data "$DATA" --model yolov10nBi.yaml \
  --box-loss wiou --name ruod_E5_weightedcat_wiou_s42 --seed 42 --dry-run

python audit/train_controlled.py --data "$DATA" --model yolov10n-dwb.yaml \
  --box-loss ciou --name ruod_E6_full_no_wiou_s42 --seed 42 --dry-run

python audit/train_controlled.py --data "$DATA" --model yolov10n-dwb.yaml \
  --box-loss wiou --name ruod_E7_full_wiou_s42 --seed 42 --dry-run
```

Remove `--dry-run` only after dataset checks, dependency/extension checks, and GPU scheduling. Then run independent final evaluation explicitly using `vali.py --split test` and only after **all model selection is frozen**.

## Reporting standards

- Report **absolute AP differences in percentage points** (e.g. 0.577 → 0.593 is +1.6 points in AP scaled 0–100), not relative percent gains.
- Separate AP50 from AP50–95 and label thresholds correctly.
- Report mean and standard deviation over matched seeds, same validation/test manifests, and confidence intervals when available.
- Log exact code commit, dataset manifest hashes, experiment arguments, per-seed metrics, model/fusion variant, model-selection criterion and selected epoch.
- Never use the final test to choose architectures, seeds, checkpoints, thresholds, or preprocessing/hyperparameter configurations.
- If all eight configurations or a fresh final test are not available, narrow the causal claims and publish only defensible comparisons.
