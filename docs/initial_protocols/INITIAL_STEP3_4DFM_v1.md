# INITIAL_STEP3_4DFM_v1: historical evidence and candidate reconstruction

This is an edited, anonymized companion to the initial experiments, separate
from the later Unified LP/AP/FT specifications. It is not a certification that
all manuscript results follow a single recipe. Original experimental code and
this portable package's integration/validation are separate matters.

## Historical evidence and its limits

The supplied evidence table describes 224-pixel runs for some models and 518-pixel runs for others. It also describes several selected SSv2 learning rates. Correspondence between these records and manuscript cells still requires the exact run records.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| V-DPM checkpoint | `edgarsucar/vdpm` `model.pt` | `vdpm/backbone/vdpm_backbone.py` `VDPM_HF_URL` | Observed main-run setting |
| ADE / NYUv2 epoch count | 20 | Dense JSON `config.epochs` | Observed main-run setting |
| Depth definition | SiLog train, metric RMSE | Shared dense head with the 3DFM probe; result key `best_rmse` | Observed main-run setting |
| COCO task | Detection bbox AP | `num_classes: 80` and Faster R-CNN keys in `coco_dense_*/results.json` | Observed main-run setting |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

Forcing every model to 224 pixels and a single learning rate is a new unification. Historical 518-pixel results remain historical results, not evidence of a completed 224-pixel rerun.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Input size | 224 square on every task | MonST3R and D²USt3R ADE/NYU/SSv2 configs store `image_size: 224`. DVLT, StreamVGGT, and some V-DPM configs store 518 | Newly selected unification |
| ImageNet LR | `0.1` at head batch 1024 | ImageNet-1k `best_lr` is `0.1` for all five included models | Newly selected unification (sweep removed) |
| SSv2 length and length of training | 8 frames, 20 epochs, LR `0.1` | SSv2 JSON `config` stores `num_frames: 8`, `epochs: 20`. Selected LRs were 0.1, 0.05, or 0.01 | Frame count and epoch count observed; LR unified |

Candidate settings and execution instructions below are not evidence of completed runs.
They must not retroactively redefine the conditions of published result cells.
No experiment, seed assignment, preprocessing or model implementation is changed
by including this document.

## Supplied candidate specification (attributed reconstruction)

The following detailed reconstruction is retained for review, including its
own evidence/decision table. Its imperative language describes a candidate
procedure, not a command to execute or a claim that each setting was used.
New unifications remain proposals unless matched to a completed run. Statements
about stored files are source attributions, not a fresh audit of those files.

Internal paths and job identifiers have been removed. `${LOCAL_REFERENCE_PATH}`
is an intentionally unresolved placeholder, not a runnable dataset path.
Relative source paths name the original experimental archive; their presence
here does not claim those scripts are delivered or runnable in this package.
Public upstream model names identify comparison methods, not submission authors.

This document is a candidate specification reconstructed from the initial campaign under `methods_step3/4DFM/`. It is not a claim that every stored score already follows this file. ImageNet and SSv2 results selected a learning rate from a sweep. Several trunks were forwarded at 518 px. This file fixes the learning rate and the input size.

### Purpose and category scope

Evaluate released 4D and dynamic-scene foundation models as frozen visual trunks. The scientific target is the encoder or aggregator patch representation, not the quality of a 4D reconstruction. MotionCrafter is a diffusion demo and is not a trunk in this protocol.

### Included models and datasets

Include trunks that have a completed ImageNet-1k `results.json`.

| Directory | Checkpoint identifier | Feature width stored on ImageNet-1k |
|---|---|---|
| `monst3r` | MonST3R ViT-L, variant `monst3r_vitl_512_dpt` | 1024 |
| `d2ust3r` | D²USt3R ViT-L, variant `d2ust3r_vitl` | 1024 |
| `streamvggt` | StreamVGGT-1B, variant `streamvggt_1b` | 2048 |
| `dvlt` | DVLT, variant `dvlt` | 768 |
| `vdpm` | V-DPM, `edgarsucar/vdpm` file `model.pt`, variant `vdpm_vggt_1b` | 2048 |

`any4d`, `opend4rt`, and `motioncrafter` have no completed five-task results and are outside the table.

| Task | Dataset | Split |
|---|---|---|
| Image classification | ImageNet-1k | official train / validation |
| Detection | COCO 2017 | train2017 / val2017 |
| Semantic segmentation | ADE20K | 20,210 / 2,000 |
| Metric depth | NYUv2 | depth valid on `[0.001, 10.0]` m |
| Action recognition | Something-Something v2 | official train / validation, 8 frames |

Dataset roots match the 3DFM loaders shared by this campaign: `${LOCAL_REFERENCE_PATH}`. ImageNet-100 is a pilot and stays out of the table.

### Global rules

- Load the released checkpoint. Freeze the trunk. Forward in `eval()` with bfloat16 autocast. Compute the loss in float32.
- Use a 224×224 square input on every task. Record 224 in the result.
- Use the wrapper’s `mean` and `std`. Record them.
- Seed `0`. One trial.
- Learning rates below are absolute at the stated effective batch. Do not sweep them and do not rescale them.
- Classification reports the best validation Top-1 inside the  scheduled epochs. Dense tasks and detection report the best validation value of the primary metric, and the epoch that produced it.

### Common representation interface

- **Checkpoint.** The variant named in the table. V-DPM loads `model.pt` from `edgarsucar/vdpm` into the VGGT aggregator (`4DFM/vdpm/backbone/vdpm_backbone.py`).
- **Included components.** The encoder or causal aggregator through its final block. Exclude DPT fusion heads and camera heads.
- **Frozen scope.** All trunk parameters.
- **Feature stage.** Final encoder or aggregator layer. Patch tokens only.
- **Tensor interface.** `(B, D, h, h)` after dropping prefix tokens and reshaping row-major. Global tasks mean-pool to `(B, D)`. Dense tasks keep the grid.
- **Width.** `D` is the checkpoint width in the table. Record it. Do not project widths onto one another.
- **Special tokens.** Drop camera and register tokens before pooling or reshaping.
- **Normalization.** The wrapper mean and std, once. No extra L2 normalization of the feature.
- **Head boundary.** Linear classifier, 1×1 segmentation conv, 1×1 log-depth conv, or Faster R-CNN heads.

A trunk that cannot emit a spatial patch grid at 224 is unsupported for ADE20K, NYUv2, and COCO. Do not tile a global vector into a fake map.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Score epoch | Primary metric |
|---|---|---|---|---|---|---|
| ImageNet-1k | 224 | linear, 1000-way | SGD, Nesterov | 100 epochs | best validation Top-1 | Top-1 (%) |
| COCO 2017 | 224, then detector resize | Faster R-CNN | SGD | 12 epochs | best validation bbox AP | bbox AP (%) |
| ADE20K | 224 | 1×1 conv, 150-way | SGD | 20 epochs | best validation mIoU | mIoU (%) |
| NYUv2 | 224 | 1×1 log-depth | AdamW | 20 epochs | best validation RMSE | RMSE (m) |
| SSv2 | 224, 8 frames | linear, 174-way | SGD, Nesterov | 20 epochs | best validation Top-1 | Top-1 (%) |

### Detailed task recipes

#### ImageNet-1k

- **Feature.** Mean of final patch tokens, cached.
- **Head.** `Linear(D, 1000)`, weight std `0.01`, zero bias. Head batch 1024.
- **Loss.** Cross-entropy, label smoothing `0`.
- **Optimizer.** SGD, momentum `0.9`, Nesterov enabled, weight decay `0`.
- **Learning rate.** `0.1`, absolute, at head batch 1024.
- **Schedule.** 100 epochs, 10-epoch linear warmup, cosine (`3DFM/_common/linear_probe.py`, reused by this campaign).
- **Checkpoint rule.** Best validation Top-1.
- **Metrics.** Top-1 (%) primary, Top-5 (%) secondary.

#### COCO 2017 detection

- **Recorded config.** 12 epochs, effective batch 2, learning rate `0.001`, weight decay `1e-4`, warmup 1 epoch.
- **Detector.** Same RPN and box limits as the 3DFM probe: train pre/post NMS 2000/1000, test 1000/500, score threshold `0.05`, NMS `0.5`, 100 detections per image, 80 classes.
- **Feature.** Spatial grid at 224. No global pool.
- **Optimizer.** SGD, momentum `0.9`.
- **Checkpoint rule.** Best validation bbox AP.
- **Metrics.** Bbox AP (%). Not multilabel mAP.

#### ADE20K

- **Head.** 1×1 conv to 150 classes, bilinear upsample, `align_corners=false`.
- **Loss.** Pixel cross-entropy, `ignore_index=255`.
- **Optimizer.** SGD, momentum `0.9`, weight decay `1e-4`, learning rate `0.01`, effective batch 32.
- **Schedule.** 20 epochs, warmup 2 epochs, cosine.
- **Checkpoint rule.** Best validation mIoU.
- **Metrics.** mIoU (%) primary.

#### NYUv2 metric depth

- **Head.** 1×1 conv on log-depth, converted back to metres.
- **Loss.** SiLog with `lam=0.85` on depths in `[0.001, 10.0]` m.
- **Optimizer.** AdamW, learning rate `0.001`, weight decay `1e-4`, effective batch 32.
- **Schedule.** 20 epochs, warmup 2 epochs, cosine.
- **Checkpoint rule.** Lowest validation RMSE.
- **Metrics.** Metric RMSE (m) primary. AbsRel at that epoch secondary. Do not apply a per-image median scale before RMSE.

#### Something-Something v2

- **Input.** 8 frames, 224 square, mean of per-frame patch tokens, then mean over time.
- **Head.** `Linear(D, 174)`, head batch 1024.
- **Optimizer.** SGD, momentum `0.9`, Nesterov enabled, weight decay `0`, learning rate `0.1`.
- **Schedule.** 20 epochs, warmup 5 epochs, cosine. The completed SSv2 JSON files store `epochs=20` and `num_frames=8`.
- **Checkpoint rule.** Best validation Top-1.
- **Metrics.** Top-1 (%) primary, Top-5 (%) secondary.

### Execution and seed policy

Use `methods_step3/4DFM/submit_all_5tasks_4dfm.sh` and the per-method probe scripts, with `image_size=224` and seed `0`. One trial per method per task.

### Evaluation and reporting

Report bbox AP in percent, mIoU in percent, and RMSE in metres, with the winning epoch. Record `D` and the checkpoint file name. A 518 px historical run is not a candidate score.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| Membership | MonST3R, D²USt3R, StreamVGGT, DVLT, V-DPM | Completed `results.json` counts; `4DFM/MASTER_PLAN.md` include list | Observed main-run setting |
| V-DPM checkpoint | `edgarsucar/vdpm` `model.pt` | `vdpm/backbone/vdpm_backbone.py` `VDPM_HF_URL` | Observed main-run setting |
| Input size | 224 square on every task | MonST3R and D²USt3R ADE/NYU/SSv2 configs store `image_size: 224`. DVLT, StreamVGGT, and some V-DPM configs store 518 | Newly selected unification |
| ImageNet LR | `0.1` at head batch 1024 | ImageNet-1k `best_lr` is `0.1` for all five included models | Newly selected unification (sweep removed) |
| SSv2 length and length of training | 8 frames, 20 epochs, LR `0.1` | SSv2 JSON `config` stores `num_frames: 8`, `epochs: 20`. Selected LRs were 0.1, 0.05, or 0.01 | Frame count and epoch count observed; LR unified |
| ADE / NYUv2 epoch count | 20 | Dense JSON `config.epochs` | Observed main-run setting |
| Depth definition | SiLog train, metric RMSE | Shared dense head with the 3DFM probe; result key `best_rmse` | Observed main-run setting |
| COCO task | Detection bbox AP | `num_classes: 80` and Faster R-CNN keys in `coco_dense_*/results.json` | Observed main-run setting |

### Evidence references

- `methods_step3/4DFM/MASTER_PLAN.md`
- `methods_step3/4DFM/submit_all_5tasks_4dfm.sh`
- `methods_step3/4DFM/vdpm/backbone/vdpm_backbone.py`
- `methods_step3/3DFM/_common/linear_probe.py` and `dense_heads.py` (shared probe implementation)
- Completed files `methods_step3/4DFM/<method>/results/*_*/results.json`
