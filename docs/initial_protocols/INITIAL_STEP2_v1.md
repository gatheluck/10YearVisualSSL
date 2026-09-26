# INITIAL_STEP2_v1: historical evidence and candidate reconstruction

This is an edited, anonymized companion to the initial experiments, separate
from the later Unified LP/AP/FT specifications. It is not a certification that
all manuscript results follow a single recipe. Original experimental code and
this portable package's integration/validation are separate matters.

## Historical evidence and its limits

The reconstruction describes fixed-contract pretraining checkpoints and shared downstream readers. The original objectives may retain method-specific views. A checkpoint milestone or a trainer existing does not establish a completed reported cell.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Reported pretraining epoch | 300 | `configs/step2_downstream_registry.yaml` `pretrain_epoch: 300` | Observed main-run setting |
| Downstream heads and schedules | Shared `downstream/*.py` full-mode arguments | `scripts/submit_step2_downstream.sh`; DINO epoch-300 `results.json` `run_config` | Observed main-run setting |
| Downstream score epoch | Last scheduled epoch | `final` field in `step2_downstream_full_resumable/*/results.json` | Observed main-run setting |
| NYUv2 | Metric RMSE, mat-order 795/654, masked L1, DPT head | `downstream/nyuv2_depth.py` | Observed main-run setting |
| COCO | Bbox AP, not multilabel mAP | Task id `coco_det_frcnn_frozen_backbone` | Observed main-run setting |
| Pretext view count | Keep the count required by the implemented loss; record it | DINO canonical config uses 2 global and 8 local crops; the project plan also lists a single-crop type-1 policy | Observed objective requirement, recorded rather than retuned |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

The unified ImageNet probe must not replace the historical method-official recipes when explaining reported results. Resolve method-specific objectives and run records before applying a shared recipe.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| ImageNet linear probe | SGD, LR `0.1`, batch 256, 100 epochs, final epoch, pooled patch tokens | Stored probes differ (DINO online probe, effective LR `0.004`, feature dim 3072; MoCo-style `lr=30`) | Newly selected unification |

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

This document is a candidate specification reconstructed from the initial Step 2 campaign in `methods/`. It is not a claim that every stored historical score already follows this file. Downstream COCO, ADE20K, NYUv2, and SSv2 scores in `local_artifacts/step2_downstream_full_resumable/` follow the shared recipes below. ImageNet linear scores were produced by method-official probes and are unified here to one probe.

### Purpose and category scope

Step 2 retrains each SSL objective on one shared trunk and one shared optimization contract, then evaluates the epoch-300 checkpoint with one frozen-backbone recipe per task. The pretext loss stays the loss implemented by that method’s Step 2 trainer. The optimization contract, the trunk, the data, and the downstream heads do not change across methods.

The candidate reported checkpoint is epoch 300. Epochs 100 and 200 are saved milestones. They are not mixed into the epoch-300 table.

### Included models and datasets

Include every `ready*` row of `local_artifacts/step2/step2_methods.tsv` (33 methods, ids matching `configs/step2_downstream_registry.yaml`). Each row’s `checkpoint_dir` is the Step 2 trunk. Reject a directory that contains `INVALID_STEP2_PROTOCOL.json`.

| Task | Dataset | Split | Root confirmed by completed epoch-300 runs |
|---|---|---|---|
| Pretext and ImageNet linear | ILSVRC2012 ImageNet-1k | official train / validation | `${LOCAL_REFERENCE_PATH}` |
| Object detection | COCO 2017 | train2017 118,287 / val2017 5,000 | `${LOCAL_REFERENCE_PATH}` |
| Semantic segmentation | ADE20K Challenge 2016 | 20,210 / 2,000 | `${LOCAL_REFERENCE_PATH}` |
| Metric depth | NYUv2 labeled | first 795 / remaining 654 in mat order | `${LOCAL_REFERENCE_PATH}` |
| Action recognition | Something-Something v2 | official train 168,913 / validation 24,777 | `${LOCAL_REFERENCE_PATH}` |

### Global rules

- Train a `vit_base_patch16_224` trunk from scratch on ImageNet-1k train. Embed dimension 768, depth 12, 12 heads, patch 16.
- Use global batch 1024, AdamW, peak learning rate `6e-4` (the project rule `1.5e-4 × 1024/256`), minimum learning rate `1e-6`, per-iteration cosine decay, 10-epoch linear warmup, and weight decay `0.05`.
- Train for 300 epochs. Write atomic checkpoints at epochs 100, 200, and 300.
- Evaluate epoch 300 only for the candidate table. Freeze the trunk, `eval()` mode, no trunk gradients.
- Downstream seed is `0`. Pretraining seed is `42`.
- Downstream learning rates are absolute at the stated batch. Do not scale them.
- Report the last scheduled downstream epoch.
- Record trunk checkpoint path, epoch, feature width, grid, batch, learning rate, and seed.

### Common representation interface

Downstream adapters dispatch Step 2 trunks as spatial ViTs (`downstream/vit_adapters.py`, kind `step2_timm` / the method’s Step 2 builder). The audited DINO contract is `methods/23_dino/step2_protocol.py`.

- **Checkpoint.** `checkpoint_epoch_300` (or the registry `source` that points at that epoch) under the method’s `checkpoint_dir`. Initialize downstream heads randomly.
- **Included components.** Patch embedding, positional embedding, transformer blocks, and the final trunk norm. Exclude the pretext projector, prototypes, EMA teacher head, and decoder.
- **Frozen scope.** The entire trunk, including norm affine parameters and positional embeddings. The trunk stays in `eval()`.
- **Feature stage.** Final trunk block, after the final norm. Prefix tokens are removed. The remaining patch tokens are the feature.
- **Tensor interface.** `(B, 768, H_grid, W_grid)`, row-major. At 224 and patch 16 the grid is 14×14. Detection exposes this map as scale `"0"`.
- **Special tokens.** Drop the class token. Do not pool it into the grid.
- **Pooling.** Detection, segmentation, and depth keep the grid. ImageNet and SSv2 adaptive-average-pool the grid to a 768-d vector. SSv2 then averages 4 frame vectors.
- **Preprocessing.** Dataset ImageNet normalization, mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`, applied before the trunk.
- **Head boundary.** The first trainable downstream parameter is the task head.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Primary metric |
|---|---|---|---|---|---|
| Pretext | 224 global crop | method trainer’s own head | AdamW | 300 epochs, warmup cosine | — |
| ImageNet-1k | 224 | linear, 1000-way | SGD | 100 epochs, cosine | Top-1 (%) |
| COCO 2017 | short side 800, long side ≤ 1333 | Faster R-CNN, one scale | SGD | 12 epochs, constant LR | bbox AP (%) |
| ADE20K | 224 square | 1×1 conv, 150-way | AdamW | 30 epochs, constant LR | mIoU (%) |
| NYUv2 | 224 square | DPT-style depth head, hidden 256 | AdamW | 30 epochs, constant LR | RMSE (m) |
| SSv2 | 224 square, 4 frames | linear, 174-way | AdamW | 30 epochs, constant LR | Top-1 (%) |

### Detailed task recipes

#### Pretext pretraining

- **Data.** ImageNet-1k train, 1,281,167 images.
- **Trunk.** ViT-B/16 at 224.
- **Objective.** The loss implemented by that method’s Step 2 entry point listed in `step2_methods.tsv`. Keep the view count that loss requires, and record it. Do not swap in a different objective.
- **Optimization.** AdamW, betas `(0.9, 0.95)`, eps `1e-8`, peak LR `6e-4`, minimum LR `1e-6`, weight decay `0.05` held fixed. Biases and 1-dimensional parameters are exempt from weight decay. Per-iteration linear warmup for 10 epochs, then cosine decay for the rest of 300 epochs. Global batch 1024 with accumulation steps `1`. Gradient clip global norm `3.0`.
- **Augmentation.** Random resized crop to 224 and horizontal flip with probability 0.5. Add the extra views required by the implemented loss, at the crop sizes that trainer already uses. Record the view count.
- **Seed.** `42`.
- **Checkpoints.** Epochs 100, 200, and 300. The downstream table loads epoch 300.

#### ImageNet-1k linear probe

- **Split.** Official validation, 50,000 images.
- **Input.** Train: random resized crop to 224 (scale 0.08–1.0, ratio 0.75–1.333) and horizontal flip 0.5. Evaluate: shorter side 256, center crop 224. ImageNet normalization.
- **Feature.** Final patch-token map, adaptive average pool to 768-d. No L2 normalization.
- **Head.** `Linear(768, 1000)`. Train only this head.
- **Loss.** Cross-entropy, no label smoothing.
- **Optimizer.** SGD, momentum `0.9`, Nesterov disabled, weight decay `0`.
- **Learning rate.** `0.1`, absolute, at effective batch 256.
- **Schedule.** 100 epochs, 10-epoch linear warmup from `1e-6`, cosine decay to `1e-6`.
- **Checkpoint rule.** Last scheduled epoch.
- **Metrics.** Top-1 (%) primary, Top-5 (%) secondary.

#### COCO 2017 detection

Same recipe as the Step 1 full-mode detector, on the epoch-300 trunk.

- **Launcher.** `scripts/submit_step2_downstream.sh` with `TASK=coco`, `MODE=full`, `PRETRAIN_EPOCH=300`.
- **Module.** `downstream.coco_frcnn`.
- **Input.** Shorter side 800, longer side at most 1333. No flip.
- **Head.** Faster R-CNN, `num_classes=91`, one feature scale `"0"`, RoIAlign 7×7, sampling ratio 2. Trunk frozen.
- **Optimizer.** SGD, momentum `0.9`, weight decay `5e-4`, learning rate `0.005`, constant, effective batch 4.
- **Schedule.** 12 epochs. Last epoch is the score.
- **Metrics.** Bbox AP (%) = `100 * COCOeval.stats[0]`. Secondary AP50 (%) = `100 * stats[1]`.

Completed evidence: `local_artifacts/step2_downstream_full_resumable/coco_method23/results.json` records `epochs=12`, `batch_size=4`, `lr=0.005`, `min_size=800`, `max_size=1333`, `seed=0`, and `final` equal to epoch 11 (0-based).

#### ADE20K

- **Module.** `downstream.ade20k_segmentation`.
- **Input.** Square resize 224. Bilinear image, nearest mask. ImageNet normalization. No flip and no scale jitter.
- **Head.** 1×1 conv to 150 classes, bilinear upsample, `align_corners=false`. Trunk frozen.
- **Loss.** Pixel cross-entropy, `ignore_index=255`. Labels follow the Step 1 shift (0 → ignore, 1–150 → 0–149).
- **Optimizer.** AdamW, learning rate `1e-3`, weight decay `0.01`, effective batch 16, constant LR, 30 epochs.
- **Checkpoint rule.** Last epoch.
- **Metrics.** mIoU (%) primary, pixel accuracy (%) secondary.

#### NYUv2 metric depth

- **Module.** `downstream.nyuv2_depth`.
- **Split.** First 795 / remaining 654 of `labeled/nyu_depth_v2_labeled.mat`. This is not `splits.mat`.
- **Input.** 224 square. Bilinear RGB, nearest depth. ImageNet normalization.
- **Head.** `DPTDepthHead` with hidden width 256 and a final `softplus`. Trunk frozen.
- **Loss.** Masked L1 on metric depth. Valid pixels are finite and `> 0`.
- **Optimizer.** AdamW, learning rate `1e-3`, weight decay `0.01`, effective batch 16, constant LR, 30 epochs.
- **Evaluation.** No per-image median scale.
- **Metrics.** RMSE (m) primary, AbsRel secondary.

#### Something-Something v2

- **Module.** `downstream.ssv2_linear`.
- **Input.** 4 frames from `linspace` over the decoded video, square resize 224, ImageNet normalization, no flip.
- **Feature.** Spatial average pool per frame, then mean over the 4 frames.
- **Head.** `Linear(768, 174)`.
- **Loss.** Cross-entropy.
- **Optimizer.** AdamW, learning rate `1e-3`, weight decay `0.01`, effective batch 8, constant LR, 30 epochs.
- **Evaluation.** One clip per validation video.
- **Metrics.** Top-1 (%) primary, Top-5 (%) secondary.

### Execution and seed policy

Pretrain with the method script in `step2_methods.tsv` for 300 epochs, seed 42. Then launch downstream with `scripts/submit_step2_downstream.sh` at `PRETRAIN_EPOCH=300`, `MODE=full`, `SEED=0`.

One trial per method per task. Do not enter `methods/_seed3_step2/` trials into this table.

Outputs:

- ImageNet linear: the method `local_artifacts/results/step2_linear_eval_epoch300/results.json` path, rewritten to this probe.
- Other tasks: `local_artifacts/step2_downstream_full_resumable/<task>_method<id>/results.json`.

### Evaluation and reporting

Publish the epoch-300 table only. Columns are Top-1 / Top-5 (%), bbox AP / AP50 (%), mIoU / pixel accuracy (%), RMSE (m) / AbsRel, and SSv2 Top-1 / Top-5 (%). Name the checkpoint epoch in the table caption. A best-epoch number may be stored as a labeled secondary field and is not the candidate score.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| Trunk and schedule | ViT-B/16, 300 epochs, batch 1024, AdamW, LR `6e-4`, WD `0.05`, 10-epoch warmup, cosine | the private study outline (not distributed); `methods/23_dino/configs/step2_vit_b.yaml`; `methods/23_dino/step2_protocol.py` | Observed main-run setting |
| Reported pretraining epoch | 300 | `configs/step2_downstream_registry.yaml` `pretrain_epoch: 300` | Observed main-run setting |
| Downstream heads and schedules | Shared `downstream/*.py` full-mode arguments | `scripts/submit_step2_downstream.sh`; DINO epoch-300 `results.json` `run_config` | Observed main-run setting |
| Downstream score epoch | Last scheduled epoch | `final` field in `step2_downstream_full_resumable/*/results.json` | Observed main-run setting |
| NYUv2 | Metric RMSE, mat-order 795/654, masked L1, DPT head | `downstream/nyuv2_depth.py` | Observed main-run setting |
| COCO | Bbox AP, not multilabel mAP | Task id `coco_det_frcnn_frozen_backbone` | Observed main-run setting |
| ImageNet linear probe | SGD, LR `0.1`, batch 256, 100 epochs, final epoch, pooled patch tokens | Stored probes differ (DINO online probe, effective LR `0.004`, feature dim 3072; MoCo-style `lr=30`) | Newly selected unification |
| Pretext view count | Keep the count required by the implemented loss; record it | DINO candidate config uses 2 global and 8 local crops; the project plan also lists a single-crop type-1 policy | Observed objective requirement, recorded rather than retuned |

### Evidence references

- Plan: the private study outline (not distributed) section “Step 2: Unified SSL Comparison”.
- Registry: `configs/step2_downstream_registry.yaml`, `local_artifacts/step2/step2_methods.tsv`.
- Audited optimization contract: `methods/23_dino/configs/step2_vit_b.yaml`, `methods/23_dino/step2_protocol.py`, `methods/28_dinov2/STEP2_RECIPE.md`.
- Launcher: `scripts/submit_step2_downstream.sh`.
- Implementation: `downstream/coco_frcnn.py`, `downstream/ade20k_segmentation.py`, `downstream/nyuv2_depth.py`, `downstream/ssv2_linear.py`, `downstream/vit_adapters.py`.
- Completed downstream example: `local_artifacts/step2_downstream_full_resumable/{coco,ade20k,nyuv2,ssv2}_method23/results.json`.
