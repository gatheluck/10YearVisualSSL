# BASIC5_LORA_v2

**Protocol ID:** `B5X_LORA_LP_v2` (COCO cells: `B5X_LORA_LP_v1`, the same recipe)  
**Status:** as run, v2.0  
**Track:** LoRA adaptation  
**Baseline:** `BASIC5_FAIR_v1` (task definition); optimizer blocks from `BASIC5_ATTENTIVE_v1` (head) and `BASIC5_FINETUNE_v1` (LoRA learning rate)

## Purpose

This protocol measures how far a light adaptation that updates the backbone moves each model beyond frozen probing.

- The released backbone keeps its weights. Rank-16 low-rank adapters (LoRA) inside every attention projection are trained together with the `BASIC5_FAIR_v1` task head on the model's canonical feature.
- Dataset, split, input geometry, epoch count, head, loss, and metric are those of `BASIC5_FAIR_v1`.

It complements the three Unified tracks:

1. `BASIC5_FAIR_v1`: linear head, frozen backbone.
2. `BASIC5_ATTENTIVE_v1`: one-block attention reader, frozen backbone.
3. `BASIC5_FINETUNE_v1`: full fine-tuning.
4. **`B5X_LORA_LP_v2`: LoRA in the attention projections + the `BASIC5_FAIR_v1` head.**

This file was written on 2026-11-19 from the code (`code_v8`, `code_v9`) and the run records. The runs read the `BASIC5_*_v1.json` files and the LoRA constants in the code; this file was not an input to them.

## Canonical skeleton

| Dataset | Epochs / schedule | Trainable | Head LR (AdamW unless noted) | LoRA LR | Primary metric |
|---|---:|---|---|---|---|
| ImageNet-1k | 100 (DINOv3-7B: 50) | LoRA + linear classifier | 1e-3 @ 256 | 1.25e-4 @ 256 | Top-1 |
| COCO | 12 / 1× | LoRA + pyramid + Faster R-CNN | SGD 0.02 @ 16 | SGD 0.02 @ 16 | bbox AP |
| ADE20K | 20 | LoRA + 1×1 Conv | 1e-3 @ 8 | 1e-4 @ 8 | mIoU |
| NYUv2 | 30 | LoRA + 1×1 depth Conv | 1e-3 @ 8 | 1e-4 @ 8 | RMSE |
| SSv2 | 50 (DINOv3-7B: 24) / 16 frames | LoRA + linear classifier | 1e-3 @ 256 | 5e-4 @ 256 | Top-1 |

## Global rules

**Backbone**
- Load every released checkpoint **as-is**.
- Keep the backbone in `eval()` during training: dropout off, normalization statistics frozen, and the DINOv3 RoPE training augmentation disabled.
- No backbone weight is trained. Gradients flow through the frozen backbone to the LoRA matrices only. The first step of every run checks that no backbone tensor received a gradient.

**Trainable parameters**
- LoRA A and B matrices.
- The `BASIC5_FAIR_v1` head on the canonical feature. For COCO this is the model's Unified feature pyramid plus the complete Faster R-CNN.

**Feature**
- The same canonical feature as the model's `BASIC5_FAIR_v1` run: the global feature for ImageNet-1k, the Unified LP video feature for SSv2, and the spatial grid for ADE20K and NYUv2. COCO uses the spatial map fed to the model's Unified feature pyramid.
- LoRA changes the backbone that produces the feature, not the readout. No feature cache.

**Recipe**
- One recipe per task for all six models. There is no model-specific learning rate, rank, target set, or layer selection.
- The only per-model change is the shortened DINOv3-7B schedule on ImageNet-1k and SSv2.

**Precision and clipping**
- bfloat16 backbone weights and autocast.
- LoRA and head parameters, loss, and metrics in float32.
- Global-norm gradient clipping at 1.0 over all trainable parameters.
- Activation checkpointing on ImageNet-1k, SSv2, and COCO (memory only).

**Seeds** `0`, `1`, `2`.

**Resume**
- Runs longer than one 12-hour job resume from the last epoch checkpoint: trainable weights, optimizer state, step counter, and sampler epoch.
- Random number generators are reseeded per segment, so a resumed run differs from an uninterrupted one only in its random draws.

## LoRA module

- `W x + (alpha / r) · B A x`, with rank `r = 16`, `alpha = 16` (scaling 1.0), and no dropout.
- `A` is Kaiming-uniform (`a = √5`) and `B = 0`, so training starts exactly at the released backbone. A and B are float32.
- **Targets:**
  - In every transformer block, every `nn.Linear` named `qkv`, `proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, or `out_proj` whose module path contains `attn` or `attention`.
  - A block must yield 2 targets (fused qkv + output) or 4 targets (q, k, v + output); otherwise the run stops.
  - MLPs, patch embeddings, poolers, mergers, and released heads stay frozen.
- **Live check:** before training, every wrapped projection must be called on the canonical feature path, and a random perturbation of `B` must change the canonical feature. Otherwise the run stops.
- **Learning-rate groups:**
  - The head trains at the `BASIC5_ATTENTIVE_v1` rate of the task; LoRA trains at the `BASIC5_FINETUNE_v1` rate of the task. Both are linearly scaled to the effective batch, with no layer decay.
  - Biases, 1-D parameters, and parameters named `norm` have weight decay 0.
  - Warm-up and cosine decay apply to each group's own base rate and advance once per optimizer step.

| Model | Wrapped projections | LoRA parameters |
|---|---|---:|
| DINOv3-7B | 40 blocks × 4 = 160 | 20,971,520 |
| SigLIP 2-G | 40 blocks × 4 = 160 | 7,864,320 |
| V-JEPA 2.1-G | 40 blocks × 2 = 80 | 5,406,720 |
| VGGT-Ω | 48 aggregator blocks (24 frame + 24 global) × 2 = 96 | 4,718,592 |
| RAEv2-K7 | 24 DINOv3-L blocks × 2 = 48 | 2,359,296 |
| Cosmos 3 Super | 27 vision-tower blocks × 2 = 54 | 2,985,984 |

## Task protocols

### ImageNet-1k

- **Split:** official ILSVRC2012 train (1,281,167) / validation (50,000).
- **Input:**
  - Training: random resized crop to 224 (scale 0.08–1.0, ratio 0.75–1.3333, bicubic) and horizontal flip with probability 0.5.
  - Evaluation: resize the shorter side to 256 (bicubic) and center-crop 224, one view.
  - Published backbone normalization.
- **Feature:** the model's Unified LP global feature, L2-normalized.
- **Head:** linear 1000-class classifier (weights N(0, 0.01), bias 0).
- **Optimization:**
  - 100 epochs, AdamW, betas `(0.9, 0.999)`, weight decay `0.05`.
  - Head LR `1e-3` at batch 256. LoRA LR `5e-4` at batch 1024, which gives `1.25e-4` at batch 256.
  - Batch 256 = 8 GPUs × 32, no accumulation.
  - 5-epoch linear warm-up from `1e-6`, then cosine decay to `1e-6`.
- **Short schedule (DINOv3-7B):**
  - 50 epochs (`B5X_LORA_LP_50EP_v2`); everything else is unchanged, and the warm-up and cosine span the 50 epochs.
  - Reason: compute. A 100-epoch run takes about 98 h per seed on one 8-GPU node.
- **Augmentation/loss:** no Mixup, CutMix, label smoothing, RandAugment, or random erasing. Cross entropy.
- **Report:** Top-1 primary and Top-5 secondary, on all 50,000 images.

### COCO 2017

- **Split:** train2017 / val2017 (5,000 images). Cells carry the id `B5X_LORA_LP_v1` (see the version history).
- **Input:**
  - Shorter side 800, longer side at most 1333; horizontal flip with probability 0.5 in training.
  - The model's Unified detection normalization.
  - Crowd boxes and boxes of 1 px or less are dropped.
- **Feature and pyramid:**
  - The canonical spatial map feeds the model's own Unified pyramid, with 4 levels and 256 channels.
  - basic5 SimpleFPN (transposed convolutions): SigLIP 2-G, Cosmos 3 Super, VGGT-Ω.
  - Package SimpleFeaturePyramid (bilinear resampling + 1×1 laterals): DINOv3-7B, RAEv2-K7, V-JEPA 2.1.
- **Head:** torchvision Faster R-CNN with anchors 32/64/128/256 × {0.5, 1, 2}, MultiScaleRoIAlign 7×7 (sampling ratio 2), and box classifier/regressor (`num_classes` 91, labels 1–80 used).
- **Trainable:** LoRA, pyramid, RPN, and RoI heads.
- **Optimization:**
  - 12 epochs / 1×, SGD with momentum 0.9 and weight decay `1e-4`.
  - LR `0.02` at batch 16, shared by LoRA and the detector. The `BASIC5_FINETUNE_v1` COCO rate equals the LP rate.
  - Batch 16 = 8 GPUs × 2.
  - 500-iteration linear warm-up from factor 0.001; ×0.1 after epoch 8 and ×0.01 after epoch 11.
- **Report:** bbox AP@[0.50:0.95] primary (final epoch, all 5,000 images); AP50, AP75, APS, APM, and APL secondary.

### ADE20K

- **Split:** official 20,210 train / 2,000 validation.
- **Input:**
  - Training: random rescale by 0.5–2.0 (bicubic, each side at least 224 px), random 224×224 crop, and horizontal flip with probability 0.5.
  - Evaluation: image resized to 224×224 (bicubic) and label resized to 224×224 (nearest).
- **Feature/head:**
  - The canonical spatial grid: 14×14 for patch-16 models; 7×7 for the Cosmos 3 Super merger.
  - The unchanged 1×1 Conv with 150 outputs (N(0, 0.01) init), then bilinear upsampling to the label size with `align_corners=false`.
- **Optimization:**
  - 20 epochs, AdamW, betas `(0.9, 0.999)`, weight decay `0.05`.
  - Head LR `1e-3` at batch 8; LoRA LR `1e-4` at batch 8.
  - Batch 8 on one GPU.
  - 1-epoch warm-up from `1e-6`, then cosine decay to `1e-6`.
- **Loss:** pixel-wise cross entropy with `ignore_index=255`.
- **Report:** mIoU over 150 classes primary; pixel accuracy secondary.

### NYUv2

- **Split:** official Eigen-style 795 train / 654 test from `labeled.mat` and `splits.mat`.
- **Input:**
  - Images and depths are read in their true 480×640 orientation (the basic5 loader transposed them).
  - Image resized to 224×224 (bicubic) and depth resized to 224×224 (bilinear); horizontal flip with probability 0.5 in training.
  - Valid depth is 0.1–10 m.
- **Feature/head:** the canonical spatial grid, the unchanged single-channel 1×1 Conv with `softplus(x) + 1e-3`, and bilinear upsampling with `align_corners=false`.
- **Optimization:** as for ADE20K, but with 30 epochs.
- **Loss:** scale-invariant log-depth loss on valid pixels with λ = 0.5: `mean(d²) − 0.5 · mean(d)²`, where `d = log pred − log gt`.
- **Report:**
  - Predictions are clamped to [0.1, 10] and evaluated on valid pixels with 0.1 < depth < 10 m.
  - The record's `rmse` is median-aligned per image. The paper reports the unaligned `rmse_unaligned`, as defined in its evaluation appendix.
  - AbsRel (aligned and unaligned), δ1, δ2, and δ3 are secondary.

### Something-Something v2

- **Split:** official train (168,913) / validation (24,777).
- **Input:**
  - 16 equal temporal segments: random within the segment for training, the segment center for evaluation.
  - Training: one random resized crop to 224 (scale 0.08–1.0, ratio 0.75–1.3333, bicubic) shared by all 16 frames, no flip.
  - Evaluation: resize the shorter side to 256 and center-crop 224; one clip, one view.
- **Loader:**
  - One decord loader for all six models.
  - Missing files are excluded at construction and counted.
  - A decode failure stops the run instead of substituting an all-zero clip.
- **Feature:** the model's Unified LP SSv2 feature.
  - DINOv3-7B and RAEv2-K7: per-frame L2-normalized global feature (DINOv3 CLS; RAEv2 mean of K7 tokens), then temporal mean.
  - SigLIP 2-G and Cosmos 3 Super: per-frame pooled feature (MAP pooler; merger mean), then temporal mean, no L2.
  - V-JEPA 2.1: mean of the clip's spatiotemporal tokens, L2-normalized.
  - VGGT-Ω: aggregator over the 16 frames, mean of patch tokens, L2-normalized.
- **Head:** linear 174-class classifier.
- **Optimization:**
  - 50 epochs, AdamW, weight decay `0.05`.
  - Head LR `1e-3` at batch 256; LoRA LR `5e-4` at batch 256.
  - Effective batch 256 = 8 GPUs × 8 clips × 4 accumulation steps.
  - 5-epoch warm-up from `1e-6`, then cosine decay to `1e-6`.
- **Short schedule (DINOv3-7B):**
  - 24 epochs (`B5X_LORA_LP_24EP_v2`); everything else is unchanged.
  - Reason: compute. A 50-epoch run takes about 138 h per seed on one 8-GPU node.
- **Report:** Top-1 primary and Top-5 secondary, on all 24,777 validation videos.

## Reporting and comparability

- Use the final scheduled epoch, with no selection on the reported split.
- Report every seed plus the mean ± sample SD (ddof = 1) over seeds 0, 1, and 2.
- A cell counts only if all of the following hold:
  - it has `COMPLETED_VALID.json`;
  - it reached the recipe's epoch count;
  - steps per epoch × epochs = total steps;
  - it was evaluated on the full split.
- Mark the DINOv3-7B short-schedule cells wherever they appear.
- If a numerical fix is needed, apply it to every model, increment the version, and disclose it.

## Version history

| Version | ID | Date | Recipe | Outcome |
|---|---|---|---|---|
| 1.0 | `B5X_LORA_LP_v1` | 2026-11-11 | LoRA at the head rate (`1e-3` at batch 8 on ADE20K/NYUv2) | ADE20K: SigLIP 2-G and DINOv3-7B diverged after warm-up (loss 1.2–1.4 after epoch 1, rising to 2.3–3.6). Cells kept under `LORA/`. |
| 2.0 | `B5X_LORA_LP_v2` | 2026-11-11 | LoRA at the `BASIC5_FINETUNE_v1` rate of the task; the head keeps the `BASIC5_ATTENTIVE_v1` rate | Applied to all six models and tasks, and every LoRA cell was rerun. COCO is unchanged because the FT and LP COCO rates are equal. |

## Deviations from the Unified code (all six models)

- NYUv2 is read upright.
- One SSv2 loader (decord, bicubic) is used for all models, and a decode failure stops the run.
- The learning-rate schedule advances per optimizer step.

## Where things are

- Code: `${EXPERIMENT_ROOT}/code_v8`, `code_v9` (`b5x/lora.py`, `train.py`, `train_cls.py`, `train_det.py`).
- Protocol note: `${EXPERIMENT_ROOT}/PROTOCOL.md`.
- Records: `${EXPERIMENT_ROOT}/results/<model>/<dataset>/{LORA_v2,LORA}/<seed dir>/result.json`.
