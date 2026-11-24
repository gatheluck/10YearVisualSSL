# BASIC5_DPT_v2

**Protocol ID:** `B5X_DPT_LP_v2`  
**Status:** as run, v2.0  
**Track:** DPT probing (dense tasks only)  
**Baseline:** `BASIC5_FAIR_v1` (task definition); optimizer block from `BASIC5_ATTENTIVE_v1`, with the learning rate of the reference DPT probe

## Purpose

This protocol measures how much dense information a stronger frozen readout recovers beyond the single-layer 1×1 probe of `BASIC5_FAIR_v1`.

- The backbone stays completely frozen.
- The 1×1 head on one layer is replaced by a DPT probe that reads four evenly spaced intermediate blocks.
- Dataset, split, input geometry, epoch count, loss, and metric are those of `BASIC5_FAIR_v1`.

**Scope:** ADE20K and NYUv2. ImageNet-1k and SSv2 have no dense readout. COCO keeps the `BASIC5_FAIR_v1` feature pyramid and Faster R-CNN.

This file was written on 2026-11-19 from the code (`code_v9`) and the run records. The runs read the `BASIC5_*_v1.json` files and the DPT constants in the code; this file was not an input to them.

## Canonical skeleton

| Dataset | Epochs | Frozen readout | Optimizer | Primary metric |
|---|---:|---|---|---|
| ADE20K | 20 | DPT probe on 4 frozen taps, 150 outputs | AdamW `5e-4` @ 8, wd 0.05 | mIoU |
| NYUv2 | 30 | DPT probe on 4 frozen taps, 1 output (`softplus + 1e-3`) | AdamW `5e-4` @ 8, wd 0.05 | RMSE |

## Global rules

- Load every released checkpoint **as-is**.
- Keep the backbone in `eval()` with gradients disabled. Taps are extracted online under `no_grad`, with no feature cache.
- Train only the DPT probe: per-tap LayerNorm, per-tap 1×1 projections, four fusion blocks, and the output convolutions.
- One recipe per task for all six models. There is no model-specific learning rate or probe width, and no tap selection beyond the VGGT-Ω exception below.
- bfloat16 backbone weights and autocast; probe parameters, loss, and metrics in float32.
- Global-norm gradient clipping at 1.0.
- Weight decay 0 for biases, 1-D parameters, and parameters named `norm`.
- The learning rate advances per optimizer step.
- Seeds `0`, `1`, `2`.

## Taps

**Which blocks**
- Blocks `ceil(k·L/4) − 1` for k = 1…4 (0-based; L = number of blocks). This is DPT's evenly spaced choice for ViT-L ([5, 11, 17, 23]).
- **Exception:** VGGT-Ω uses the four aggregator layers that its own DPT heads read, (4, 11, 17, 23); the rule would give 5 in place of 4. The adapter checks that the cached set equals these layers.

**Which tensor**
- Block output tokens, with non-spatial special tokens removed, reshaped to the patch grid in raster order.
- RAEv2-K7 uses the official DINOv3 `get_intermediate_layers(norm=True)` of its DINOv3-L encoder.
- VGGT-Ω uses its cached aggregator outputs (frame and global tokens concatenated).
- Cosmos 3 Super taps its vision-tower blocks before the merger (1152-d, 14×14). Its LP/LoRA feature is the 5120-d merger output at 7×7.

| Model | Blocks | Tap blocks | Tap channels | Grid | Probe parameters (ADE20K / NYUv2) |
|---|---:|---|---:|---|---|
| DINOv3-7B | 40 | 9, 19, 29, 39 | 4096 | 14×14 | 44,511,894 / 43,825,153 |
| SigLIP 2-G | 40 | 9, 19, 29, 39 | 1536 | 14×14 | 39,248,534 / 38,561,793 |
| V-JEPA 2.1-G | 40 | 9, 19, 29, 39 | 1408 | 14×14 | 38,985,366 / 38,298,625 |
| VGGT-Ω | 24 | 4, 11, 17, 23 | 2048 | 14×14 | 40,301,206 / 39,614,465 |
| RAEv2-K7 | 24 | 5, 11, 17, 23 | 1024 | 14×14 | 38,195,862 / 37,509,121 |
| Cosmos 3 Super | 27 | 6, 13, 20, 26 | 1152 | 14×14 | 38,459,030 / 37,772,289 |

## DPT probe

The architecture follows the DPT probe of Probe3D (El Banani et al., CVPR 2024) as released in Geosubprobe (`evals/models/probes.py`, class `DPT`, config `depth_dpt`), with hidden width 512 and kernel 3.

1. **Per tap:** LayerNorm over channels → 1×1 Conv C → 512 → ×2 bilinear upsampling (`align_corners=true`).
2. **Fusion from the deepest tap to the shallowest:**
   - `fuse4(tap4)`, then `fuse3(tap3, ·)`, `fuse2(tap2, ·)`, `fuse1(tap1, ·)`.
   - Each fusion computes `ResidualConvUnit(tap) + previous`, then another `ResidualConvUnit`.
   - A `ResidualConvUnit` is Conv3×3 → ReLU → Conv3×3 → ReLU, plus the identity.
   - All taps share one grid, so the stages do not upsample between taps.
3. **Output:** ×4 bilinear upsampling (`align_corners=true`) → Conv3×3 512→512 → ReLU → Conv3×3 512→out → bilinear resize to the target size (`align_corners=false`).
4. **Outputs:** 150 logits for ADE20K. For NYUv2, one channel with depth `= softplus(x) + 1e-3`, the `BASIC5_FAIR_v1` parameterization.

Two changes from the reference, applied to every model:
- A LayerNorm over channels on each tap before its projection. The six backbones expose raw block outputs whose scales differ by orders of magnitude (pre-norm residual streams; Cosmos 3 Super block outputs are about 1e3).
- The reference's final fixed ×2 upsampling is replaced by a bilinear resize to the target size, as in the Unified LP readout.

## Task protocols

### ADE20K

- **Split:** official 20,210 train / 2,000 validation.
- **Input:**
  - Training: random rescale by 0.5–2.0 (bicubic, each side at least 224 px), random 224×224 crop, and horizontal flip with probability 0.5.
  - Evaluation: image resized to 224×224 (bicubic) and label resized to 224×224 (nearest).
  - Published backbone normalization.
- **Optimization:**
  - 20 epochs, AdamW, betas `(0.9, 0.999)`, weight decay `0.05`.
  - LR `5e-4` at batch 8, from Geosubprobe `probe_lr` with linear scaling.
  - Batch 8 on one GPU.
  - 1-epoch warm-up from `1e-6`, then cosine decay to `1e-6`.
- **Loss:** pixel-wise cross entropy with `ignore_index=255`.
- **Report:** mIoU over 150 classes primary (final epoch, 2,000 images); pixel accuracy secondary.

### NYUv2

- **Split:** official Eigen-style 795 train / 654 test from `labeled.mat` and `splits.mat`.
- **Input:**
  - Images and depths are read in their true 480×640 orientation (the basic5 loader transposed them).
  - Image resized to 224×224 (bicubic) and depth resized to 224×224 (bilinear); horizontal flip with probability 0.5 in training.
  - Valid depth is 0.1–10 m.
- **Optimization:** as for ADE20K, but with 30 epochs.
- **Loss:** scale-invariant log-depth loss on valid pixels with λ = 0.5: `mean(d²) − 0.5 · mean(d)²`, where `d = log pred − log gt`.
- **Report:**
  - Predictions are clamped to [0.1, 10] and evaluated on valid pixels with 0.1 < depth < 10 m.
  - The record's `rmse` is median-aligned per image. The paper reports the unaligned `rmse_unaligned`, as defined in its evaluation appendix.
  - AbsRel (aligned and unaligned), δ1, δ2, and δ3 are secondary.

## Reporting and comparability

- Use the final scheduled epoch, with no selection on the reported split.
- Report every seed plus the mean ± sample SD (ddof = 1) over seeds 0, 1, and 2.
- A cell counts only if all of the following hold:
  - it has `COMPLETED_VALID.json`;
  - it reached the recipe's epoch count;
  - steps per epoch × epochs = total steps;
  - it was evaluated on the full split.
- If a numerical fix is needed, apply it to every model, increment the version, and disclose it.

## Version history

| Version | ID | Date | Recipe | Outcome |
|---|---|---|---|---|
| 1.0 | `B5X_DPT_LP_v1` | 2026-11-11 | Probe at the `BASIC5_ATTENTIVE_v1` rate (`1e-3` at batch 8) | On ADE20K, the training loss of DINOv3-7B, V-JEPA 2.1, and RAEv2 rose after warm-up and ended above that of the 1×1 linear probe; these are the three models where DPT scored below LP. Seed spread was 3–5× that of LP/LoRA. Cells kept under `DPT/`. |
| 2.0 | `B5X_DPT_LP_v2` | 2026-11-12 | Probe at the reference DPT probe's rate (`5e-4` at batch 8, linear scaling); everything else unchanged | Applied to all six models and both tasks. The rate comes from the reference implementation, not from the reported split. |

## Deviations from the Unified code (all six models)

- NYUv2 is read upright.
- The learning-rate schedule advances per optimizer step.

## Where things are

- Code: `${EXPERIMENT_ROOT}/code_v9` (`b5x/dpt.py`, `train.py`, `backbones.py`).
- Protocol note: `${EXPERIMENT_ROOT}/PROTOCOL.md`.
- Records: `${EXPERIMENT_ROOT}/results/<model>/{ade20k,nyuv2}/DPT_v2/seed{0,1,2}/result.json`.
