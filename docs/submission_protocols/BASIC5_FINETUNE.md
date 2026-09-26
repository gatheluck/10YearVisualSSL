# BASIC5_FINETUNE_v1

**Status:** publication draft based on canonical v1.0; replication profiles proposed, not an execution audit  
**Track:** full fine-tuning  
**Baseline:** `BASIC5_FAIR_v1`

## Purpose

This protocol measures downstream adaptability by training the released pretrained backbone and the same basic task head used by `BASIC5_FAIR_v1` end-to-end. It preserves dataset splits, input definitions, epoch counts, task heads, and metrics wherever possible. Changes are restricted to those needed for stable full fine-tuning: optimizer, LR, weight decay, warmup/schedule, layer-wise LR decay, and regularizing augmentation.

It is the third point in the controlled comparison:

1. `BASIC5_FAIR_v1`: minimal linear/frozen readout.
2. `BASIC5_ATTENTIVE_v1`: stronger readout, still a frozen backbone.
3. `BASIC5_FINETUNE_v1`: baseline task head with an adaptable backbone.

## Canonical skeleton

| Dataset | Epochs / schedule | Unchanged task head | Primary metric |
|---|---:|---|---|
| ImageNet-1k | 100 epochs | linear classifier | Top-1 |
| COCO | 12 epochs / 1x | Faster R-CNN + FPN | bbox AP |
| ADE20K | 20 epochs | 1x1 Conv | mIoU |
| NYUv2 | 30 epochs | 1x1 Conv | RMSE |
| SSv2 | 50 epochs / 16 frames | linear classifier | Top-1 |

## Global rules

- Initialize from each released pretrained checkpoint as-is. No intermediate task adaptation is allowed.
- Train the entire backbone and the same task head used by the frozen-linear baseline. A stronger decoder is not allowed: ADE20K and NYUv2 keep 1x1 heads, while COCO keeps Faster R-CNN/FPN.
- Use the same canonical feature output and token-to-grid conversion as `BASIC5_FAIR_v1`. Feature caching is forbidden.
- Train normalization affine parameters. BatchNorm backbones update running statistics during training and use them at evaluation; SyncBN is allowed where required by effective batch size.
- Use one task recipe for every backbone. Model-specific LR, layer decay, augmentation, feature-layer, or head tuning is prohibited.
- Select THREE_SEEDS ([0, 1, 2]) or SINGLE_SEED ([0]) under the shared replication policy in 00_scope_and_replication.md. Mixed precision, activation checkpointing, accumulation, and parameter sharding are allowed only when they preserve optimization semantics. Clip gradients to global norm 1.0.

### Parameter-group rule

The listed base LR is the head/output-side LR. The head receives multiplier 1.0. Starting from the latest backbone block, multiply the LR by the task's layer-decay factor once for each successively earlier block; embeddings/stem receive the smallest LR. Biases, normalization affine parameters, positional embeddings, and class tokens receive zero weight decay. Their LR is still determined by depth.

## Task protocols

### ImageNet-1k

- **Split/input:** official ILSVRC2012 train / validation; training random resized crop to 224 (scale 0.08-1.0, ratio 0.75-1.3333) and horizontal flip 0.5; evaluation resize-shorter-side 256 plus 224 center crop; published backbone normalization.
- **Feature/pooling:** retain the baseline's canonical final global representation, such as CLS or global average pooling.
- **Head/trainable scope:** unchanged 1000-class linear classifier; entire backbone and classifier trainable.
- **Optimization:** 100 epochs; AdamW; base LR `5e-4` at effective batch 1024 with linear batch scaling; batch 1024; weight decay `0.05`; betas `(0.9, 0.999)`; 5-epoch warmup from `1e-6`; cosine decay to `1e-6`; layer decay `0.75`.
- **Augmentation:** RandAugment `9/0.5`, Mixup alpha=`0.8`, CutMix alpha=`1.0`, application probability `1.0`, label smoothing `0.1`, random erasing probability `0.25`.
- **Loss/report:** soft-target cross entropy under Mixup/CutMix, otherwise label-smoothed cross entropy. Report Top-1 primary and Top-5 secondary.
- **Minimal deviation:** 100 epochs, split, resolution, pooling, classifier, and metrics remain fixed. AdamW, lower scaled LR, weight/layer decay, warmup, and regularization are the end-to-end changes.

### COCO 2017

- **Split/input:** train2017 / val2017; shorter side 800 and longer side at most 1333; training flip probability 0.5; published backbone normalization.
- **Feature/pooling:** use the same spatial/multiscale features and FPN mapping as the baseline; never globally pool.
- **Head/trainable scope:** unchanged Faster R-CNN with FPN, RPN, RoIAlign, box classifier, and box regressor; entire backbone and detector trainable.
- **Optimization:** 12 epochs / 1x; SGD; LR `0.02` at effective batch 16 with linear scaling; batch 16; momentum `0.9`; weight decay `1e-4`; 500-iteration linear warmup with factor `0.001`; LR x0.1 at epochs 8 and 11; layer decay disabled (`1.0`).
- **Augmentation/loss:** unchanged scale resize and horizontal flip only; standard Faster R-CNN losses.
- **Report:** bbox AP@[0.50:0.95] primary; AP50, AP75, APS, APM, and APL secondary.
- **Minimal deviation:** this is already the standard end-to-end 1x recipe. The sole protocol change from the frozen baseline is unfreezing the backbone and its normalization.

### ADE20K

- **Split/input:** official 20,210 train / 2,000 validation; random scale 0.5-2.0, random 224x224 crop, flip 0.5; single-scale 224 evaluation under the baseline policy.
- **Feature/pooling:** reshape the same final patch tokens to a 2-D grid, apply the unchanged 1x1 Conv, and bilinearly upsample with `align_corners=false`.
- **Head/trainable scope:** unchanged 150-output 1x1 Conv; entire backbone and head trainable. UPerNet, FPN, auxiliary decoders, and multilevel fusion are prohibited.
- **Optimization:** 20 epochs; AdamW; base LR `1e-4` at effective batch 8 with linear scaling; batch 8; weight decay `0.05`; betas `(0.9, 0.999)`; 1-epoch warmup from `1e-6`; cosine decay to `1e-6`; layer decay `0.8`.
- **Augmentation/loss:** retain baseline geometry and add color jitter `0.4` only. Pixel-wise cross entropy with `ignore_index=255`; no auxiliary loss.
- **Report:** mIoU over 150 classes primary; pixel accuracy secondary.
- **Minimal deviation:** 20 epochs, 224 input, final feature, 1x1 head, upsampling, loss, and metrics remain fixed. Only backbone trainability and the low-LR AdamW/layer-decay recipe with mild color jitter change.

### NYUv2

- **Split/input:** official Eigen-style 795 train / 654 test from `labeled.mat` and `splits.mat`; baseline 224x224 geometry and valid-pixel mask; valid depth 0.1-10.0 m.
- **Feature/pooling:** reshape the same final patch tokens to a 2-D grid, apply the unchanged depth head, and bilinearly upsample with `align_corners=false`.
- **Head/trainable scope:** unchanged single-channel 1x1 Conv with positive depth parameterization; entire backbone and head trainable. DPT and multiscale decoders are prohibited.
- **Optimization:** 30 epochs; AdamW; base LR `1e-4` at effective batch 8 with linear scaling; batch 8; weight decay `0.05`; betas `(0.9, 0.999)`; 1-epoch warmup from `1e-6`; cosine decay to `1e-6`; layer decay `0.8`.
- **Augmentation/loss:** retain baseline geometry, add color jitter `0.2`, and use no Mixup, CutMix, or synthetic depth data. Keep the baseline valid-pixel scale-invariant log-depth loss.
- **Report:** RMSE in meters primary; AbsRel, delta1, delta2, and delta3 secondary.
- **Minimal deviation:** official split, valid range, 30 epochs, 1x1 head, loss, and metrics remain fixed. Conservative low-LR fine-tuning and mild jitter address the small training set.

### Something-Something v2

- **Split/input:** official train / validation; 16 equal temporal segments with random within-segment training samples and segment-center evaluation samples; temporally consistent 224 random resized crop; one clip and one center crop at evaluation; horizontal flip disabled.
- **Feature/pooling:** retain the baseline rule: temporal mean of per-frame global features for image backbones, or the canonical final global representation for native video backbones.
- **Head/trainable scope:** unchanged 174-class linear classifier; entire backbone and classifier trainable.
- **Optimization:** 50 epochs; AdamW; base LR `5e-4` at effective batch 256 with linear scaling; physical batch 64 and accumulation as needed; weight decay `0.05`; betas `(0.9, 0.999)`; 5-epoch warmup from `1e-6`; cosine decay to `1e-6`; layer decay `0.75`.
- **Augmentation:** temporally consistent RandAugment `9/0.5`, Mixup alpha=`0.8`, CutMix alpha=`1.0`, application probability `1.0`, label smoothing `0.1`, random erasing probability `0.25`, and no horizontal flip.
- **Loss/report:** soft-target cross entropy under Mixup/CutMix, otherwise label-smoothed cross entropy. Report Top-1 primary and Top-5 secondary.
- **Minimal deviation:** 16 frames, 50 epochs, pooling, classifier, and single-view evaluation remain fixed. AdamW, lower LR, weight/layer decay, warmup, and video fine-tuning regularization are the necessary changes.

## Reporting and comparability

- Use the final scheduled epoch for the canonical score; do not select on the reported validation/test split. A best-validation checkpoint is allowed only as a labeled secondary result when a separate validation split exists.
- Follow the shared replication policy: report all individual results and actual n; THREE_SEEDS uses a mean and standard deviation, whereas SINGLE_SEED reports the seed-0 score without a standard deviation.
- Record checkpoint ID/checksum, backbone family and parameter count, trainable/total parameters, feature layer/shape, input resolution/views, effective batch, realized head and stem LRs, peak accelerator memory, training compute, software revision, and dataset version.
- Mark unsupported tasks **UNSUPPORTED** with a reason. Never replace a task, split, head, or feature type silently.
- Only untuned runs conforming to this file may enter the `BASIC5_FINETUNE_v1` main table.
- Memory-only changes--accumulation, checkpointing, and FSDP/parameter sharding--must preserve effective batch, realized LR, precision, evaluation views, and optimization semantics.
- If a numerical fix is necessary, apply it protocol-wide, increment the version, and disclose it; do not tune one failing model.

## Separate optional tracks

Native resolution, COCO 2x / 24 epochs, stronger ADE/NYU decoders such as UPerNet or DPT, SSv2 native clip lengths, and multi-view video testing are separate tracks and must not be mixed into the canonical table.
