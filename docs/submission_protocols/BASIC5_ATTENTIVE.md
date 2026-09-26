# BASIC5_ATTENTIVE_v1

**Status:** publication draft based on canonical v1.0; replication profiles proposed, not an execution audit  
**Track:** attentive probing  
**Baseline:** `BASIC5_FAIR_v1`

## Purpose

This protocol measures how well a frozen pretrained representation can be read by one small, shared attention module. It preserves the dataset, split, input, epoch count, basic task head, and evaluation metric of `BASIC5_FAIR_v1`. The backbone stays completely frozen; only the minimal attention reader/adapter and the baseline task head are trained.

This is the middle point in the three-track comparison:

1. `BASIC5_FAIR_v1`: minimal linear/frozen readout.
2. `BASIC5_ATTENTIVE_v1`: stronger readout, still a frozen backbone.
3. `BASIC5_FINETUNE_v1`: the same basic task head with an adaptable backbone.

## Canonical skeleton

| Dataset | Epochs / schedule | Frozen feature readout | Primary metric |
|---|---:|---|---|
| ImageNet-1k | 100 epochs | 1-block query attention + linear classifier | Top-1 |
| COCO | 12 epochs / 1x | spatial attention adapter + unchanged Faster R-CNN | bbox AP |
| ADE20K | 20 epochs | spatial attention adapter + 1x1 Conv | mIoU |
| NYUv2 | 30 epochs | spatial attention adapter + 1x1 Conv | RMSE |
| SSv2 | 50 epochs / 16 frames | temporal/spatiotemporal query attention + linear classifier | Top-1 |

## Global rules

- Load every released pretrained checkpoint **as-is**. Do not perform intermediate adaptation or task-specific pretraining.
- Keep the backbone in `eval()` with gradients disabled. Freeze normalization affine parameters, BatchNorm running statistics, positional embeddings, and all other backbone state.
- Train only the attention reader/adapter, its input/output projections and normalization, and the task head already used by `BASIC5_FAIR_v1`.
- Use the same single canonical feature layer as the frozen-linear baseline, normally the final pre-classifier representation. Do not concatenate extra layers.
- Dense tasks require real spatial tokens or a spatial feature map. If these are unavailable, report **UNSUPPORTED** rather than substituting a global feature.
- Use one task recipe for all backbones. Model-specific LR sweeps, layer selection, reader widths, query counts, or other tuning are prohibited.
- Select THREE_SEEDS ([0, 1, 2]) or SINGLE_SEED ([0]) under the shared replication policy in 00_scope_and_replication.md. Mixed precision is allowed, but sensitive losses and metrics remain float32. Clip trainable gradients to global norm 1.0.

## Shared minimal reader

The reader has one pre-LayerNorm attention/MLP block, 8 attention heads, MLP ratio 4, and no attention dropout, projection dropout, or stochastic depth.

For image classification and video recognition, project inputs to width 512 and use 32 learned queries in one cross-attention block. Mean-pool the 32 query outputs before the unchanged linear classifier.

For dense prediction and detection, use a residual spatial bottleneck adapter:

`C -> 256 -> one self-attention/MLP block -> C -> unchanged task head`

The final `256 -> C` projection is zero-initialized. For multiscale detection features, apply the same adapter weights independently at each selected scale. Report reader parameters, task-head parameters, and their percentage of total parameters.

## Task protocols

### ImageNet-1k

- **Split:** official ILSVRC2012 train / validation.
- **Input:** training uses random resized crop to 224 (scale 0.08-1.0, ratio 0.75-1.3333) and horizontal flip with probability 0.5. Evaluation resizes the shorter side to 256 and center-crops 224. Use the backbone's published normalization.
- **Feature/pooling:** all final-layer patch tokens attend to 32 learned queries; mean-pool query outputs. Add neither CLS nor global tokens unless they are the model's only canonical token representation.
- **Head:** unchanged 1000-class linear classifier.
- **Trainable:** reader projections/norms, learned queries, reader block, and classifier only.
- **Optimization:** 100 epochs; AdamW; LR `1e-3` at effective batch 256 with linear batch scaling; batch 256; weight decay `0.05`; betas `(0.9, 0.999)`; 5-epoch linear warmup from `1e-6`; cosine decay to `1e-6`; layer decay disabled (`1.0`).
- **Augmentation/loss:** retain frozen-linear geometry only. No Mixup, CutMix, label smoothing, RandAugment, or random erasing. Use cross entropy.
- **Report:** Top-1 primary and Top-5 secondary.
- **Minimal deviation:** only the attention reader precedes the linear head. AdamW replaces probe SGD because learned queries, attention, and normalization are now trained.

### COCO 2017

- **Split:** train2017 / val2017.
- **Input:** shorter side 800, longer side at most 1333; random horizontal flip with probability 0.5 during training; backbone-specific published normalization.
- **Feature/pooling:** preserve spatial/multiscale features. Insert the shared spatial adapter immediately before the unchanged FPN. Never globally pool.
- **Head:** unchanged Faster R-CNN with FPN, RPN, RoIAlign, box classifier, and box regressor.
- **Trainable:** spatial adapter and the complete Faster R-CNN/FPN detector; backbone fully frozen.
- **Optimization:** 12 epochs / 1x; SGD; LR `0.02` at effective batch 16 with linear scaling; batch 16; momentum `0.9`; weight decay `1e-4`; 500-iteration linear warmup with factor `0.001`; LR x0.1 at epochs 8 and 11; layer decay disabled.
- **Augmentation/loss:** unchanged resize and flip only; standard Faster R-CNN losses.
- **Report:** bbox AP@[0.50:0.95] primary; AP50, AP75, APS, APM, and APL secondary.
- **Minimal deviation:** detector, schedule, optimizer, batch size, and augmentation stay fixed. Only the frozen-feature attention adapter is added.

### ADE20K

- **Split:** official 20,210 train / 2,000 validation.
- **Input:** random scale 0.5-2.0, random 224x224 crop, horizontal flip with probability 0.5; single-scale 224 evaluation under the baseline resize/crop policy.
- **Feature/pooling:** reshape final patch tokens to their 2-D grid, apply the residual spatial adapter and unchanged 1x1 Conv, then bilinearly upsample with `align_corners=false`.
- **Head:** unchanged 1x1 Conv with 150 outputs.
- **Trainable:** spatial adapter and 1x1 Conv only.
- **Optimization:** 20 epochs; AdamW; LR `1e-3` at effective batch 8 with linear scaling; batch 8; weight decay `0.05`; betas `(0.9, 0.999)`; 1-epoch warmup from `1e-6`; cosine decay to `1e-6`; no layer decay.
- **Augmentation/loss:** retain baseline geometry and photometric policy. Use pixel-wise cross entropy, `ignore_index=255`, with no auxiliary loss.
- **Report:** mIoU over 150 classes primary; pixel accuracy secondary.
- **Minimal deviation:** split, 20 epochs, feature layer, 1x1 head, upsampling, and metric remain fixed; one spatial adapter is inserted.

### NYUv2

- **Split:** official Eigen-style 795 train / 654 test from `labeled.mat` and `splits.mat`.
- **Input:** baseline 224x224 train/evaluation geometry, horizontal flip with probability 0.5 for training, and the same valid-pixel mask. Valid depth is 0.1-10.0 m.
- **Feature/pooling:** reshape final patch tokens to their 2-D grid, apply the residual spatial adapter and unchanged 1x1 depth head, then bilinearly upsample with `align_corners=false`.
- **Head:** unchanged single-channel 1x1 Conv with positive depth parameterization.
- **Trainable:** spatial adapter and depth head only.
- **Optimization:** 30 epochs; AdamW; LR `1e-3` at effective batch 8 with linear scaling; batch 8; weight decay `0.05`; betas `(0.9, 0.999)`; 1-epoch warmup from `1e-6`; cosine decay to `1e-6`; no layer decay.
- **Augmentation/loss:** retain baseline geometry; no Mixup, CutMix, or task-specific pretraining. Use the baseline valid-pixel scale-invariant log-depth loss.
- **Report:** RMSE in meters primary; AbsRel, delta1, delta2, and delta3 secondary.
- **Minimal deviation:** official split, mask/range, 30 epochs, 1x1 head, loss, and metrics stay fixed; only the spatial adapter and its optimizer-appropriate weight decay are added.

### Something-Something v2

- **Split:** official train / validation.
- **Input:** 16 equal temporal segments. Sample randomly within each segment for training and use each segment center for evaluation. Use one clip and one 224 center crop at evaluation. Training uses a temporally consistent random resized crop; horizontal flip is disabled.
- **Feature/pooling:** for image backbones, spatially mean-pool each frame's patch tokens and pass the 16 ordered frame tokens to the query reader. For native video backbones, pass the ordered final-layer spatiotemporal tokens directly. Mean-pool query outputs; do not temporal-mean-pool before attention.
- **Head:** unchanged 174-class linear classifier.
- **Trainable:** reader projections/norms, learned queries, reader block, and classifier only.
- **Optimization:** 50 epochs; AdamW; LR `1e-3` at effective batch 256 with linear scaling; physical batch 64 with accumulation as needed; weight decay `0.05`; betas `(0.9, 0.999)`; 5-epoch warmup from `1e-6`; cosine decay to `1e-6`; no layer decay.
- **Augmentation/loss:** retain baseline spatial/temporal policy. No Mixup, CutMix, RandAugment, random erasing, or horizontal flip. Use cross entropy.
- **Report:** Top-1 primary and Top-5 secondary.
- **Minimal deviation:** 16 frames, 50 epochs, classifier, and evaluation views stay fixed. Ordered-token attention replaces the baseline's pre-classifier temporal mean pooling.

## Reporting and comparability

- Use the final scheduled epoch for the canonical score; do not select on the reported validation/test split. A best-validation checkpoint is allowed only as a labeled secondary result when a separate validation split exists.
- Follow the shared replication policy: report all individual results and actual n; THREE_SEEDS uses a mean and standard deviation, whereas SINGLE_SEED reports the seed-0 score without a standard deviation.
- Record checkpoint ID/checksum, backbone family and size, feature layer/shape, reader/head parameter counts, input resolution/views, effective batch and realized LR, software revision, and dataset version.
- Mark unavailable spatial tasks **UNSUPPORTED** with a reason. Never replace the task or feature type silently.
- Only untuned runs conforming to this file may enter the `BASIC5_ATTENTIVE_v1` main table.
- If a numerical fix is necessary, apply it protocol-wide, increment the version, and disclose it; do not tune one failing model.

## Separate optional tracks

Native resolution, COCO 2x / 24 epochs, SSv2 native clip lengths, and multi-view video testing must be labeled as separate tracks and must not be mixed into the canonical table.
