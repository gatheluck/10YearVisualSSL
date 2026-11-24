# BASIC5_FAIR_v1

**Status:** canonical v1.0  
**Track:** frozen backbone / minimum task-appropriate probe

## Purpose

`BASIC5_FAIR_v1` is the cross-family reference protocol. Released pretrained weights are used as-is, the backbone remains completely frozen in evaluation mode, and only the minimum task head is trained. Each dataset uses one standard-inspired recipe shared by every backbone; model-specific tuning is prohibited.

| Dataset | Canonical probe | Schedule | Primary metric |
|---|---|---:|---|
| ImageNet-1k | linear classification | 100 epochs | Top-1 |
| COCO 2017 | Faster R-CNN detection | 12 epochs / 1× | bbox AP |
| ADE20K | spatial 1×1 Conv segmentation | 20 epochs | mIoU |
| NYUv2 | spatial 1×1 Conv depth | 30 epochs; official 795/654 | RMSE |
| SSv2 | linear action recognition | 50 epochs; 16 frames | Top-1 |

## Global rules

- Use released pretrained checkpoints without intermediate adaptation.
- Freeze every backbone parameter, normalization affine parameter, BatchNorm statistic, and positional embedding; keep the backbone in `eval()`.
- Use one canonical final feature layer. Do not search layers or concatenate multiple layers.
- ADE20K and NYUv2 require genuine spatial tokens/features. Otherwise report **UNSUPPORTED**.
- Use seeds `0`, `1`, and `2`; report each result and mean ± standard deviation.
- Model-specific LR sweeps, resolution changes, head changes, and other tuning are prohibited.

## Canonical task settings

### ImageNet-1k

Official ILSVRC2012 train/validation; 224 training crop and 224 center-crop evaluation; L2-normalized final global feature; linear 1000-class head. Train only the head for 100 epochs with SGD, LR `0.1` at effective batch 256 with linear scaling, momentum `0.9`, zero weight decay, and cosine decay. Use random resized crop and horizontal flip only. Report Top-1 and Top-5.

### COCO 2017

train2017/val2017; Faster R-CNN with FPN, RPN, RoIAlign, classifier, and regressor trained over a frozen backbone. Resize to 800–1333 and use horizontal flip during training. Use the 12-epoch 1× recipe: SGD `0.02` at batch 16, momentum `0.9`, weight decay `1e-4`, 500-iteration warmup, and ×0.1 LR steps at epochs 8 and 11. Report bbox AP@[0.50:0.95], AP50, AP75, APS, APM, and APL.

### ADE20K

Official 20,210/2,000 split; final spatial tokens, a trainable 150-output 1×1 Conv, and bilinear upsampling. Train for 20 epochs with AdamW, LR `1e-3` at batch 8, weight decay `1e-4`, one-epoch warmup, and cosine decay. Use random scale 0.5–2.0, 224 crop, and horizontal flip. Report mIoU and pixel accuracy.

### NYUv2

Official Eigen-style `labeled.mat`/`splits.mat` split with 795 train and 654 test images; valid depth 0.1–10.0 m. Use final spatial tokens, a single-channel 1×1 Conv, positive depth parameterization, and bilinear upsampling. Train for 30 epochs with AdamW, LR `1e-3` at batch 8, weight decay `1e-4`, one-epoch warmup, and cosine decay. Use the fixed valid-pixel scale-invariant log-depth loss. Report RMSE, AbsRel, δ1, δ2, and δ3.

### Something-Something v2

Official train/validation; 16 equal temporal segments with random training samples and segment-center evaluation samples. Use temporally consistent 224 crops and no horizontal flip. For image backbones, spatially mean-pool each frame then temporally mean-pool; for native video backbones, use the canonical final global representation. Train only the 174-class linear head for 50 epochs with SGD, LR `0.1` at effective batch 256, momentum `0.9`, weight decay `1e-4`, five-epoch warmup, and cosine decay. Report Top-1 and Top-5.

## Reporting

Use the final scheduled epoch for the canonical score; never select on the reported validation/test split. Record checkpoint checksum, backbone/parameter count, feature layer and shape, resolution/views, effective batch and LR, all per-seed metrics, software revision, and dataset version. Only untuned conforming runs enter the main table.

Native resolution, COCO 2×/24 epochs, SSv2 native clip length, and multi-view evaluation are separate optional tracks and must not be mixed into the canonical table.
