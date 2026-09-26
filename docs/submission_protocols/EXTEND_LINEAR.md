# EXTEND_LINEAR_v1

**Status:** canonical v1.0  
**Track:** frozen backbone / minimum task-appropriate probe  
**Trial unit:** one model x one dataset x seed 0

## Purpose

`EXTEND_LINEAR_v1` extends the Basic Five frozen-backbone philosophy to the full downstream catalog. The released backbone is kept completely frozen in evaluation mode. Only the smallest practical task readout is trained. Settings are shared at the task-family level and overridden per dataset only for splits, label spaces, metrics, geometry, or other genuine dataset requirements.

The protocol model set is:

- CLIP
- SigLIP-Giant
- SAM3
- C-RADIOv4-H
- DINOv3-7B
- RAEv2-K7
- V-JEPA 2.1
- VGGT-omega

This protocol is LINEAR-only. It must not launch attentive probing or full fine-tuning.

## Execution policy

- One logical job is exactly `model x dataset x EXTEND_LINEAR_v1 x seed 0`.
- Run one trial only. Report the seed-0 result, not mean +/- standard deviation.
- Submit every eligible logical job without manually dividing the campaign into waves. PBS may queue jobs when nodes are unavailable.
- Reuse a valid Basic Five LINEAR result when the dataset, model, split, head, metric, and seed are equivalent.
- Resume an interrupted run under the same logical ID. Never duplicate a healthy owner.
- A failure in one cell must not stop other dataset jobs.

## Lightweight preflight

The campaign is intentionally attempt-first. Before a full job, check only that:

1. the dataset and required annotation file open;
2. the label mapping is nonempty;
3. one mini-batch completes frozen feature extraction, loss computation, and head-only backward;
4. the backbone has no gradients.

Do not require a strict canary, multi-seed reproducibility, full annotation audit, or model-specific tuning before launch. If a run fails, record the reason, continue other jobs, and repair/retry that logical job later.

## Dataset selection

The source list contains 100 named entries. `EXTEND_LINEAR_v1` contains **81 trainable dataset configurations** after the following normalization:

- DomainNet and its separately listed subdomains are one six-domain benchmark.
- Industrial iSeg and its `insp-wqisd` path are one benchmark.
- Exclude ImageNet-21k, ImageNet-100, Kinetics-700, FractalDB-WN60, and Places30 for scale, redundancy, or limited downstream value.
- Keep ImageNet-1% and ImageNet-10% as low-label-efficiency benchmarks.
- Keep Kinetics-400 instead of Kinetics-700. K400 is less expensive and more directly comparable with common frozen video representation evaluations.
- Do not schedule CIFAR-10C, CIFAR-100C, ImageNet-A/C/R/Sketch, ObjectNet, or Places365-past because they have no new probe-training role in this campaign.
- Do not schedule PASS because the listed corpus does not provide a supervised downstream target for a head-only probe.

Five configurations overlap the Basic Five and are intentionally retained in the protocol catalog so that the expanded table is complete. Their valid LINEAR seed-0 results should be imported rather than trained again. Therefore, the campaign contains **76 new datasets per model**, or **608 new logical jobs** for all eight models. If Basic Five is deliberately rerun under the new ID, the total would be 648 jobs.

The jobs do not have to be submitted for all models at once. A recommended incremental unit is one complete 76-dataset model campaign. For example, running two models uses 152 logical jobs while preserving the same protocol and result layout.

## Shared recipes

| Family | Canonical schedule | Minimum readout | Primary metric |
|---|---:|---|---|
| Image classification | 100 epochs | linear classifier | Top-1 |
| Video classification | 50 epochs, 16 frames | linear classifier | Top-1 |
| Multi-label classification | 50 epochs | linear multi-label classifier | mAP |
| Detection | 12 epochs / 1x | FPN + Faster R-CNN head | bbox AP |
| Instance segmentation | 12 epochs / 1x | FPN + Mask R-CNN head | mask AP |
| Semantic segmentation | 20 epochs | 1x1 Conv | mIoU |
| Monocular depth | 30 epochs | positive 1x1 depth Conv | AbsRel/RMSE |
| Human pose | 30 epochs | 1x1 heatmap head | AP/PCKh |
| Tracking | 30 epochs | linear projection + fixed correlation | dataset standard |
| Optical flow | 30 epochs | correlation + shallow linear flow head | EPE |
| Matching / MVS | 30 epochs | linear descriptor projection | pose/reconstruction metric |
| Localization | 30 epochs | linear global/spatial descriptor projection | recall or pose error |
| Visual reasoning / VQA | 50 epochs | smallest repository task head over frozen visual tokens | accuracy |
| Action detection | 12 epochs | RoIAlign + linear multi-label action head | frame mAP |

The exact optimizer, LR, weight decay, warmup, schedule, augmentation, feature rule, split, head output size, and metric for every dataset are normative in `EXTEND_LINEAR_v1.json`.

## Feature and tuning rules

- Use the released checkpoint without intermediate adaptation.
- Freeze every backbone parameter, normalization affine value/statistic, and positional embedding. Keep the backbone in evaluation mode.
- Use the model adapter's canonical final feature. Do not search layers or learn a layer mixture.
- Use global features for classification, spatial features for dense/correspondence tasks, and temporal features for video tasks.
- Model-specific LR sweeps, resolution changes, extra decoder depth, and head-capacity tuning are prohibited.
- Physical batch size may be reduced for memory, provided gradient accumulation preserves the effective batch and LR rule.

## Reporting

Use the final scheduled epoch as the canonical score. Record the logical job ID, checkpoint identifier, dataset path/version, split, recipe and overrides, feature shape, effective batch/LR, final metric, and exit state. Failed cells should remain visible as `FAILED_RETRYABLE`, `UNSUPPORTED_FEATURE`, `MISSING_ANNOTATION`, `OUT_OF_MEMORY`, or `WALLTIME`; they may be diagnosed and retried later without blocking the campaign.

Only runs without model-specific tuning belong in the main `EXTEND_LINEAR_v1` table.
