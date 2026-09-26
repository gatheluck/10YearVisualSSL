# EXTEND_FINETUNE_v1

**Status:** canonical v1.0  
**Track:** full fine-tuning  
**Parent:** `EXTEND_LINEAR_v1`  
**Trial unit:** one model x one dataset x seed 0

## Definition

This protocol inherits all 81 dataset definitions, paths, splits, input geometry, epoch counts, task heads, losses, and metrics from `EXTEND_LINEAR_v1`. It unfreezes the entire released backbone and trains it end-to-end with the same minimum task head. It does not include the attentive reader.

The protocol covers CLIP, SigLIP-G, SAM3, C-RADIOv4-H, DINOv3-7B, RAEv2-K7, V-JEPA2.1, and VGGT-omega.

FT jobs are independent. They must not launch or wait for LP/AP. Use seed 0 once per model-dataset cell. Valid equivalent `BASIC5_FINETUNE_v1` seed-0 results for the five Basic Five datasets should be imported rather than rerun.

## Invariants

- Train every backbone parameter and the unchanged LINEAR task head.
- Retain the parent feature output, token-to-grid conversion, split, epoch count, loss, and metric.
- Do not introduce a stronger decoder. For example, a 1x1 segmentation/depth head remains a 1x1 head.
- Do not use LoRA, partial freezing, quantized/distilled replacements, or smaller substitute models.
- Feature caching is forbidden.
- AMP/bfloat16, activation checkpointing, accumulation, FSDP/sharding, and a smaller physical batch are permitted when they preserve effective batch, LR, and optimizer semantics.

## Optimization

Image and video classification use AdamW, base LR `5e-4`, weight decay `0.05`, five-epoch warmup, cosine decay, and layer decay `0.75`. Small-image classification and smaller/specialized tasks use the more conservative base LR `1e-4`. Semantic segmentation, depth, pose, tracking, flow, matching, localization, reasoning, and action detection use AdamW `1e-4` with layer decay `0.8`. Detection and instance segmentation retain the standard SGD 1x recipe.

The latest backbone block receives the listed base LR; each earlier block is multiplied once by the layer-decay factor. Biases, normalization affine parameters, positional embeddings, and class/register/prefix tokens receive zero weight decay.

Classification/video fine-tuning adds standard RandAugment, Mixup/CutMix where labels permit, label smoothing, and random erasing. Dense and geometry tasks use only mild task-preserving photometric/geometry augmentation. Exact settings are normative in the JSON file.

## Execution and reporting

One PBS job is one model x one dataset x `EXTEND_FINETUNE_v1` x seed 0. Jobs may be submitted incrementally when sufficient nodes are available; no protocol chaining is allowed. Skip valid results and healthy owners, resume incomplete checkpoints under the same logical ID, and isolate failures.

Use the final scheduled epoch. Report trainable/total parameters, feature shape, effective batch, realized head and stem LRs, layer-group mapping, peak memory, distributed strategy, final seed-0 metric, and exit state. Do not report mean +/- standard deviation.

No model-specific optimizer, LR, layer-decay, augmentation, layer, resolution, or head-capacity search is permitted.
