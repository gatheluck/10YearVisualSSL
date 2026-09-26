# EXTEND_ATTENTIVE_v1

**Status:** canonical v1.0  
**Track:** frozen backbone / minimal attentive probe  
**Parent:** `EXTEND_LINEAR_v1`  
**Trial unit:** one model x one dataset x seed 0

## Definition

This protocol inherits all 81 dataset definitions, paths, splits, inputs, epoch counts, task heads, losses, and metrics from `EXTEND_LINEAR_v1`. The backbone remains fully frozen. The only architectural addition is one shared attention reader/adapter placed before the unchanged task head.

The protocol covers CLIP, SigLIP-G, SAM3, C-RADIOv4-H, DINOv3-7B, RAEv2-K7, V-JEPA2.1, and VGGT-omega.

AP jobs are independent. They must not launch or wait for LP or FT. Use seed 0 once per model-dataset cell. Valid equivalent `BASIC5_ATTENTIVE_v1` seed-0 results for the five Basic Five datasets should be imported rather than rerun.

## Reader

- Global image tasks: 32 learned queries, width 512, one pre-LN cross-attention/MLP block, and mean pooling over query outputs.
- Video tasks: the same reader consumes ordered temporal or spatiotemporal tokens. Do not temporal-mean-pool before attention.
- Dense tasks: `C -> 256 -> one residual spatial self-attention/MLP block -> C -> unchanged head`; the final projection is zero-initialized.
- Paired/multiview tasks: apply the same spatial-reader weights independently to each view, then retain the fixed correlation, matching, or localization pipeline.
- Attention has 8 heads, MLP ratio 4, zero dropout, and zero stochastic depth.

Train only the reader, its projections/norms, and the existing task head. The complete backbone, normalization state, positional embeddings, and special tokens remain frozen.

## Optimization

Keep every parent epoch count. Classification, video, dense, geometry, and reasoning readers use AdamW with base LR `1e-3`, linear batch scaling, weight decay `0.05`, warmup, cosine decay, and no layer decay. Detection, instance segmentation, and action detection preserve their parent SGD 1x recipes. Exact effective batches and schedule details are defined per recipe in the JSON file.

Augmentation, split, task head, loss, and evaluation metric remain unchanged from `EXTEND_LINEAR_v1`.

## Execution and reporting

One PBS job is one model x one dataset x `EXTEND_ATTENTIVE_v1` x seed 0. Jobs may be submitted whenever resources are available; no LP/AP/FT chaining is allowed. Skip valid results and healthy owners, resume incomplete checkpoints under the same logical ID, and isolate failures.

Use the final scheduled epoch. Report the reader/head parameter counts, trainable percentage, frozen-backbone check, feature shape, effective batch/LR, and the single seed-0 metric. Do not report mean +/- standard deviation.

No model-specific query count, width, layer choice, resolution, or LR search is permitted.
