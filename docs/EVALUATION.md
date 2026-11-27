# Evaluation settings in the paper

Section 3 defines three settings. Choose a setting by the question the figure
asks before selecting a checkpoint, readout or training command.

| Setting | Where to find the comparison | Starting point |
| --- | --- | --- |
| ASIS | Section 4, Figure 2: transfer from released or reproduced models using family-specific interfaces and protocols. | [Historical model-family specifications](initial_protocols/README.md) |
| CTRL | Section 5, Figure 4: compare 32 VSSL methods trained from scratch on unlabeled ImageNet-1K with a ViT-B/16 backbone for 300 epochs. | [Controlled training recipes](STEP2_VIT_PORTING.md) |
| Unified | Section 6, Figure 3: compare five checkpoints under shared downstream tasks, splits and metrics; feature interfaces remain model-specific. Figure 5 extends the evaluation to 40 settings. | [Evaluation and adaptation guide](CVPR_ADAPTATION.md) |

The Basic Five measure image classification on ImageNet-1K, bounding-box detection
on COCO, semantic segmentation on ADE20K, depth estimation on NYUv2 and video
classification on SSv2. The [Figure 6 frontier-system comparison](FRONTIER_EVALUATION.md)
uses adapted tasks on matched subsets, including category presence for COCO.
Those scores belong to a separate comparison.

## Choose an implementation

- For Figure 1 ranks, Figure 7 representation analysis, or a specific reported
  result, start with the [figure-by-figure guide](PAPER_REPRODUCTION_GUIDE.md).
- For checkpoint selection, see the [feature extraction guide](STEP1_WEIGHTS.md).
  A stored encoder can be used in several analyses. Its filename alone does
  not identify a paper result.
- For per-method training or ImageNet probing, see the [method index](../OPERATING_GUIDE.md#methods).
  `pretrain` and `linear_eval` are adapter operations. A single adapter smoke
  test establishes that the selected path runs on its test inputs.
- For downstream training and evaluation, use the [current protocol index](../README.md#evaluation-protocols)
  and check the [execution audit](BASIC5_V1_AUDIT.md) for actual run settings
  and reported metrics.

For older configuration names, see [manuscript terminology](PAPER_TERMINOLOGY.md).

The [August 2026 implementation record](history/2026-08.md) preserves the
checkpoint registries, early porting results and implementation limits measured then.
See [package scope](SUBMISSION_SCOPE.md) for implementation and validation limits.
