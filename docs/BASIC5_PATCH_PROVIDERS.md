# Trunk and final-merger Basic5 components

Status: 2026-09-26. `sam3_trunk` and `cosmos3_super_vm` add 14 component
execution paths: ImageNet frozen LP, and frozen/FT on ADE20K, NYUv2 and SSv2.
Use the [downstream environment and complete task configurations](DOWNSTREAM.md)
with `capture_basic5_components`. These are implementation paths, not verified
released-weight scores or complete Basic5 reproduction.

Replace the `backbone` in the complete ImageNet configuration with one of:

```json
[
  {"kind": "sam3_trunk", "arch": "released", "encoder": "/path/to/sam3.pt", "img_size": 224, "patch_size": 14},
  {"kind": "cosmos3_super_vm", "arch": "released", "encoder": "/path/to/vision_encoder", "img_size": 224, "patch_size": 16}
]
```

For other supported tasks select `adaptation: frozen` with
`optimizer_profile: basic5_frozen_v1`, or `adaptation: finetune` with
`optimizer_profile: basic5_finetune_v1`. Existing task batch/schedule constraints
apply. `probe.image_size` controls task input geometry. No network fetch or
random fallback is permitted. `arch: fixture` explicitly permits tiny locally
saved models for tests; it never represents a released experiment.

SAM3 accepts the official trunk file through the existing converter, or a local
vision-only `Sam3ViTModel.save_pretrained` directory. In the latter case,
`img_size` must match its configuration. It uses ImageNet normalization,
bottom/right zero padding to patch 14, dynamic global axial RoPE, and the mean
of all final patch tokens (1024 channels). It instantiates no task or text tower.
The existing official converter's assumptions, including dropped patch-projection
bias, remain unchanged; this work does not independently certify conversion of
the released checkpoint.

Cosmos3 requires a complete local vision-only `Qwen3VLVisionModel` directory.
Its Basic5 readout is the **final merger**, 5120 channels, with official
merge-window patch packing and temporal patch repetition. Task pixels arrive
ImageNet-normalized; the wrapper converts once to 0.5/0.5 normalization and pads
to patch-times-merge (32 pixels) with normalized zero. Spatial maps use the
merged grid. This is distinct from the existing method adapter's 1152-channel
patch-token readout and packing: existing extraction behavior/artifacts are
unchanged and must not be relabeled as final-merger features.

Both families normalize only the frozen image LP feature, use unnormalized
per-frame features before temporal averaging, and initialize classification
heads with normal standard deviation .01. FT uses the captured block-layer
decay mapping and last-block head endpoint. Cosmos3's final merger belongs to
that endpoint; DeepStack parameter groups follow their configured tap layers,
but DeepStack outputs are not used as features.

## Verification and limits

Tests exercise real tiny HF models, local checkpoint loading, non-square padded
grids, outputs, input gradients and three parameter updates, parameter ownership,
and all 14 task entrypoints with artifact-contract checks. CI explicitly imports
both model classes and runs these tests with downstream dependencies. Private
source comparisons and validation outcomes are recorded in the task PR.

- Results stay `canonical_eligible: false`; fixture scores are not paper values.
- COCO is rejected: SAM3 patch-14 and Cosmos3 merged-stride-32 maps do not meet
  the current shared stride-16 pyramid assumptions, and detection normalization
  requires separate reconciliation.
- AP and ImageNet FT remain rejected. This does not resolve the existing reader
  or augmentation discrepancies across sources.
- Distributed/BF16 training, released-weight execution, full-data metrics and
  workbook score agreement remain unverified. Local shape validation does not
  authenticate checkpoint identity; verify immutable checkpoint hashes before
  an experiment.
- Existing providers and feature-extraction artifacts retain their interfaces.
