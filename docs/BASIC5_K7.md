# RAEv2 K7 downstream components

This is the BasicFive tokenizer profile used in final Tables 36–38, not the
historical last-seven-layer sum or the final-layer-only reconstruction proposal.
It uses the pinned DINOv3 author encoder; no decoder or diffusion transformer is
constructed. Experimental sources were compared read-only with the captured
snapshot and current sources on 2026-09-27. Private paths and identities are not
distributed here.

The existing DINOv3 pin matches the experiment's revision. Of 171 inspected
Python files, only the checkpoint-download helper differed; that helper is not
called by either local-checkpoint loading path used here. Direct comparisons
with the unchanged experimental wrapper matched image/video outputs exactly
and checked gradients and all encoder parameters after three SGD updates.

## Configuration

Overlay an existing [downstream task configuration](DOWNSTREAM.md) with:

```json
{
  "profile": "capture_basic5_components",
  "adaptation": "attentive",
  "reader_profile": "captured_single_block_v1",
  "optimizer_profile": "basic5_frozen_v1",
  "backbone": {
    "kind": "raev2_k7",
    "arch": "released",
    "encoder": "/path/to/dinov3_vitl16_pretrain.pth",
    "img_size": 224,
    "patch_size": 16
  }
}
```

Use the optimizer profile supported by the base configuration. COCO additionally
requires `detector_profile: captured_native_detection_v1`. Frozen LP and FT omit
`reader_profile`; FT selects `basic5_finetune_v1`. An explicit `fixture` architecture
accepts reduced width/head count but retains all 24 layers; it is not a released
model. All checkpoints are local, strictly loaded, and never downloaded implicitly.

The component exposes LP/AP on ImageNet and LP/AP/FT on ADE20K, COCO, NYUv2 and
SSv2. ImageNet FT execution remains refused because its full augmentation recipe
has not been reconciled. Results remain `canonical_eligible: false` and
`record_value: false`; successful component execution is not paper-score parity.

## Readout and training boundaries

- Select zero-based layers 11, 13, 15, 17, 19, 21 and 23, with special tokens
  removed and a non-affine final LayerNorm. Average these seven outputs and add
  the spatial mean of the last selected layer to every patch.
- Inputs from task datasets already have ImageNet normalization. Pad in that
  normalized space to multiples of 16; do not normalize them again. Detection
  receives raw RGB from its identity transform and normalizes exactly once.
- Image LP/FT uses L2-normalized patch means. Video LP/FT first normalizes each
  frame's patch mean, then averages frames. Video AP receives ordered,
  unnormalized per-frame patch means. No temporal position encoding is invented.
- AP selects the single cross-attention reader and single residual spatial
  block; COCO uses the bilinear four-scale pyramid. The historical/common
  cross-plus-self-attention profile is a different implementation.
- Frozen execution stays in eval mode and disables encoder gradients. FT
  unfreezes the whole visual encoder while keeping RoPE augmentation disabled;
  layer decay spans stem 0, encoder layers 1–24 and task output 25.

Tests exercise the pinned author encoder at reduced width, checkpoint rejection,
padding, image/video readouts, gradients and optimizer updates, plus all 14 task
entrypoint routes on small datasets. These checks do not validate released-weight
GPU execution, distributed BF16 training, full datasets, or paper scores. See the
[full-paper coverage audit](FINAL_PAPER_COVERAGE.md) for remaining work.

See [scope and terminology](SUBMISSION_SCOPE.md) for experimental versus portable
implementation status and the separate evidence required for reproduced results.
