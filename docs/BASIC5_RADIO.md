# C-RADIOv4-H downstream components

See [scope and terminology](SUBMISSION_SCOPE.md) for the distinction between
original experiments, portable components and reproduced results.

Use the [downstream environment](DOWNSTREAM.md). This provider loads a complete
local `nvidia/C-RADIOv4-H` Hugging Face snapshot, including its official Python
modules, configuration and `model.safetensors`. No model code or weights are
bundled here. Loading executes that local custom code; use a reviewed official
snapshot. It requests local files only and rejects reported incomplete weights.

Overlay an existing downstream task configuration with:

```json
{
  "profile": "capture_basic5_components",
  "adaptation": "attentive",
  "reader_profile": "captured_cross_self_v1",
  "optimizer_profile": "basic5_frozen_v1",
  "backbone": {
    "kind": "cradiov4_h",
    "arch": "released",
    "encoder": "/path/to/local/official/snapshot",
    "img_size": 224,
    "patch_size": 16
  }
}
```

LP selects `frozen` and omits `reader_profile`. FT selects `finetune` and
`basic5_finetune_v1`, without `reader_profile`. COCO also selects
`detector_profile: captured_native_detection_v1`. ImageNet LP/AP and
ADE20K/COCO/NYUv2/SSv2 LP/AP/FT are component routes. ImageNet FT now uses an
[explicit source profile](IMAGENET_FINETUNE.md); augmentation/run identity
remains unreconciled and local reduced fixtures do not verify released weights.

## Preserved experimental behavior

- The official summary is 2560-dimensional, selecting teacher CLS indices 0 and
  1. The separate 1280-dimensional spatial features exclude ten prefix tokens.
  The official model performs that selection; the wrapper never fabricates tokens.
- Task tensors have ImageNet normalization. The wrapper reverses it once before
  the internal official conditioner. Detection supplies raw RGB using identity
  detector normalization. Spatial dimensions are padded to patch multiples,
  then cropped to the original complete-patch grid, matching the source.
- ImageNet LP L2-normalizes summary features; FT composition leaves them raw.
  Video classification averages frame summaries without L2 normalization.
  Video AP uses ordered per-frame means of spatial tokens. Both retain the
  official video CPE context when available.
- AP explicitly selects the captured cross-plus-self attention reader; this
  remains distinct from the single-block protocol interpretation. Detection
  retains the learned four-scale pyramid and 91-output category-ID convention.
- FT trains the visual tower only. Stem parameters use layer 0, blocks use
  layers 1 through 32 and heads use endpoint 32. Bias, normalization and position/
  token parameters follow captured decay exemptions. Classifiers use normal
  initialization with standard deviation 0.01 and zero biases.

## Verification boundary

Interface fixtures test outputs, input/parameter gradients, three updates,
freezing, checkpoint/layout rejection and all 14 task routes. They are not
released weights and do not establish paper-score parity. `fixture` mode requires
explicit reduced dimensions and remains noncanonical. The captured and freshly
read experimental backbone source matched on 2026-09-28. Full-size released-weight
GPU execution, numerical precision parity and full-data measurements remain
unverified. See [coverage](FINAL_PAPER_COVERAGE.md) and
[result accounting](DOWNSTREAM_ACCOUNTING.md).

The real official custom-code loader was also exercised privately with a reduced
saved/reloaded official model. Unmodified experimental wrapper methods matched
spatial/video outputs, input/parameter gradients and three updates; maximum
observed spatial output difference was `2.4e-7`. CPE training randomness used
identical RNG states. This FP32 reduced-model check does not establish BF16 or
released-weight equivalence. Private source copies and test checkpoints are
excluded from Git.

## Explicit activation checkpointing

Fine-tuning supports `backbone.activation_checkpointing: true`; omission
preserves previous execution. See [activation checkpointing](ACTIVATION_CHECKPOINTING.md)
for the inspected mechanism, image/video verification and GPU/full-run limits.
