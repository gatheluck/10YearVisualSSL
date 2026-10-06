# VGGT-Omega downstream components

The provider uses the visual aggregator of the selected 512 release. The author
submodule is pinned to `39a0cb8af88554f15ddcb5354cd52bde588fa014`, matching the
experimental checkout's contents inspected on 2026-09-27. Camera, depth and text
heads are never instantiated. This is not the pre-aggregator patch embedding or
a camera-head representation.

## Configuration

Overlay an existing [downstream task configuration](DOWNSTREAM.md) with:

```json
{
  "profile": "capture_basic5_components",
  "adaptation": "attentive",
  "reader_profile": "captured_cross_self_v1",
  "optimizer_profile": "basic5_frozen_v1",
  "backbone": {
    "kind": "vggt_omega",
    "arch": "released",
    "encoder": "/path/to/vggt_omega_1b_512.pt",
    "img_size": 224,
    "patch_size": 16
  }
}
```

The released file must match the SHA-256 recorded for the selected visual
checkpoint in the inspected experiment identity. A different release, including
the 256 text-alignment checkpoint, is rejected before constructing the model.
Only explicit `fixture` mode permits reduced aggregator dimensions. Released
weights and full GPU execution have not been validated by the local fixture tests.

COCO additionally requires `detector_profile: captured_native_detection_v1`.
Frozen LP and FT omit `reader_profile`; FT selects `basic5_finetune_v1`.
LP/AP/FT execute on all five tasks. ImageNet FT requires its
[explicit source profile](IMAGENET_FINETUNE.md), retaining Mixup-only and the
legacy unit-interval heuristic without declaring the recipe reconciled.
All routes remain noncanonical, nonrecordable components.

## Source-specific behavior

- One image uses exactly one frame. Video uses the ordered frames jointly,
  without independent frame encoding or artificial duplication. The last cached
  aggregator layer concatenates frame and inter-frame features; the camera token
  and sixteen register tokens are removed. Video AP consumes all frame/patch
  tokens, unlike [K7's per-frame means](BASIC5_K7.md).
- Task tensors have ImageNet normalization, which the wrapper reverses before
  the author's internal conditioner. Detection's identity transform supplies raw
  RGB directly. Spatial inputs must be multiples of 16; no grid is fabricated.
- LP L2-normalizes the final global mean, including video. FT leaves that mean
  unnormalized. AP uses the explicitly selected cross-plus-self reader; this
  remains distinct from the single-block protocol interpretation.
- COCO uses the learned four-scale pyramid, an 81-output detector and sorted
  contiguous foreground labels 1–80. Evaluation maps those labels back to COCO
  category IDs. The recorded label map is part of the run result. Other providers
  retain their existing 91-output/category-ID convention. Train/validation label
  maps must agree; unknown predicted contiguous labels fail visibly.
- FT layer decay assigns the patch-embedding tower to stem 0, pairs frame and
  inter-frame block indices at layers 1–24, and uses endpoint 24 for the head.

Small pinned-author models exercise all 14 task/adaptation routes. Direct CPU
comparisons with unchanged captured/current wrapper methods checked image/video
outputs, input/parameter gradients and three SGD updates. The largest observed
output difference was below `6e-7`; these are synthetic comparisons, not scores.
Dataset interpolation/augmentation, full schedule/distributed execution and
result-to-paper identity still require run-level verification. See the
[final-paper audit](FINAL_PAPER_COVERAGE.md).

See [scope and terminology](SUBMISSION_SCOPE.md) for experimental versus portable
implementation status and the separate evidence required for reproduced results.
