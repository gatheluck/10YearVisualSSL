# Explicit experimental readers and native detection

Status: 2026-09-27. See the [final-paper coverage audit](FINAL_PAPER_COVERAGE.md)
and [terminology map](PAPER_TERMINOLOGY.md). These are ports of inspected
experimental components, not certification of the reported scores.
See [submission scope](SUBMISSION_SCOPE.md) for the distinction from original
experimental implementations and results.

The five existing vision providers now accept explicit AP reader profiles on
ImageNet, ADE20K, NYUv2 and SSv2, plus native COCO detection in frozen, attentive
and finetune modes. Tests execute **35 profile routes** (5 x (4 + 3)) using
small saved models and synthetic data. DINOv3 already had a different frozen/FT
COCO path; the count is not 35 entirely new task combinations. SAM3 is a
component integration, not one of the eight models in final Tables 37 and 38.

## Configuration

Use the [complete downstream task configurations](DOWNSTREAM.md) and its locked
environment. Merge one of these top-level overlays into a task configuration;
replace the local checkpoint placeholder. Keep the task's dataset, seed, device,
probe/detector, batch and schedule fields. These overlays are parsed through all
five real configuration validators in tests.

```json
[
  {"profile":"capture_basic5_components","adaptation":"attentive","optimizer_profile":"basic5_frozen_v1","reader_profile":"captured_cross_self_v1","backbone":{"kind":"clip_hf","arch":"released","encoder":"/path/to/local-snapshot","img_size":336,"patch_size":14}},
  {"profile":"capture_basic5_components","adaptation":"attentive","optimizer_profile":"basic5_frozen_v1","reader_profile":"captured_cross_self_v1","backbone":{"kind":"siglip2_g","arch":"released","encoder":"/path/to/local-snapshot","img_size":384,"patch_size":16}},
  {"profile":"capture_basic5_components","adaptation":"attentive","optimizer_profile":"basic5_frozen_v1","reader_profile":"captured_single_block_v1","backbone":{"kind":"dinov3_hf","arch":"released","encoder":"/path/to/local-snapshot","img_size":224,"patch_size":16}},
  {"profile":"capture_basic5_components","adaptation":"attentive","optimizer_profile":"basic5_frozen_v1","reader_profile":"captured_cross_self_v1","backbone":{"kind":"sam3_trunk","arch":"released","encoder":"/path/to/local-snapshot","img_size":224,"patch_size":14}},
  {"profile":"capture_basic5_components","adaptation":"attentive","optimizer_profile":"basic5_frozen_v1","reader_profile":"captured_cross_self_v1","backbone":{"kind":"cosmos3_super_vm","arch":"released","encoder":"/path/to/local-snapshot","img_size":224,"patch_size":16}}
]
```

For COCO additionally set `detector_profile: captured_native_detection_v1` and
four positive integer `detector.anchor_sizes`, normally `[32,64,128,256]`.
To select COCO frozen LP, omit `reader_profile` and set `adaptation: frozen`.
For COCO FT, omit `reader_profile`, set `adaptation: finetune` and
`optimizer_profile: basic5_finetune_v1`. Existing optimizer/scheduler constraints
still apply; a profile is not a complete experiment configuration. Do not use
fixture batch limits for paper experiments. `arch: released` checks architecture,
not checkpoint provenance; authenticate the complete local weights separately.

The chosen profiles are saved in `results.json` and the hashed resolved config.
Missing, mismatched or unused reader choices are rejected. Family AP detection
requires the native detector profile. Existing implicit/default paths keep their
earlier behavior. Results remain `canonical_eligible: false`, `record_value: false`.

## Scientific behavior retained from the source

| Profile | Image/video reader | Dense adapter |
| --- | --- | --- |
| `captured_cross_self_v1` | 32 queries, width 512; cross attention then a self-attention/MLP block | Width 256, source double residual, zero output projection |
| `captured_single_block_v1` | 32 queries, width 512; one cross-attention/MLP block | Width 256, single internal residual, zero output projection |

Image AP consumes final spatial tokens, without substituting a CLS/MAP pooler.
Image-provider video AP consumes ordered per-frame patch means. No temporal
position encoding is invented: these readers are permutation invariant despite
receiving ordered inputs. This is distinct from native video-encoder tokens.

Native detection keeps the actual feature grid. Four learned transposed-conv
scales are used for the common family; the register-token family retains its
bilinear pyramid. Patch-14 or merged grids are not resampled to pretend they
have stride 16. Faster R-CNN has 91 output classes, four anchor levels, RoI size
7, sampling ratio 2 and batch padding to 32. Family normalization is applied
once, preserving the source ordering relative to padding. The trunk family uses
score threshold 0 at evaluation; other families retain 0.05. LP freezes the
encoder and trains pyramid/head; AP additionally trains its adapter; FT permits
encoder gradients with the existing family-owned parameter groups.

## Evidence and boundaries

Unchanged experimental implementations were compared on identical reduced FP32
inputs: six reader/pyramid components matched all initial parameters, outputs,
input/parameter gradients and all parameters after three SGD updates. Private
source revisions and hashes remain outside Git. The public synthetic numeric
fixture retains sampled state statistics, output and gradient summaries for
regression; it contains neither learned checkpoint weights nor paper scores.
Tests also exercise normalization, a non-square patch-14 grid, optimization
scope, all 35 entrypoint artifact contracts and existing-provider regressions.

The manuscript's one-block AP description and the common experimental
cross-plus-self reader remain different. Explicit profiles preserve this
difference; they do not decide which run produced a table cell. Similarly, some
experimental depth evaluators median-align predictions, whereas final Appendix
E.4 specifies unaligned metric RMSE. The public metric is not silently changed.
ImageNet FT now has [explicit source profiles](IMAGENET_FINETUNE.md), while
historical recipe reconciliation, distributed/BF16 execution, released-weight GPU
feasibility, complete schedules/data recipes, repeated-run score correspondence
and full-data paper reproduction remain unverified or require further porting.
Component parity does not establish identical whole-run RNG consumption or
construction order. For example, common experimental dense runners construct
the head before the adapter, whereas the portable dense runner constructs the
adapter first. Their initialization distributions match, but identical top-level
seeds alone do not guarantee identical initial tensors across those compositions.
