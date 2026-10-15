# Frozen evaluation of IDv2 scaling exports

Support and validation statements describe this portable package. See
[scope and terminology](SUBMISSION_SCOPE.md) for its distinction from the
original experimental implementations and reported results.

This integrates existing experimental evaluation code into the portable package.
It does not establish reproduced paper scores. The scope is the LINEAR evaluation
of ViT-B/16 and ViT-L/16 from the [scaling entries](IDV2_SCALING.md), including the
separate ImageNet-1k 300-epoch profile. AP and FT are deliberately refused: the
inspected evaluation campaign is LINEAR-only.

## Loading and readouts

Use backbone kind `idv2_scaling`, `img_size: 224`, `patch_size: 16`, and `arch`
equal to the export's `vit_base_patch16_224` or `vit_large_patch16_224` preset.
`encoder` must name a local version-1 scaling `encoder.pt` (or retained epoch
export). There is no download, random fallback or dimension override. Missing or
extra encoder weights and mismatched identity fail. Generic ablation exports and
native historical training checkpoints are not accepted directly. Use the
explicit conversion below for verified epoch-boundary checkpoints. The file's preset
identity is not evidence of its historical training epoch or paper-row provenance.

Training and evaluation share native timm preset construction: final LayerNorm
epsilon is 1e-6, with dynamic bicubic/antialiased positional interpolation.
Task inputs are already ImageNet-normalized. The provider does not normalize
them twice, resize images, pad incompatible inputs, use the projector, or use
the memory bank. Both spatial dimensions must be positive multiples of 16.
The encoder stays in eval mode with gradients disabled, even under `train()`.

| Task | Frozen representation and connection |
| --- | --- |
| ImageNet | Final CLS, L2-normalized; `downstream.imagenet`, explicit `preprocessing_profile: captured_provider_v1` for bicubic geometry |
| SSv2 | Final CLS per frame, then temporal mean without L2; `downstream.ssv2`, explicit `video_recipe: captured_provider_video_v1` for Python temporal sampling, shared bicubic crop, no horizontal flip |
| ADE20K | Final patch tokens without CLS, reshaped to the actual rectangular grid; `downstream.ade20k` |
| NYUv2 | Same spatial grid; `downstream.nyuv2`; depth-loss version remains a separate unresolved question below |
| COCO | Same grid after detector-owned ImageNet normalization, learned transposed-convolution pyramid and 91-class Faster R-CNN; `downstream.coco` with `detector_profile: captured_native_detection_v1` |

All routes require `profile: capture_basic5_components`, `adaptation: frozen`
and an explicit task configuration. Classification heads retain the inspected
normal initialization (standard deviation 0.01) and zero bias. Image/video
execution exposes the existing single-process accumulation/epoch-continuation
machinery, discarding incomplete accumulation groups. Distributed execution is
refused for this provider until independently verified. Dense distributed
execution is not added. CUDA/BF16 behavior is not established by CPU tests.

## Invocation

Copy the [validated ImageNet example](examples/idv2_scaling_linear.json), replace
the dataset and export paths, and run from the repository root:

`python -m downstream.imagenet --config /path/to/idv2-linear.json --out /path/to/new-evaluation`

The example describes a full 100-epoch linear-probe configuration; it is not a
record of an executed run. Use the environment in `downstream/requirements.txt`
and its platform lock. For a smoke, explicitly reduce sample counts and epochs.
All results remain `canonical_eligible: false`; a smoke additionally remains
`record_value: false`. For the other task schemas, see [downstream tasks](DOWNSTREAM.md),
[native BasicFive paths](BASIC5_NATIVE_PATHS.md), and the existing task examples.
Only substitute the backbone and explicit profiles above; do not infer completed
experiments from these connections or silently substitute the generic ViT kind.

## Evidence and unresolved scope

### Importing a native epoch-boundary checkpoint

Run `python methods/10_inst_disc/models/scaling_encoder.py --source /path/to/native.pth --sha256 SOURCE_SHA256 --arch vit_large_patch16_224 --completed-epoch 100 --state-key state_dict --prefix vit. --out /path/to/new-export`.

The source hash must be independently recorded before conversion. Choose the
architecture, mapping and epoch from inspected checkpoint metadata, not its
filename. The converter verifies the hash before and after reading, requires
an exact integer `completed_epoch` and `step_in_epoch: 0`, and strictly checks
encoder keys, shapes, FP32 dtypes and finite values against the native preset.
It supports explicit `state_dict` or `model` mappings and `vit.` or
`encoder.vit.` prefixes, optionally preceded by `module.`. No automatic prefix,
architecture or epoch correction is performed. Unknown pickle classes fail;
the restricted loader permits the native NumPy RNG array metadata but never
falls back to unrestricted loading.

The new directory contains `encoder.pt` for the five evaluation routes and
`export.json` with hashes, mapping and literal epoch metadata. Original paths,
configuration, optimizer, RNG, projector and memory bank are not exported.
Existing output directories are refused. A failed filesystem write can leave
an incomplete directory; do not use it as a completed export. This is encoder
conversion, not native training continuation or certification of a paper row.

Read-only archive metadata inspection on 2026-10-13 found the three ImageNet-1k
retained checkpoints at 100/200/300 consistent with the boundary requirement.
The three inspected ImageNet-21k retained checkpoints have nonzero
`step_in_epoch`; their historical epoch interpretation remains pending and
this converter deliberately refuses them.

### Full-weight GPU validation (2026-10-15)

All three ImageNet-1k ViT-L retained checkpoints (100/200/300) now pass physical
conversion with 294 strict encoder tensors each and unchanged source hashes.
On an H200 with Torch 2.5.1, 54 original-wrapper/portable comparisons are exact:
three checkpoints, FP32/BF16, 224-square and both 224-by-320 orientations, and
image CLS/L2, spatial patch grid and temporal-mean frame CLS readouts. Original
LayerNorm/position tensors remain unchanged. Four refreshed campaign sources
match Capture and current originals. Large exports were tested on transient
node-local storage; this is not persistent artifact delivery or paper-score reproduction.

The initial import exposed Torch 2.5's callable-only safe-global API rejecting
tuple aliases. The converter now provides narrow callable aliases for both
NumPy pickle namespaces while retaining restricted loading, hash/epoch checks
and rejection of unapproved objects. Four importer tests pass in that runtime.
The broader detector test requires newer Python APIs and failed in the existing
Python 3.10 GPU environment; it passes in the local Python 3.12 regression suite.
Do not infer that the complete downstream CLI supports Python 3.10 from this
isolated encoder experiment. Use the documented task environment for full runs.

### Earlier evaluation evidence

Read-only refresh on 2026-10-12 found all five inspected campaign Python files
and the shared data/protocol files identical to the capture (seven of nine files).
Shared modules and training code had a later depth-loss/diagnostic change:
the log-residual mean-square coefficient changed from 1.0 to 0.5, and evaluation
added clamp-fraction diagnostics. Neither version is silently selected as the
paper's historical run. This change leaves existing depth loss/metrics untouched.
Original paths, source copies and detailed comparison records stay outside Git.

Reduced CPU comparisons against the captured wrapper and heads cover normalized
inputs, native and rectangular spatial features, image CLS/L2, frame mean,
pyramid outputs, gradients and two optimizer updates. Tests first failed on the
missing provider, missing explicit video recipe, missing CI execution and missing
documented example. Five real task entrypoints train/evaluate small fixtures and
verify output manifests; this is integration evidence, not full-size training.

Import of historical non-boundary checkpoints, epoch/result attribution,
full-weight validation of other scaling artifacts, distributed evaluation,
full-size downstream training/continuation and full-data paper-score reproduction
remain unfinished. The owner-held FSDP discrepancy is unrelated and unchanged.
Record mutation, local whole-suite, Linux and CI results separately in the PR.
