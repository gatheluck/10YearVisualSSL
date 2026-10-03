# Extended detection and instance-segmentation components

See [scope and terminology](SUBMISSION_SCOPE.md) and the
[final-paper coverage ledger](FINAL_PAPER_COVERAGE.md). This package now exposes
COCO 2017 box detection and LIVECell/LVIS v1 instance-segmentation inputs,
five-provider LP/AP detector compositions, and an executable prediction evaluator.
These are ports of existing experimental components, not new measured paper results.

## Inputs and model construction

`downstream.extended_detection.ExtendedDetectionData(root, annotation, train=...,
mask=...)` reads an explicit COCO-format annotation file with official category IDs.
It maps sorted sparse category IDs to contiguous training labels starting at one;
background is zero. `num_classes` includes background. LVIS images without
`file_name` resolve the `train2017`/`val2017` portion of `coco_url` beneath the
provided image root; no URL is downloaded. Escaping paths, image symlinks,
duplicate IDs, missing images, unknown categories and inconsistent image sizes
are rejected.

The captured input profile limits the image's longest side to 512, using bicubic
image and nearest-neighbor mask resizing. It keeps at most 64 eligible instances,
ordered by annotation area; training shuffles before the stable area sort and
uses one joint horizontal flip with probability 0.5. Crowd and boxes with width
or height at most one are omitted from training targets. Original annotation
areas are retained as in the reference. **Scoring always uses the untouched
original annotations**, including instances absent from these training targets.
Malformed masks fail instead of the original all-zero fallback. The official
pycocotools decoder is required; there is no approximate polygon fallback.

`build_extended_detector(body, num_classes, dataset=..., reader_profile=...,
geometry_profile=...)` accepts a frozen provider from
`downstream.spatial_backbones.build_frozen_backbone`. The verified providers are
DINOv3, RAEv2 K7, SigLIP2-G, V-JEPA2.1 and VGGT-Omega. It attaches the captured
transposed-convolution FPN and Faster R-CNN or Mask R-CNN. This differs from some
Basic5 provider pyramids, which remain unchanged. Raw RGB detector inputs are
converted to the shared provider normalization contract exactly once. Native
spatial grids remain spatial; no global pooling or still-image frame duplication
is introduced.

For LP, use `reader_profile=None`. For AP, the first four providers use
`captured_single_block_v1`; Omega uses `captured_cross_self_v1`. The encoder remains
frozen and in evaluation mode when the detector trains. RPN, ROI/mask heads, FPN
and the selected adapter are trainable.

Geometry must be selected explicitly: `captured_224_256` or
`captured_800_1333`. The current original builder selects 224/256 for hidden
widths at least 2048 and 800/1333 otherwise; the Omega AP builder explicitly uses
224/256. The supplied catalog specifies 800/1333. **This discrepancy is not
resolved by these components.** A chosen profile records an executable source
condition, not proof that it produced a paper value. The factory is a low-level
component and cannot authenticate a caller's provider/profile/run combination.

The detector inference limits are 100 for COCO, 300 for LVIS and 3000 for
LIVECell. LIVECell RPN inference proposal limits are also 3000. These limits are
separate from the official scoring limits below.

## Executable evaluation

The shared [downstream environment](DOWNSTREAM.md) includes pycocotools and
`lvis==0.5.3`; its hashed Linux lock is used in CI. LVIS imports its OpenCV
visualizer at package initialization; minimal Debian/Ubuntu images also need
`libgl1`, `libglib2.0-0` and `libxcb1`. A Python-only install in such an image
is insufficient even when only evaluation is requested. The LVIS API's removed
`np.float` alias is adapted in a private copy of the accumulator's globals.
Neither NumPy nor the installed evaluator module is modified. Empty prediction
sets are valid and score zero where ground truth exists; empty scored ground
truth is an error.

Prepare a `torch.save` archive containing a list of `(image_id, (height, width),
prediction)` records. Predictions have `boxes` as float `[N,4]` XYXY tensors,
`labels` as int64 `[N]` contiguous labels from the dataset mapping, and `scores`
as finite `[N]` probabilities. Instance predictions additionally have finite
probability `masks` of shape `[N,1,height,width]`. Coordinates and masks refer to
the specified input size. Every annotation image must appear exactly once,
including images with zero detections. Only load archives from trusted sources;
the CLI uses `torch.load(..., weights_only=True)`.

Adapt the [evaluation configuration](examples/extended_detection_evaluation.json),
then run:

```sh
python -m downstream.extended_detection --config resolved.json --out new-evaluation
```

The evaluator restores original-resolution boxes/masks, reverses category-label
mapping and scores original annotation IDs. Mask probabilities are resized with
bilinear interpolation and thresholded strictly above 0.5. It uses:

| Dataset | Official evaluator behavior |
| --- | --- |
| COCO 2017 | COCOeval bbox AP, categories separate, maxDets 100 |
| LIVECell | COCOeval bbox/mask AP, `useCats=0`, maxDets 2000 |
| LVIS v1 | LVISEval bbox/mask AP, maxDets 300, negative and not-exhaustive category metadata required |

Reported AP, AP50 and AP75 are percentages on area `all`. Dataset-specific metric
names prevent LIVECell class-agnostic AP from being confused with ordinary COCO
AP. HICO-DET, Open Images and AVA are refused rather than assigned these metrics.

Outputs are `results.json`, `metrics.json` and a hash-checked `run_manifest.json`.
Results include annotation and prediction hashes. Existing output directories
are refused. A failed evaluation produces a failed manifest, not an invented
score. `record_value` and `canonical_eligible` are always false: hashes alone do
not prove official split identity, checkpoint identity or historical provenance.

## Training, distributed execution and continuation

The training route is `python -m downstream.extended_detection --config resolved.json --out new-training-run`.
Start from the [training configuration](examples/extended_detection_training.json).
It trains frozen LP or attentive AP on COCO, LIVECell or LVIS, evaluates the full
validation population and writes contract artifacts. FT is not included.

Provide a common image root and separate explicit training/validation annotation
files. Category IDs and names must agree across splits; foreground counts must
match the supplied registry (80, 8 and 1203 respectively). Image IDs and resolved
paths must not overlap. LVIS federated metadata is required before constructing
the model. These checks do not authenticate official membership or detect
identical image bytes stored under different names. Keep input bytes immutable
throughout training and continuation.

The supplied LP/AP recipes use SGD with momentum 0.9, weight decay 0.0001 and
base learning rate 0.02 at effective batch 16. Bias/normalization/position
exemptions follow the captured parameter-group rule. Effective batch is physical
batch times accumulation times world size. There is no automatic OOM-driven
batch change. The captured detector uses FP32 training; BF16 is refused here,
even though other Extended tasks accept it.

The schedule warms up over 500 **optimizer updates**, from factor 0.001, then
uses factors 0.1 and 0.01 starting at zero-based epochs 8 and 11. Warmup takes
precedence even when a small fixture reaches those epochs before update 500.
This is not the cosine schedule used by other Extended tasks. LP discards an
incomplete accumulation group. AP flushes a short group (still divided by the
full accumulation count) for DINOv3 and RAEv2, but discards it for SigLIP2-G,
V-JEPA2.1 and Omega. AP clips all trainable gradients to norm 1. The configured
tail policy must match the provider. The frozen encoder stays in evaluation mode.

Use the same entry point under `torchrun --standalone --nproc-per-node=2 -m downstream.extended_detection --config resolved.json --out new-training-run`
for two processes. CPU uses Gloo and CUDA uses NCCL. Rank zero alone evaluates
all validation images and publishes artifacts; training uses a distributed
sampler. All ranks must agree on configuration and membership.

`resume.pt` saves completed epochs, optimizer/scheduler state and per-rank RNGs.
Add `"resume": "/runs/previous/resume.pt"` and increase `probe.epochs` within the
12-epoch horizon, using a different output directory. The shared
[continuation contract](EXTENDED_EXECUTION.md#epoch-checkpoints-and-continuation)
requires unchanged configuration, world size, data and encoder identity.
The checkpoint and `probe.pt` omit frozen encoder weights, while retaining the
trainable FPN, reader and detector heads. Native historical checkpoints are not
interchangeable with this format. Final metrics describe the requested last
epoch, including shorter diagnostic runs; no best-epoch selection is introduced.
`canonical_eligible` and `record_value` stay false.

## Verification and remaining work

Behavioral tests cover image/mask geometry, labels, original-ground-truth scoring,
empty/invalid predictions, LVIS federated semantics, output delivery and overwrite
protection. Reduced real encoders exercise 20 box/mask LP/AP loss/update routes.
Private comparisons cover captured detector initialization, losses, gradients and
updates, plus input/evaluator comparisons. Mutation and regression outcomes are
recorded in the PR; component tests do not certify released-weight CUDA behavior.

The October 3 integration supersedes the earlier training-CLI gap: it now
connects the components to 12-epoch LP/AP recipes, accumulation, distributed
execution, epoch continuation and final evaluation. Reduced actual providers
exercise 20 training routes; CPU two-rank tests verify uninterrupted/resumed
parameter equality. The CLI example is exercised with a local reduced encoder.
Source comparisons cover warmup/milestone factors and 12-epoch reduced updates.

Remaining gaps are source/catalog geometry reconciliation, FT integration,
CUDA/NCCL and released-weight validation, authentic full-data runs, complete
split authentication and table-to-run mapping. The inspected catalog labels
23 LP/AP entries for these datasets `runnable`, not completed. No score is
certified by these component tests. Other Extended task families remain separate.
