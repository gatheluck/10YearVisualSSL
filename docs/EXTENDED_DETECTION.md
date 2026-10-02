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

## Verification and remaining work

Behavioral tests cover image/mask geometry, labels, original-ground-truth scoring,
empty/invalid predictions, LVIS federated semantics, output delivery and overwrite
protection. Reduced real encoders exercise 20 box/mask LP/AP loss/update routes.
Private comparisons cover captured detector initialization, losses, gradients and
updates, plus input/evaluator comparisons. Mutation and regression outcomes are
recorded in the PR; component tests do not certify released-weight CUDA behavior.

**A complete Extended detection training CLI is not added in this change.** The
12-epoch optimizer/accumulation/distributed/continuation path, source/catalog
geometry reconciliation, complete official split membership and table-to-run
mapping remain to be integrated and verified. In particular, do not substitute
the Basic5 trainer's labels, pyramid or recipe for these Extended components.
The current rerun catalog's `runnable` entries are not evidence of completed
training. Full-data and released-weight reproduction remain unverified.
