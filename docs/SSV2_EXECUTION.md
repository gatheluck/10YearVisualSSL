# BasicFive video recipes and portable execution

This explicit SSv2 component supports the eight inspected BasicFive encoder
families across linear probing (LP), attentive probing (AP) and fine-tuning (FT).
It adds provider-owned temporal sampling and clip augmentation, gradient
accumulation, CUDA BF16 selection, replicated DDP and portable epoch continuation.
See [package scope](SUBMISSION_SCOPE.md): component support is distinct from
reproducing the paper's results or identifying a historical table cell.

## Selecting the component

Use the [configuration example](examples/ssv2_execution.json) with
`video_recipe: captured_provider_video_v1`, `profile: capture_basic5_components`,
explicit `adaptation`, and the matching BasicFive optimizer profile. LP/AP use
`basic5_frozen_v1`; FT uses `basic5_finetune_v1`. AP additionally needs its
provider's reader profile when required. Existing configurations without
`video_recipe` retain their previous behavior.

`execution` accepts exactly `profile: captured_video_v1`, positive integer
`accumulation_steps`, `precision: fp32` or `bf16`, and `tail_policy: discard`.
Without this block the explicit recipe uses accumulation one and FP32. Resume
requires an explicit execution block. BF16 requires native CUDA BF16 support;
there is no automatic precision fallback.

`probe.batch_size` is physical batch per rank. Effective batch is physical batch
times world size times accumulation. The reference schedule requires effective
batch 256 for all three adaptations, with 50 epochs, five warmup epochs and cosine
decay (LP to zero, AP/FT to 1e-6). Its horizon is 50 times the full per-rank loader
length, while its clock advances only on optimizer updates. A shorter run keeps
that horizon. Without the reference schedule, learning rate remains constant.
Only complete accumulation groups update parameters; the trailing incomplete
group is discarded. AP/FT clip gradients after accumulation. An epoch with no
update fails. `max_steps_per_epoch` must be zero; epochs must be 1 through 50.

Released profiles require 16 frames at 224 pixels. Small local fixtures test
mechanics without claiming released-weight validation. Dataset layout remains
`videos/*.webm` and `labels/{train,validation,labels}.json`; the existing nested
SSv2 layout is also accepted. Missing or undecodable videos fail explicitly.

## Source-specific behavior

All profiles sample one frame per rounded temporal segment and share a spatial
crop across the clip. Evaluation uses segment centers, resize-short-side 256 and
center crop. No horizontal flip is applied. LP/AP use no strong augmentation.

| Family | Temporal RNG / interpolation | FT clip augmentation | FT batch mixing |
| --- | --- | --- | --- |
| CLIP, SigLIP2, C-RADIO | Python / bicubic | Shared RandAugment and random-valued raw-domain erasing | Python/NumPy MixUp or CutMix |
| Cosmos | Python / bicubic | Shared RandAugment and raw-domain erasing with value 0.5 | Python/NumPy MixUp or CutMix |
| DINOv3 | Torch / bilinear | Plain crop | Torch MixUp or CutMix |
| RAEv2 K7 | Torch / bilinear | Shared RandAugment and raw-domain zero erasing | Torch MixUp or CutMix |
| V-JEPA2.1 | Torch / bilinear | Shared RandAugment; separate erasing region per normalized frame | Torch MixUp or CutMix |
| VGGT-Omega | Python / bicubic | Shared RandAugment/erasing, then whole-clip range repair | MixUp only |

Batch mixing uses one permutation and spatial rectangle across a video's frames.
It returns new tensors without modifying caller inputs. Inspected FT functions
produce unsmoothed targets, whereas the supplied protocol specifies smoothing
0.1 and a more uniform augmentation recipe. This profile preserves the inspected
implementation, reports the conflict, and does not resolve historical attribution.

Identical decoded fixture frames were compared with both the captured and freshly
read source: 384 clip cases and 192 mixing cases matched outputs and
Python/NumPy/Torch RNG exactly. This does not compare video decoders: the portable
runner uses PyAV and fails closed; some original loaders used decord and black
fallback frames. Codec equality and historical fallback populations are unverified.

## Distributed execution and continuation

The [shared continuation validation contract](CONTINUATION_INTEGRITY.md) applies
to optimizer recipes, state tensors and RNG records before live restoration.

For example:

`torchrun --standalone --nproc-per-node=2 -m downstream.ssv2 --config /data/two-rank-video.json --out /results/video-01`

Adjust accumulation when changing world size. Rank seed strides and the seven
supported replicated-DDP families are the same as the inspected
[ImageNet execution policy](IMAGENET_EXECUTION.md). Omega distributed execution
is refused because its source calls unwrapped forwards. Only rank zero evaluates
the complete validation population without sampler padding. Training retains
per-rank sampler padding and drop-last behavior. DDP is not native FSDP.

Every completed epoch atomically writes `resume.pt` (`ssv2_epoch_v1`). LP/AP save
the trained probe; FT also saves the encoder. Optimizer, scheduler, counters and
per-rank Python/NumPy/Torch/loader RNG are included. Continue with `resume` pointing
to that file, a larger total `probe.epochs`, and a new output directory. Keep all
other training settings, initial encoder, precision, world size and data unchanged.
Ordered membership, annotation contents and video bytes are hashed before training;
this incurs a full selected-population read. Mismatches fail without overwriting
prior outputs or source checkpoints. Native experimental checkpoints are not
accepted as portable continuation files.

`results.json` records recipe, membership, optimization, execution and continuation.
`canonical_eligible` and `record_value` remain false. CPU fixture tests cover all
24 model/adaptation routes and exact uninterrupted versus resumed LP/AP/FT states,
including two-rank Gloo. Released-weight CUDA/BF16, native FSDP, original resume
compatibility and full-data paper scores remain unverified. ImageNet and SSv2 share
one classification execution engine; preprocessing and task contracts remain distinct.
