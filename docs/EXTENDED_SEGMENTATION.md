# Extended semantic segmentation components

See [submission scope](SUBMISSION_SCOPE.md), [final-paper coverage](FINAL_PAPER_COVERAGE.md)
and [Extended image classification](EXTENDED_CLASSIFICATION.md). Original
experimental implementations exist; this page describes their portable component
integration, not reproduced manuscript scores.

## Execution

`python -m downstream.extended_segmentation --config /path/config.json --out /path/new-run`

<!-- segmentation-example -->
```json
{
  "task": "extended_semantic_segmentation",
  "profile": "capture_extended_components",
  "dataset": "bdd100k",
  "seed": 0,
  "device": "cuda",
  "data_root": "/path/data",
  "samples": "/path/samples.json",
  "transform_profile": "captured_fixed224_v1",
  "adaptation": "attentive",
  "reader_profile": "captured_single_block_v1",
  "backbone": {"kind": "siglip2_g", "arch": "released", "encoder": "/path/local-checkpoint", "img_size": 384, "patch_size": 16},
  "probe": {"epochs": 20, "batch_size": 8, "num_workers": 2}
}
```

For LP choose `adaptation: frozen` and `reader_profile: null`. Five providers
have explicit dense profiles: `dinov3_hf`, `raev2_k7`, `siglip2_g`, `vjepa2_1`
use `captured_single_block_v1`; `vggt_omega` uses `captured_cross_self_v1`.
Dense AP uses a residual spatial adapter of width 256, not a global query reader.
The final patch grid excludes prefix/register tokens. The encoder remains frozen
and in evaluation mode. A normal-0.01, zero-bias 1x1 convolution predicts class
logits, upsampled bilinearly with `align_corners=False`.

The accepted catalog entries have a fixed class count, `semantic_seg_20` and
primary metric `mIoU`: ADE20K (150), PASCAL VOC 2012 (21), Cityscapes (19),
Mapillary Vistas (66), BDD100K (19), and PASCAL Context (59). Cityscapes is a
catalog entry, not one of the final 45 measured rows. This adds a shared route
relevant to five measured rows, not six independently validated native ports.

SpaceNet's catalog metric is building IoU/F1 but the final table says mIoU.
SUN RGB-D is depth/RMSE in the supplied catalog but semantic mIoU in the final
table. Both are refused pending run/specification reconciliation. Entries whose
class counts must be inferred, instance segmentation and other task families are
also refused. No tracked original protocol is rewritten to hide these differences.

## Membership and label identity

Supply a JSON object containing exactly `schema_version: 1`, `classes` (ordered,
unique class names), `label_map`, `split_evidence`, `train`, and `validation`.
Each split is a nonempty list of `{"image":"relative.jpg","mask":"relative.png"}`.
Class count must match the selected catalog entry. `label_map` explicitly maps
canonical decimal native-ID strings to zero-based class IDs or ignore ID 255.
For example `{"0":255,"1":0,"2":1}` maps background to ignored and two native
labels to two classes; a real dataset needs its entire ontology.

The runner refuses absolute/escaping paths, missing files, duplicate or
overlapping image/mask paths, invalid mappings, unknown observed IDs, RGB masks,
and mismatched image/mask geometry. Palette PNG indices are labels; the palette
colors are never inverted into classes. Mapping precedes nearest-neighbor target
resize. A hash and split evidence are recorded, but do not authenticate official
membership. Keep local paths and sample lists outside Git.

Native membership generation uses
`python -m downstream.extended_segmentation_membership --config /path/membership.json --out /path/new-samples.json`.
The config has exactly `dataset` and `data_root`; CLI counts cannot be overridden.

| Dataset | Root layout and membership | Train/validation counts |
| --- | --- | --- |
| ADE20K | `images/{training,validation}` joined to `annotations/{training,validation}` by stem; native 0 ignored, 1-150 shifted | 20210/2000 |
| VOC 2012 | `ImageSets/Segmentation/{train,val}.txt`, `JPEGImages`, `SegmentationClass`; IDs 0-20 plus ignore 255 | 1464/1449 |
| BDD100K | `images/{train,val}`, `labels/{train,val}`; optional `_train_id` mask suffix; IDs 0-18 plus ignore 255 | 7000/1000 |

ADE counts/layout follow the [publisher development kit](https://github.com/CSAILVision/sceneparsing);
VOC counts follow the [publisher statistics](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/dbstats.html).
BDD pairing/counts match the inspected experimental builder. BDD's root is the
semantic `seg` directory, not its parent archive folder. Orphans, ambiguous mask
aliases, missing files and cross-split path overlap fail. VOC `trainaug.txt`
presence fails pending an explicit augmented manifest; it is never silently
replaced with the smaller training list. Output classes retain numeric train-ID
order. Mask bytes are validated by the runner, not this standard-library builder;
release-sized synthetic tests do not certify real dataset contents. Native
Cityscapes/Context/Mapillary preparation remains external.

Profiles are explicit because experimental sources differ:

- `captured_fixed224_v1`: bicubic image resize and nearest target resize to 224
  for both splits. Current explicit BDD/Cityscapes/Context builders use this.
- `captured_pair_crop224_v1`: matched image/mask geometry from the older pair
  loader; cap the longest side at 512 (each resulting side at least 224), then
  train-only scale 0.5-2, random 224 crop and horizontal flip. Evaluation resizes
  to 224. Select only with actual run evidence, not a dataset-name guess.
- `captured_ade_crop224_v1`: the inspected ADE training geometry, with the same
  scale/crop/flip operations but **no 512-pixel cap**; evaluation resizes directly
  to 224. Use this profile for the inspected ADE route. The paired-crop profile
  is a different preprocessing identity, even though both output 224 pixels.

The input normalization bridges the existing providers' ImageNet-normalized
interface to the reference RGB input. Corrupt images and unknown IDs fail rather
than becoming black images or silently ignored labels. That is a deliberate
validation improvement over older permissive loaders.

## Metrics, training and boundaries

Evaluation visits every supplied item once, accumulates integer pixel confusion
counts, and averages IoU over classes with nonzero union. Pixel accuracy is
weighted by valid pixels, not image or batch means. ID 255 is ignored. An empty
valid evaluation population or all-ignore training batch is rejected rather than
reported as zero. Scores refer to the **224 grid**, not a native-resolution
benchmark. They remain `canonical_eligible: false` and `record_value: false`.

The tracked LP/AP recipe selects AdamW (captured betas 0.9/0.999), batch-scaled
learning rate, one-epoch warmup from 1e-6, cosine to 1e-6 and 20-epoch horizon.
LP decay is 0.0001; AP decay is 0.05. Bias/norm/positional parameters have no
decay. AP clips trainable gradients at 1.0. Shortened executions retain the
20-epoch schedule and report their terminal epoch, never the best epoch.

This port is single-process, with optional
[accumulation and CUDA BF16](EXTENDED_EXECUTION.md). FP32 and one physical batch
per update remain the default. The captured microbatch schedule horizon and
provider-specific tail policies are explicit; distributed execution and native
resume remain unported. The saved `probe.pt` contains only head/adapter state;
`results.json`, numeric metrics and the downstream manifest record the run.
An existing destination is refused. Released weights, full-data/GPU execution,
real dataset contents and exact run-to-table score matching remain unverified.

Strict RED tests preceded execution, provider wiring and membership conversion.
Reduced reference comparisons matched LP and both dense AP variants in
initialization, outputs, gradients and three SGD updates. Actual CLI tests check
artifact verification and overwrite refusal; five reduced providers exercise ten
LP/AP routes. See the PR for full-suite, mutation and Linux/CI results. These
observations do not establish released-model numerical reproduction.
