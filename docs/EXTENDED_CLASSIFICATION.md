# Extended image classification components

See [submission scope](SUBMISSION_SCOPE.md) and [final-paper coverage](FINAL_PAPER_COVERAGE.md).
This is a portable component execution path for the image-classification part
of final Tables 39-40, not a claim that the paper scores have been reproduced.
Those tables contain 24 image-classification datasets among 45 total datasets.
The larger protocol catalog is not the measured table population.
Only catalog entries whose primary metric is ordinary top-1 are accepted.
DomainNet's six-domain macro average requires a separate evaluator and is refused.

## What executes

`python -m downstream.extended_classification --config /path/config.json --out /path/new-run`

The runner reads the tracked Extended LP/AP optimizer recipes, freezes the
encoder (including evaluation mode), trains the selected head/reader, evaluates
every supplied validation item once, and saves final top-1/top-5, `probe.pt`,
`results.json`, and a hash-checked downstream manifest. An existing output
directory is refused. Corrupt images fail instead of becoming labelled black
images. Results remain `canonical_eligible: false` and `record_value: false`.

| Provider kind | LP image feature | Extended AP reader |
| --- | --- | --- |
| `dinov3_hf` | normalized final CLS | single cross-attention/MLP |
| `raev2_k7` | normalized K7 spatial mean | single cross-attention/MLP |
| `siglip2_g` | normalized official MAP pool | single cross-attention/MLP |
| `vjepa2_1` | normalized native T=1 patch mean | single cross-attention/MLP |
| `vggt_omega` | normalized final aggregator patch mean | cross attention, self attention, MLP |

All LP outputs receive L2 normalization. AP excludes prefix/register tokens and
uses the real spatial grid. The classifier uses normal initialization with
standard deviation 0.01 and zero bias, including the CLS/K7 families. This is
the Extended head, not the different BasicFive head initialization. Extended
SigLIP AP uses its own single-block reader; BasicFive remains unchanged.

## Explicit input and recipe identity

<!-- extended-example -->
```json
{
  "task": "extended_image_classification",
  "profile": "capture_extended_components",
  "dataset": "cifar10",
  "seed": 0,
  "device": "cuda",
  "data_root": "/path/images",
  "samples": "/path/samples.json",
  "transform_profile": "captured_rgb_rrc_v1",
  "adaptation": "attentive",
  "reader_profile": "captured_single_block_v1",
  "backbone": {"kind": "siglip2_g", "arch": "released", "encoder": "/path/local-vision-checkpoint", "img_size": 384, "patch_size": 16},
  "probe": {"epochs": 100, "batch_size": 256, "num_workers": 2}
}
```

For LP select `adaptation: frozen` and `reader_profile: null`. Checkpoint schemas
are those of the existing provider documents. No checkpoint is downloaded.
Reduced fixtures are not released model substitutes. The runner supports only
one process, FP32 and one physical batch per update; learning rate scales with
that batch. Gradient accumulation, BF16, native resume and distributed training
are not yet ported. AP clips trainable gradients at 1.0. Weight decay excludes
bias, normalization and positional parameters as in the experimental grouping.
Schedules retain the 100-epoch horizon even for an explicitly shortened run.
The terminal, possibly shortened epoch is reported, never the best epoch.

`samples.json` has exactly the following schema. Paths are relative to
`data_root`; label IDs index the ordered `classes` list. Both splits must be
nonempty. Duplicate paths, escaping links/paths, overlap and invalid labels
are rejected. The supplied membership file is hashed and its evidence retained;
this is not independent proof that a user's list is the official split.

```json
{"schema_version":1,"classes":["class-a","class-b"],"split_evidence":"Describe the actual annotation lists and version","train":[{"path":"train/a.jpg","target":0},{"path":"train/b.jpg","target":1}],"validation":[{"path":"test/a.jpg","target":0},{"path":"test/b.jpg","target":1}]}
```

All profiles evaluate with bicubic resize 256, center crop 224, RGB conversion
and model normalization. Training profiles must be selected from actual run
evidence, not inferred from dataset names:

- `captured_rgb_rrc_v1`: bicubic random-resized crop 224, scale 0.08-1, horizontal flip.
- `captured_small_crop_v1`: bicubic resize 224, random crop 224 with padding 4, horizontal flip.
- `captured_small_no_flip_v1`: the preceding small-image geometry without flip.

Current experimental classification builders reuse the first profile even for
some datasets whose protocol describes the small-image recipe. Keep that
source/protocol difference explicit until table rows are matched to their runs.
The portable loader fails on corrupt input and does not perform the older
loader's 1024-pixel thumbnail optimization.

## Official-membership conversion

`python -m downstream.extended_membership --config /path/membership-config.json --out /path/new-samples.json`

The config contains exactly `dataset` and `data_root`. Supported native layouts:

| Dataset | Membership source | Release counts: train/evaluation/classes |
| --- | --- | --- |
| `cub200` | CUB-200-2011 `classes.txt`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`; images under `images/` | 5994/5794/200 |
| `fgvc_aircraft` | `data/variants.txt`, `data/images_variant_trainval.txt`, `data/images_variant_test.txt`; `data/images/` | 6667/3333/100 |
| `dtd` | partition 1 `labels/train1.txt`, `labels/val1.txt`, `labels/test1.txt`; `images/` | 1880/1880/47 |

CUB joins numeric IDs and preserves class-ID order; it refuses an older CUB
release. Aircraft uses variant-list order and trainval membership. DTD validates
the separate validation list but does not merge it into training, and ignores
prepared split copies. Image inventories are checked for CUB/Aircraft; DTD checks
every listed item in all three lists. Annotation bytes are hashed; local hashes
and counts do not establish publisher authenticity. Native full-data execution
and annotation authenticity remain unverified. Other datasets require an
explicit externally prepared sample manifest; their native builders are not
claimed ported. Keep manifests containing local information outside Git.

## Evidence and remaining work

RED tests first exposed absent execution, provider declarations, membership
conversion and documentation. Small CPU tests cover both probe types, all five
reduced encoder integrations, invalid input and actual CLI artifact delivery.
Unchanged experimental LP and two AP classes matched initialization, outputs,
gradients and three SGD updates on identical synthetic features. This comparison
does not validate released weights or end-to-end paper scores. Private source
hashes and run records remain outside Git.

Video classification and the remaining detection, segmentation, pose, tracking,
flow, localization and reasoning routes still require their Extended-specific
readers, metrics, native input builders and run-to-table reconciliation. Native
BasicFive task components are not automatically equivalent Extended recipes.
