# Native image inputs

See [package scope](SUBMISSION_SCOPE.md). These converters connect six reported
CLUE/SUN397 datasets and seven prepared-folder datasets in Tables 39–40 to the
existing five-provider [image LP/AP runner](EXTENDED_CLASSIFICATION.md).
They preserve the inspected experimental memberships. They do not certify the
paper's scores, upstream release authenticity or statistical independence.

## Staging

Keep source data read-only and choose a new output directory outside its tree.
Create a configuration with exactly `dataset` and `data_root`, for example:

```json
{"dataset": "clue_egg", "data_root": "/data/clue/egg"}
```

`python -m downstream.native_images --config /work/input.json --out /work/clue-staged`

For SUN397 use `dataset: sun397` and the directory containing `protocol_tar`.
The CLI checks full split/class counts. The Python `fixture_counts` argument is
only for reduced tests and is not accepted in CLI configuration.

| Dataset | Training | Evaluation | Classes |
| --- | ---: | ---: | ---: |
| clue_egg | 19,376 | 6,300 | 283 |
| clue_feather | 53,243 | 15,268 | 555 |
| clue_footprint | 12,575 | 3,834 | 117 |
| clue_skulls | 10,337 | 3,787 | 269 |
| clue_stools | 13,399 | 3,639 | 101 |
| sun397 | 19,850 | 19,850 | 397 |

These are requirements from the inspected builders, not a claim that conversion
or full training has been run on every current cluster input.

## Prepared train/val inputs

The same CLI additionally accepts these seven dataset identifiers:

| Dataset | Training | Evaluation | Classes | Per-class train / evaluation |
| --- | ---: | ---: | ---: | --- |
| action40 | 4,000 | 5,532 | 40 | 100 / variable |
| cifar10 | 50,000 | 10,000 | 10 | 5,000 / 1,000 |
| cifar100 | 50,000 | 10,000 | 100 | 500 / 100 |
| kmnist | 60,000 | 10,000 | 10 | 6,000 / 1,000 |
| imagenet_1percent | 12,811 | 50,000 | 1,000 | variable / 50 |
| imagenet_10percent | 128,116 | 50,000 | 1,000 | variable / 50 |
| omniglot15 | 24,345 | 8,115 | 1,623 | 15 / 5 |

Provide `train/class/image` and `val/class/image`. Both splits must contain the
same complete vocabulary. Sorted class names define labels and path order;
CIFAR-100 requires fine classes, KMNIST one-digit class names, and ImageNet
`n` followed by eight digits. Unknown files, nested images, empty classes,
selected symlinks, wrong counts and corrupt images are refused. `.DS_Store` is
ignored. CIFAR/KMNIST filenames are split-local array indices; the other four
profiles reject a repeated `class/filename` across splits. Identical bytes under
different valid identities are preserved and disclosed, matching the inspected
membership rule. Byte overlap does not establish independent samples.

These inputs are **prepared experimental exports**. Counts do not authenticate
upstream membership. No CIFAR pickle/IDX download or ImageNet subset sampling is
performed. Omniglot15 is the provided 15/5-per-character benchmark, **not** the
publisher's disjoint-alphabet background/evaluation partition. Source lists and
their correspondence to final table runs remain unverified.

ImageNet subsets may use a separate explicit evaluation root, containing `val`:

```json
{"dataset": "imagenet_1percent", "data_root": "/data/subset", "eval_data_root": "/data/imagenet"}
```

Only ImageNet subset profiles accept `eval_data_root`. The converter then reads
only `train` from the subset and `val` from the evaluation root; an unused subset
`val` symlink is not followed. This matches the inspected current builder's
separate-root correction. Without that option, both splits must be real local
directories. Output must be outside both source trees. `sources.json` records
relative names and training/evaluation roles, never the absolute input roots.

Use the [prepared-image training example](examples/extended_prepared_images.json)
after staging. The captured folder builders use `captured_rgb_rrc_v1` for all
seven datasets. For CIFAR-10/100, KMNIST and Omniglot15 this differs from the
registry's small-image augmentation recipe. The evidence explicitly flags that
conflict. Neither transform profile is silently declared the historical paper
recipe; selecting another profile requires independent run evidence. Flowers102
split reconciliation and Kuzushiji-Kanji's class/membership ambiguity remain
outside this port.

## CLUE membership and overlap

Provide `train/class/image` and `test/class/image`. Optional `valid` images are
diagnostic only: they never replace test or enter training. The sorted training
class names define labels. Evaluation classes must belong to that vocabulary.
Image bytes and train/test path order are preserved; no new random split is made.
The local category names `skulls` and `stools` are retained as source identifiers.

Filename submission prefixes are compared across train/test/valid. The output
records overlap counts and unknown IDs. Unparsed names do not establish
independence. In accordance with the selected reference, cross-split identical
image bytes are disclosed and retained for egg/feather/skulls; footprint/stools
refuse such content overlap. No deduplication, regrouping or repaired split is
silently substituted. Submission overlap and byte overlap are distinct evidence.
Externally supplied submission-ID mappings are not supported by this converter.

## SUN397 redistributed split 1

Provide `protocol_tar/manifest.jsonl` and the referenced `shards/*.tar`. Rows must
include archive, member, class_name, integer label, train/test split, positive byte
count and content_sha256. Members are content-addressed `images/<sha256>` files
with JPG/PNG/GIF/BMP suffixes. Every selected payload is decoded and checked against
its declared byte count and hash; ambiguous TAR members and links are refused.
No archive member is extracted to its original path.

Numeric labels and archive/member ordering follow the source, even when label
names are not alphabetical. Both splits require all classes and 50 occurrences
per class. Separate occurrences are retained even when content hashes overlap;
the overlap count is recorded. Original SUN filenames and upstream partition
lists are unavailable to this converter. It therefore does not claim a verified
join to upstream `Training_01.txt` / `Testing_01.txt`.

## Training, outputs and verification

The new directory contains byte-identical selected images, `sources.json` with
relative source identifiers and image hashes, and `samples.json` published last.
All selected inputs are validated before creating output, and content is checked
again while copying. An I/O error can leave partial assets but no success manifest;
use a new destination for a retry. Existing output is never overwritten. Generated
memberships, source inventories and datasets stay outside Git.

Use the [CLUE](examples/extended_clue.json) or [SUN397](examples/extended_sun397.json)
example with actual local encoder and staged-data paths. Select the corresponding
provider's LP/AP reader. The captured RGB random-resized-crop/flip training profile
and resize/center-crop evaluation profile remain in the existing image runner.
The converter does not alter optimizer, schedule or model behavior.

`python -m downstream.extended_classification --config /work/run.json --out /work/run-001`

Tests cover native membership, source-specific overlap policy, label order,
hashes, malformed inputs, write failures and actual reduced LP/AP training on all
six CLUE/SUN identifiers and seven prepared-folder identifiers. Reference
comparisons, mutations and regression results
are recorded separately in the PR. Full-data/released-weight GPU measurements and
historical result attribution remain unverified. Results retain
`canonical_eligible: false` and `record_value: false`.
