# Native CLUE and SUN397 inputs

See [package scope](SUBMISSION_SCOPE.md). These converters connect six reported
datasets in Tables 39–40 to the existing five-provider [image LP/AP runner](EXTENDED_CLASSIFICATION.md).
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
six dataset identifiers. Reference comparisons, mutations and regression results
are recorded separately in the PR. Full-data/released-weight GPU measurements and
historical result attribution remain unverified. Results retain
`canonical_eligible: false` and `record_value: false`.
