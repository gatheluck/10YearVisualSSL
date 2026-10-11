# Frontier matched-subset evaluation

See [package scope and terminology](SUBMISSION_SCOPE.md) for the distinction
between original experiments, portable functionality and reproduced results.

This entry point implements the explicit prediction interfaces and metrics in
the submitted manuscript's Table 42 and Appendix E.6, together with the
[verbatim prompts](submission_protocols/BASIC5_FRONTIER_SYSTEM_PROMPTS.md).
It scores supplied predictions; it does not call a model API or train a model.
The five interfaces remain separate from unified LP/AP/FT rankings. In
particular, the COCO interface here is category presence, not box detection.

The historical scoring implementation was not located in the inspected capture
snapshot or current experimental directories. This is a manuscript-specified
evaluator, not a verified port of the original scoring program. Original sample
manifests, raw outputs, retries, model identifiers/settings, and dense-output
conversion are still needed to reproduce the reported comparison. A passing
fixture does not certify those artifacts or the published scores.

## Run on one dataset

`python -m downstream.frontier --manifest /data/labelled-manifest.json --responses /data/batch1.json /data/batch2.json /data/batch3.json /data/batch4.json /data/batch5.json --out /results/frontier-score.json`

The output must not exist. The scorer reads all inputs and validates them before
creating the report. Each response file must be the JSON object requested by the
prompt, with exactly `dataset` and `predictions`. Markdown fences, duplicate JSON
keys, nonfinite JSON constants, extra fields and invalid labels are rejected.
It does not repair responses, select a retry, sort predictions, drop failures,
or average the five batch scores. Preserve all original attempts separately and
explicitly choose the five response files to score.

The fixed subset contains exactly 500 distinct sample IDs in a declared order.
Each of the five responses contains the corresponding consecutive 100 samples,
in that order. Unknown, duplicated, reordered or missing predictions fail.
The Python API `downstream.frontier.evaluate(manifest, responses)` accepts the
same objects and applies the same structural and scoring rules.

## Explicit labelled-manifest format

The top-level object has exactly four fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `dataset` | One of the exact prompt names listed below |
| `class_ids` | Full supplied vocabulary IDs, distinct nonnegative integers; empty for NYUv2 |
| `samples` | The 500 ordered labelled sample objects |

Each sample has a nonempty string `sample_id`, a `target`, and an input identity.
Image tasks use `input_sha256`, a lowercase 64-digit SHA256 of the supplied image
bytes. SSv2 instead uses `frame_sha256`, the 16 frame hashes in chronological
order. The scorer validates hash syntax and records the declaration; it does
not read images, authenticate dataset membership, verify chronological order,
or prove that a model used those inputs. The labelled manifest contains answers
and must **not** be supplied to the system being evaluated.

| Dataset | Vocabulary size | Sample target and queries | Response field | Metric |
| --- | --- | --- | --- | --- |
| `ImageNet-1k` | 1000 | Integer target class ID | `class_id` | Top-1 percentage over 500 images |
| `SSv2` | 174 | Integer target class ID; 16 ordered frame hashes | `class_id` | Top-1 percentage over 500 videos |
| `COCO` | 80 | List of distinct target category IDs, possibly empty | `category_ids` | Global category-presence Micro-F1 percentage |
| `ADE20K` | 150 | 32 target IDs and `points`: 32 ordered `[x,y]` pairs | `point_class_ids` | Accuracy percentage over 16,000 point decisions |
| `NYUv2` | 0 | 20 target codes and `pairs`: 20 ordered `[[ax,ay],[bx,by]]` pairs | `pair_answers` | Accuracy percentage over 10,000 pair decisions |

Every prediction object contains exactly `sample_id` and its response field.
Vocabulary IDs need not start at zero or be contiguous; use the supplied
vocabulary, including the actual COCO category IDs. Booleans, floats and numeric
strings are not integer labels. Coordinates are finite numbers in `[0,1]`.

NYUv2 targets use `0` (A closer), `1` (B closer), or `2` (equal). Predictions may
also use `3` (unable to determine); this is incorrect and stays in the denominator.
The evaluator requires already-established ground-truth codes. It does not infer
the equal-depth threshold, sample valid depth pixels, align depth, or convert a
continuous depth prediction into a code. Those historical procedures remain
unverified. Likewise ADE20K target sampling and conversion from dense predictions,
and COCO conversion from detector outputs, are outside this entry point.

For COCO, accumulate category-set intersections and differences over all images:
`Micro-F1 = 100 * 2TP / (2TP + FP + FN)`. Correctly empty images add no counts.
An all-empty target/prediction collection has an undefined denominator and fails;
the scorer does not assign an invented perfect score or silently choose zero.

## Output and verification boundary

The report contains the percentage, actual decision/count totals, dataset and
metric names, plus semantic SHA256 fingerprints of the labelled manifest and
each response. The CLI additionally fingerprints the original bytes of all six
files, so whitespace changes remain visible. Reports contain no input paths,
sample identifiers, labels, model names or raw responses. Private manifests,
responses and result reports should remain outside Git.

Results always retain `canonical_eligible: false`, `record_value: false`, and
`paper_score_verified: false`. They demonstrate evaluation of supplied labels
and predictions, not numerical reproduction of Table 42. Fixed subset selection,
raw model execution, retry provenance, matching all systems to the original
inputs, and historical conversion/scoring equivalence remain separate work.

Tests exercise all five full-size structural populations with synthetic labels,
imperfect predictions, globally pooled Micro-F1, depth abstentions, vocabulary
IDs, query boundaries, malformed responses, byte fingerprints, CLI delivery and
existing-output protection. Mutation and regression results belong in the PR.
