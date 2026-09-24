# Audited validation-cache delivery

An existing reference cache can avoid duplicate inference. Before importing it,
verify the producer's checkpoint, preprocessing, readout, precision, dataset and
sample ordering against the actual reference code and run configuration. A
filename or completion marker alone does not establish these facts. Compare
selected cached rows with fresh reference inference when provenance is uncertain.

`bin/import-reference-cache.py` accepts a private JSON profile with `id`, `count`,
`feat_dim`, `expected_labels` and `shards`. Each shard has `features`, `labels`
and `indices`; each of these, and `expected_labels`, is an object containing a
local `path` and its `sha256`. `method` is an optional display name. Extra
provenance fields are retained in the output's `meta.json` as `source_profile`.
Keep private profiles and resulting metadata outside Git and public PRs.

Indices identify each row's original dataset position. Obtain them from the
audited producer: rank-concatenated distributed caches often need reordering.
Never reconstruct sample identity by sorting labels, since each class contains
multiple different images. Every index must occur exactly once and the ordered
labels must match the separately verified dataset labels.

Run `python3 bin/import-reference-cache.py --profile /path/to/private-profile.json --out /path/to/new-staging-directory` in an environment with NumPy.
The tool checks every input hash, row count, width, integer labels and indices,
sample order, finite nonzero features and unit-length normalization. It writes
float32 L2 `features.npy`, int64 `labels.npy`, `meta.json` and `result.json`.
Existing output directories are refused. Inputs are never modified. A write
failure may leave incomplete staging; only a successful exit and final result
are success. Do not point the tool directly at an existing shared delivery.

This is an integrity and conversion tool, not evidence of model equivalence or
paper-score reproduction. After staging, verify the shared copy and preserve
existing manifest records. Use stable model identifiers or a versioned figure
mapping: a later spreadsheet row number may identify an entirely different model.
