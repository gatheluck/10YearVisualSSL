# Descriptive LP/AP comparison

See [package scope and replication terminology](SUBMISSION_SCOPE.md) for the
distinction between original experiments and portable component validation.

`python -m downstream.transfer_comparison` audits explicitly selected **reported
means**. It implements two mathematical operations described in the revised
manuscript: Figure 3's unordered LP-to-AP pair reversals and Figure 5's
separately normalized radial values. It does not select experiments, average
seeds, certify provenance, or reproduce training results.

The Basic Five seed aggregator remains `bin/aggregate-seeds.py`; this utility
neither replaces that tool nor assumes its seed or standard-deviation rules
apply to the revised figures. An experiment owner's older revision discusses
six models and conflicting depth definitions. Those historical values are not
silently adopted into this five-model comparison.

## Input and command

Supply a JSON object with exactly `schema_version: 1`, five distinct `models`,
and a nonempty `settings` list. Each setting requires:

- `id`: unique setting identifier; dataset subsets remain separate settings.
- `group`: explicit grouping label for descriptive counts, not a task dispatch.
- `metric`: `percent` for scores in [0, 100], or `error` for nonnegative,
  lower-is-better errors in the reported units.
- `protocol_id`: nonempty `LP` and `AP` recipe/measurement identities.
- `scores`: `LP` and `AP` objects, each containing all five model identifiers
  mapped to finite reported means. Missing values, booleans and extra models
  are rejected, not imputed.

For example, one synthetic setting could use model identifiers
`["a", "b", "c", "d", "e"]`, metric `percent`, and scores
`{"LP": {"a":10,"b":20,"c":30,"d":40,"e":50},
"AP": {"a":20,"b":10,"c":30,"d":40,"e":50}}`.
Use explicit protocol identities such as `lp-example` and `ap-example`.

```sh
python -m downstream.transfer_comparison --input selected-means.json --out comparison.json
```

The output must not already exist. It preserves the input values and protocol
identities, records the input-byte SHA256, and does not add the local input path.
Identifiers are copied verbatim, so use publication-safe identifiers when
sharing the report; this utility is not an anonymizer.

## Mathematics and limits

For each setting, all ten unordered model pairs are compared. A reversal means
strict opposite ordering under LP and AP. Errors reverse the direction of
comparison. Output margins are signed in the direction of better performance,
and remain in original units; they are not confidence intervals. Exact ties
are listed explicitly and are not counted as reversals. No epsilon or
statistical significance threshold is invented.

For radial display, percent scores become `score / 100`; errors become
`1 / (1 + error)`. Each setting and protocol is then divided by its own maximum
over the five models. LP and AP radial magnitudes must not be compared directly.
An all-zero percent column has no defined maximum normalization and is rejected.

Ranks are intervals when means tie. `unique_first` and `unique_second` count
only unambiguous first and second positions, both overall and by supplied group.
No arbitrary model-name tie breaker is used. Settings count equally, including
correlated subsets, as stated in the manuscript; these counts are descriptive,
not independent evidence or significance tests.

This module accepts selected means for review; it does not establish which
checkpoint, seed set, alignment or measurement protocol produced a paper cell.
For adopted metrics see the [Figure 5 ledger](CVPR_EXTENDED_COVERAGE.md) and [SSv2 AP records](SSV2_EXECUTION.md#figure-3-attentive-probes). Do not
substitute a classification task for detection, raw for aligned depth, or a
proposed protocol for an executed one. Only supply adopted means after resolving
those identities externally. The tool does not assert that a partial input
covers all 40 Figure 5 settings or all five Basic Five tasks.

Validation uses synthetic inputs, including error direction, ties, per-protocol
normalization, complete membership, group counts, malformed values and CLI
output protection. Passing these tests does not verify any manuscript number.

See the [package scope](SUBMISSION_SCOPE.md) for result attribution.
