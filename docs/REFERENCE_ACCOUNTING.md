# Historical Extended result accounting

See [submission scope](SUBMISSION_SCOPE.md) for the distinction between
experimental sources, portable components and reproduced results.

This read-only auditor connects inspected native Extended LP/AP result schemas
to explicit per-cell run accounting. It does not import checkpoints, rerun
evaluation or certify paper scores. Use [portable accounting](DOWNSTREAM_ACCOUNTING.md)
for this package's own outputs.

`python -m downstream.reference_accounting --config /work/cells.json --out /work/new-report.json`

The [example plan](examples/reference_accounting.json) intentionally has no runs:
it produces an incomplete report with three missing seeds, not invented scores.
Replace its placeholder identities and counts using actual evidence. Multiple
datasets, models and LP/AP cells can be audited together; cells remain separate.
No attempt is chosen by score and no retry is launched.

## Required evidence

Each cell declares a unique `cell_id`, expected seeds, scheduled `epochs`,
`evaluation_count`, identity and metric definition. The evaluated count uses the
native evaluator's unit: images, objects, frames or valid pixels as applicable.
It is not inferred from a dataset name. Metric `field` selects a native scalar;
`name`, `unit` and `direction` label it. No percentage conversion, sign reversal,
best-epoch selection or metric alias guessing occurs.

Identity pins dataset, model, LINEAR/ATTENTIVE track, protocol ID/hash, recorded
checkpoint identity, recipe, readout, evaluation split, effective batch, realized
learning rate, world size, per-GPU batch and accumulation. Additional native
fields can be pinned, including reader, data revision or overrides. The checker
compares supplied fields, not every possible scientific setting. Reconcile
omitted recipe, data and code details before combining separate runs.

Each run specifies its seed, recorded `exit_status`, and three pinned artifacts:

```json
{
  "seed": 0,
  "exit_status": 0,
  "result": {"path": "/work/run/result.json", "sha256": "<actual SHA-256>"},
  "completion": {"path": "/work/run/COMPLETED_VALID.json", "sha256": "<actual SHA-256>"},
  "resume_meta": {"path": "/work/run/resume_meta.json", "sha256": "<actual SHA-256>"}
}
```

Use actual immutable files and lowercase 64-character hashes, not reconstructed
substitutes. Unknown exits may be null and are invalid. A native completion marker
does not override a failed enclosing attempt. Check job/attempt evidence before
supplying the exit code; the tool does not discover or authenticate scheduler
records. Campaign wrappers with nested `native` payloads are not flattened.

All three hashes must match. Result and completion JSON must agree; duplicate
keys, nonfinite JSON numbers and missing files are refused. Native status, seed,
identity and final-epoch selection must agree with the plan. Both result epoch
fields and resume metadata must match the scheduled epoch. Scores must be finite
real values, including valid zeros. Evaluated count must equal the declared
positive integer. This is stricter than the inspected native completion guard,
which accepts an integral float count and checks only finite primary/count values.
Records explicitly reporting nonzero `n_all_zero_clips` are invalid even if a
native completion marker exists. Absence of that diagnostic does not prove input
validity; other data-quality and subset limitations still require source review.

## Interpretation and limits

Missing seeds and invalid records remain separate. Accepted values produce mean
and sample standard deviation (`ddof=1`); n=1 has null deviation, and no accepted
records produce a null summary. Duplicate seeds/cell IDs or malformed plans abort
the audit. Exit zero means a report was written. `complete` means all expected
seed records passed artifact checks, not that the scientific experiment is verified.

Reports bind the entire plan hash, selected artifact hashes, checked epoch and
population, and identity hash. Source paths, job IDs and arbitrary error text are
omitted. User-supplied cell IDs and metric labels are retained: keep plans,
originals and reports outside Git and review them before sharing. Existing output
is never overwritten. A write failure can leave a partial report; check the exit
status and use a new destination for retries.

All outputs remain `canonical_eligible: false`, `record_value: false` and
`paper_score_verified: false`. Recorded identities do not rehash model weights.
Metadata does not prove correct pixels, execution authenticity or paper/workbook
correspondence. Failed, partial, stale or contradictory attempts need reconciliation.
FT, BasicFive reuse records, legacy checkpoint continuation and frontier response
evaluation have different schemas and remain outside this auditor.
