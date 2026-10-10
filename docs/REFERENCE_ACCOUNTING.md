# Historical downstream result accounting

See [submission scope](SUBMISSION_SCOPE.md) for the distinction between
experimental sources, portable components and reproduced results.

This read-only auditor connects inspected native Extended LP/AP and selected BasicFive LP/AP/FT result schemas
to explicit per-cell run accounting. It does not import checkpoints, rerun
evaluation or certify paper scores. Use [portable accounting](DOWNSTREAM_ACCOUNTING.md)
for this package's own outputs.

`python -m downstream.reference_accounting --config /work/cells.json --out /work/new-report.json`

The [example plan](examples/reference_accounting.json) intentionally has no runs:
it produces an incomplete report with three missing seeds, not invented scores.
Replace its placeholder identities and counts using actual evidence. Multiple
datasets, models and adaptation cells can be audited together; cells remain separate.
No attempt is chosen by score and no retry is launched.

## Required evidence

Each cell declares a unique `cell_id`, expected seeds, scheduled `epochs`,
`evaluation_count`, identity and metric definition. The evaluated count uses the
native evaluator's unit: images, objects, frames or valid pixels as applicable.
It is not inferred from a dataset name. Metric `field` selects a native scalar;
`name`, `unit` and `direction` label it. No percentage conversion, sign reversal,
best-epoch selection or metric alias guessing occurs.

For the existing Extended schema, identity pins dataset, model, LINEAR/ATTENTIVE track, protocol ID/hash, recorded
checkpoint identity, recipe, readout, evaluation split, effective batch, realized
learning rate, world size, per-GPU batch and accumulation. Additional native
fields can be pinned, including reader, data revision or overrides. The checker
compares supplied fields, not every possible scientific setting. Reconcile
omitted recipe, data and code details before combining separate runs.

Each Extended run specifies its seed, recorded `exit_status`, and three pinned artifacts:

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

## BasicFive native records

The [BasicFive example](examples/basic5_reference_accounting.json) contains three
empty LP/AP/FT cells. It reports missing repeats, not measured values. Keep the
top-level `schema_version: 1`; each BasicFive cell additionally declares
`source_schema` and `track` (`LINEAR`, `ATTENTIVE` or `FINETUNE`). Legacy Extended
plans and their strict population checks are unchanged. Never select a schema
from a method name alone: verify the actual fields and source version.

| `source_schema` | Inspected format | Required artifacts and final-epoch evidence |
| --- | --- | --- |
| `basic5_epoch_v1` | Common SigLIP2, CLIP, C-RADIO and Cosmos3 trainer result format | Pinned `result`, identical `completion`, and `resume_meta`; result `epochs`, metadata `epoch`, and `final_epoch_only: true` |
| `basic5_scheduled_v1` | DINOv3, RAEv2 K7 and V-JEPA2.1 child-run result format | Pinned `result` and `resume_meta`; `epochs_scheduled`, `epochs_run`, metadata `epoch`, and `final_epoch_canonical: true` |

The second format does not create a duplicate completion JSON, so the auditor
does not require or fabricate one. `resume_meta` may point to an actual
`progress.json` from the inspected writer if it contains the required epoch,
dataset, protocol and seed. These metadata identities must match. Do not create
replacement metadata for an absent historical file. Both formats still require
an explicit successful attempt exit; a completed result alone does not supply it.

Both identities pin `dataset`, `method_id`, `protocol_id`, `checkpoint_identity`,
`feature_layer`, `effective_batch`, `realized_lr` and `world_size`. The common
format additionally pins `campaign`, `track`, `protocol_sha256`, `input_size`,
`eval_views`, `per_gpu_batch` and `accumulation`. The scheduled format instead
pins `protocol_hash`, `input_resolution`, `evaluation_views`,
`micro_batch_per_gpu` and `grad_accum`. Include additional native fields when
needed to distinguish generations, recipes, data revisions or readouts.

The recorded LP protocol ID is **`BASIC5_FAIR_v1`**, even though its source
filename contains `BASIC5_LINEAR_v1`. AP and FT use `BASIC5_ATTENTIVE_v1` and
`BASIC5_FINETUNE_v1`. The checker binds the selected track to these inspected
IDs; it does not rename historical protocols or infer one from filenames.

All records require `status: COMPLETED_VALID` and `smoke: false`. An explicit
false/nonboolean `canonical` is refused; the scheduled format also requires
`canonical: true` to be present. The older common format omits that field.
This catches inspected smoke records whose status still says completed and
checkpoint-only evaluations whose final-epoch flag is false. It cannot establish
the training work performed from metadata alone.

The scheduled writer averages rank metrics and can serialize `metrics.n` as
an integral float. Only this schema accepts that representation, and only when
it exactly equals the declared positive integer population; fractions, missing
counts and booleans fail. The common and Extended schemas still require integer
counts. The expected count and its meaning must come from independently checked
membership/evaluator evidence. Some dense native records omit `n` entirely:
they remain invalid for this audit rather than borrowing a dataset size or
fabricating a count. Omega uses another completion schema and lacks population
evidence in the inspected records; it is not supported here.

Read-only source and result inspection on 2026-10-10 informed these adapters.
Tests cover both formats across all three tracks, source-style floating counts,
missing metadata, protocol/identity mismatch, malformed artifacts and CLI output
protection. This is an accounting implementation, not seven-family numerical
reproduction. No historical successful exits or paper-to-run mappings are invented.

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
Extended FT, BasicFive reuse/wrapper records, Omega accounting, legacy checkpoint
continuation and frontier response evaluation remain outside this auditor.
