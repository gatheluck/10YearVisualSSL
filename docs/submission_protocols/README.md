# Supplied experiment protocols

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](../SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

These eight documents are supplied protocol specifications and a scope/replication
publication draft, preserved verbatim.
These describe experimental procedures separately from the coverage and
validation of their portable implementations in this package.
Instructions inside the documents describe the original experiment workflow;
they are not commands to submit jobs when reading this archive.

- [Scope and replication profiles](00_scope_and_replication.md)
- [Basic5 linear evaluation](BASIC5_LINEAR.md)
- [Basic5 attentive evaluation](BASIC5_ATTENTIVE.md)
- [Basic5 fine-tuning](BASIC5_FINETUNE.md)
- [Extended linear evaluation](EXTEND_LINEAR.md)
- [Extended attentive evaluation](EXTEND_ATTENTIVE.md)
- [Extended fine-tuning](EXTEND_FINETUNE.md)
- [Frontier system prompts](BASIC5_FRONTIER_SYSTEM_PROMPTS.md)

## Companion delivery and replication scope

The scope/replication publication draft was supplied on 2026-09-26 and is included
above. Its replication profiles are editorial proposals, not evidence that past
runs followed them. It does not change executable configurations or prescribe
a retroactive seed assignment. Its reference to `BASIC5_LINEAR_v1.md` uses the
legacy name of the supplied `BASIC5_LINEAR.md` specification.

The matching 81-configuration reference registries are now included:

- [Extended linear registry](EXTEND_LINEAR_v1.json)
- [Extended attentive overrides](EXTEND_ATTENTIVE_v1.json)
- [Extended fine-tuning overrides](EXTEND_FINETUNE_v1.json)

AP and FT inherit the LINEAR catalog and apply their named recipe overrides.
These are the reference version accompanying the supplied Markdown; a separate
82-configuration operational revision is not silently substituted. This catalog
is not proof of which configuration produced each reported result.

All original dataset locations are replaced by `${DATA_ROOT}/<dataset-id>`.
For example, the `action40` entry is `${DATA_ROOT}/action40`. Set up each directory
under your own data root, or map it to your local dataset before invoking a reader.
These paths deliberately do not preserve the original machine's directory layout.
JSON does not expand environment variables by itself: a consuming loader must
substitute `DATA_ROOT` explicitly and reject missing data. The registries are
protocol records, not drop-in launch configurations for every portable runner.
No optimizer, schedule, split, model list, metric or other non-path value changed.

The sanitized registries belong in version control with their documentation and
tests. Keep original private copies, local path mappings, credentials and actual
machine-specific resolved configurations outside Git and outside the submission.

The FT Markdown and JSON retain their original wording and precedence rules.
Dataset-specific overrides and run records must be checked before claiming a
particular reported result was reproduced; no source conflict is resolved here.

Frontier prompts request 100 samples per batch; the manuscript describes 500
samples per task. The batch manifests, vocabularies and response records are
needed to verify how those totals correspond. The prompts are unchanged;
neither a discrepancy nor a complete match is established without that evidence.

## Implementation boundary

The package integrates selected components from experimental implementations.
Inspected source implementations include Basic5 readers/training and Extend
linear, attentive and fine-tuning trainers. Their full model/task coverage,
family-specific readers, distributed paths and complete recipes have not all
been ported or validated in this package. The parent Basic5 and provider guides
describe those package boundaries; they do not assert that the corresponding
original experimental code is absent.

Source-code existence, successful porting, and correspondence to a particular
reported result are separate findings. Exact run identities and configuration
records are still needed for the last of these. The protocol catalog alone
does not establish which cells were completed.

The NYUv2 evaluator now refuses an evaluation with no valid pixels and skips
empty-mask batches. This intentionally corrects the captured zero fallback;
valid-batch metric formulas and aggregation are unchanged. It does not resolve
differences between historical depth protocols or certify manuscript scores.
