# Supplied experiment protocols

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](../SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

These seven documents are supplied protocol specifications, preserved verbatim.
These describe experimental procedures separately from the coverage and
validation of their portable implementations in this package.
Instructions inside the documents describe the original experiment workflow;
they are not commands to submit jobs when reading this archive.

- [Basic5 linear evaluation](BASIC5_LINEAR.md)
- [Basic5 attentive evaluation](BASIC5_ATTENTIVE.md)
- [Basic5 fine-tuning](BASIC5_FINETUNE.md)
- [Extended linear evaluation](EXTEND_LINEAR.md)
- [Extended attentive evaluation](EXTEND_ATTENTIVE.md)
- [Extended fine-tuning](EXTEND_FINETUNE.md)
- [Frontier system prompts](BASIC5_FRONTIER_SYSTEM_PROMPTS.md)

## Pending companion inputs

As of 2026-09-26, `00_scope_and_replication.md` has not been supplied for this
code package. The referenced `EXTEND_LINEAR_v1.json` and the corresponding
attentive/fine-tuning registries also require delivery and review. Do not infer
the detailed dataset recipes or replication assignment from these summaries.
This is a submission candidate, not a complete Appendix E companion bundle.

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
