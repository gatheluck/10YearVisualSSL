# Supplied experiment protocols

These seven documents are supplied protocol specifications, preserved verbatim.
Their presence does not certify implementation coverage or reproduced scores.
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

The code package supports selected executable components. Its parent Basic5
protocol and provider guides document supported model/task/adaptation paths and
remaining gaps. In particular, supplying these specifications does not add
Extend dataset runners, missing attentive readers, native distributed training,
or complete fine-tuning recipes. Do not convert protocol tables into a claim
that all configurations ran or that paper scores were reproduced.

The NYUv2 evaluator now refuses an evaluation with no valid pixels and skips
empty-mask batches. This intentionally corrects the captured zero fallback;
valid-batch metric formulas and aggregation are unchanged. It does not resolve
differences between historical depth protocols or certify manuscript scores.
