# Experimental implementations and the portable submission package

This package is a selected, portable integration of experimental implementations
and public model dependencies. It is not an export of the entire original
experimental environment. Some reference pipelines have been inspected and
partly ported, while other parts have not yet been integrated or validated in
this package. Descriptions of those packaging and validation gaps do not mean
that the original experimental implementations are absent or that the reported
experiments were not performed.

Conversely, finding reference code does not by itself verify a particular
reported result. A result-to-run correspondence requires its actual checkpoint,
configuration, data/split, evaluation procedure and output records. We keep this
evidence question separate from whether code has been ported successfully.

## How to read status statements

Unless explicitly attributed to an original experiment, implementation,
support, test and reproduction status in the accompanying guides and method
READMEs describes **this portable package**, at the date recorded. Historical
notes retain their original dates and are not current execution records.
The supplied protocol documents specify intended experimental procedures;
their status labels are not an audit of this package or proof that every
catalog entry was executed.

| Status | Meaning for the submission package | What it does not establish |
| --- | --- | --- |
| Reference implementation inspected; port incomplete | Corresponding experimental code was inspected, but its full execution path has not been brought into this package | Absence of that implementation from the original experiment |
| Component ported; package validation incomplete | An executable component is present; the listed environment, scale, distributed path or end-to-end run has not been validated here | That the original experiment was not run, or that a small test reproduces a paper score |
| Reference or result correspondence unverified | Available source, settings or result records do not yet establish the particular correspondence | That the original implementation certainly exists, or that the reported result is invalid |
| Feature/interface unavailable for the selected model or runner | The requested input/output feature or execution interface is genuinely unavailable in that configuration | That another model, original pipeline or experimental configuration has the same limitation |
| Intentionally outside the comparison or package scope | For example, an as-is comparison uses released pretrained weights instead of repeating their original foundation-model pretraining | A missing implementation that can necessarily be remedied by porting |

`UNSUPPORTED` and related runtime labels retain their specific technical meaning.
They are not rewritten as successful execution or universally relabeled as
unfinished porting. Likewise, unresolved scientific differences are not resolved
by an editorial change.

## Current examples and their evidence boundaries

- **Basic5 readers and model integrations:** inspected experimental wrappers,
  readers and training code provide the references for the components described
  in the provider guides. Some family-specific readers and full recipes remain
  unintegrated or unreconciled in this package. This is a porting/configuration
  boundary, not a statement that the source experiments lacked those pipelines.
- **Extended evaluation:** linear, attentive and fine-tuning reference training
  implementations are present in the inspected experimental sources. This
  package does not yet integrate their full task catalog. Exact per-dataset
  coverage and correspondence to reported cells require the companion registries
  and actual run records; the existence of those trainers alone is insufficient.
- **Step-4 components:** reduced tests compare selected heads, losses, updates
  and checkpoint transitions with inspected experimental implementations. Native
  distributed execution, full-scale reruns and score matching have separate
  validation boundaries in this package. “Not validated here” concerns the port,
  not the execution history of the original experiments.
- **Released-weight comparisons:** an original model's large-scale pretraining
  may intentionally be outside the evaluated comparison. Using its released
  weights is different from failing to port an experiment that was required.

The companion index separately lists documents or registries awaiting delivery
to this package. That delivery status does not imply that their owners lack
the files. Preserve the distinction between source availability, porting,
package validation, and verification of each reported experimental result.
