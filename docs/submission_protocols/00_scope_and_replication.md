# Scope and replication profiles (publication draft)

## Status of this appendix

The following documents describe evaluation specifications, not a ledger of completed experiments. The Basic Five replication profiles below are an editorial proposal for the supplementary material. They do not establish that past runs followed either profile. The original Markdown files are preserved separately. No executable JSON configuration or experiment has been changed.

The legacy identifier BASIC5_FAIR_v1 in BASIC5_LINEAR_v1.md denotes the frozen-feature downstream protocol; it must not be confused with the FAIR pretraining comparison in Section 3. These cross-family downstream specifications must not be assumed to describe every historical ASIS/FAIR or Section 6 run.

## Basic Five: select one replication profile

For each model, dataset, and evaluation track, declare replication_profile before launching the corresponding evaluations:

- THREE_SEEDS: run downstream seeds [0, 1, 2]. Retain all individual scores and report their mean and standard deviation, with n = 3.
- SINGLE_SEED: run downstream seed [0]. Report the individual seed-0 score with n = 1. Do not report an estimated standard deviation or a value of zero as its substitute.

The two profiles differ only in the number of downstream repetitions. They retain the same pretrained checkpoint, data split, feature extraction, task head, optimizer, schedule, and metric. They do not change the pretraining seed or request new pretraining runs. For future runs, use the same designated profile across compared models within a task/track whenever feasible, and disclose exceptions.

Record requested seeds, completed technically valid seeds, all individual scores, and actual n for every reported cell. If only two seeds of THREE_SEEDS are available, report n = 2 rather than labeling it as a completed three-seed experiment. For historical results, report the actual seed coverage; do not retroactively claim that a profile was selected in advance. Document the standard-deviation convention used in the result exporter.

Do not choose seeds or replication profiles based on their scores. A zero or low score remains in the aggregate if its evaluation is technically valid. An incomplete or technically invalid evaluation is not a measured zero. Record exclusions and their technical reasons separately.

## Extended evaluation

The supplied EXTEND protocols retain seed [0] only. Report single-run scores with n = 1, not mean plus/minus standard deviation. A compatible Basic Five result may be reused only as its individual seed-0 result, not as a three-seed mean. Reuse requires agreement in checkpoint, data split, feature extraction, head, optimization, preprocessing, metric, and evaluation procedure.

The catalog's 81 configurations and planned job totals describe scope, not verified completion. The result inventory determines actual coverage and the denominator for each analysis.

## Remaining implementation details

The supplied EXTEND Markdown files defer exact dataset/recipe settings to their companion JSON files. Reproducing these Markdown files alone does not make this appendix a complete per-dataset specification. A finalized appendix must also point to an accessible dataset/protocol registry or include those details in a separate reader-facing table or Markdown document.

Operational scheduling and retry instructions in the supplied EXTEND documents are reproduced as execution instructions. They are not evidence of successful execution, scientific validity, or completion. This publication draft preserves those instructions rather than claiming they were followed.
