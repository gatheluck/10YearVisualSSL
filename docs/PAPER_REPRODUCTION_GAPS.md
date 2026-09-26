# Portable-package porting and validation ledger

> **Legacy terminology:** numbered Step 1-4 labels in this file are historical
> code/experiment identifiers, not manuscript section numbers. See the
> [ASIS, CTRL and manuscript comparison map](PAPER_TERMINOLOGY.md) before matching a recipe to a paper result.

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

Audit date: 2026-09-25, starting public baseline `552ca1e` (PR 194 merged).
The current manuscript, all 14 sheets of the supplied workbook, captured active
experiment configurations and the public implementation were inspected. The
workbook predates the latest manuscript; agreement is checked per experiment,
not inferred from matching names. Private source revisions, hashes and numerical
comparison evidence remain outside Git.

This ledger prioritizes porting and validation of this submission package,
not whether the original experimental code exists or the reported experiments
were performed. Inspected reference implementations provide the basis for
selected ports; correspondence to each reported result remains a separate
evidence question. Extracted feature arrays,
model-provider availability, passing small tests and paper-score reproduction
are different statuses. Earlier extracted-method counts and partial Basic5
support do not imply complete experiment coverage.

| Priority | Package gap and reference evidence | Next porting/validation boundary | Package status |
| --- | --- | --- | --- |
| 1 | Step-4 DINOv3 Tables 34–35 have several head layouts and a Gram training stage; the public trainer previously only ran independent heads and the core objective | Sharing ownership, projection gradients, EMA, checkpoint/export, weighted objectives, clean-crop Gram lifecycle | [Implemented components and limits](DINOV3_STEP4.md); CPU comparisons and stage-boundary tests; no validated full ImageNet score rerun using this port |
| 2 | Step-4 continuation and joint-assignment coverage | H1S200 checkpoint split at epoch 200; H1CORE continuation without Gram; H1JA joint assignment/mass logic | Component paths now implemented with version-1 checkpoint resume and reduced reference parity; native distributed/BF16 paths and checkpoint conversion not fully integrated or validated in this package |
| 3 | IDv2 Step-4 configurations use two-view/multicrop bank behavior beyond the public path | Capture-aligned bank initialization, update order, sample identities and loss/update parity | [Seven component profiles implemented](IDV2_COMPONENTS.md), including adapter execution and checkpoint resume; native distributed integration and full-run score matching remain unvalidated in this package |
| 4 | The Basic5 port covers only part of the inspected experimental model/task/adaptation pipelines | Extend the [provider support matrix](BASIC5_VISION_PROVIDERS.md), integrate remaining family-specific forward/gradient/optimizer paths in groups | [SAM3/Cosmos3 added 14 component paths on 2026-09-26](BASIC5_PATCH_PROVIDERS.md); remaining family-specific/AP/COCO paths and recipes are not fully ported or reconciled here; full-workbook reruns in this package remain unvalidated |
| 5 | Extend LP/AP/FT reference trainers exist; their full catalog is not integrated in this package, and per-result correspondence needs separate verification | Port supported task protocols, require actual checkpoint/config/data identities, then aggregate repeated runs | Porting and result-to-run correspondence remain incomplete here; this is not a claim that the original trainers are absent |

## Decisions and boundaries

- User direction: maximize evidence-supported coverage in grouped PRs, using
  tests before implementation and leaving source contradictions pending.
- The newer H1-09 result combines shared heads with a mask upper bound of 0.65.
  Do not attribute the entire difference to head sharing or overwrite the
  earlier workbook's historical values.
- Attentive-pooling protocol differences, ImageNet fine-tuning augmentation and
  other conflicting source configurations require per-experiment reconciliation.
  Existing Basic5 detection boundaries, distributed execution and repeated-run
  aggregation must remain visible until implemented and measured.
- First priority has component parity evidence, not the captured full training
  environment. Single-process float32 cannot establish the reference's
  distributed Sinkhorn or CUDA BF16 behavior. A full 300-epoch rerun and paper-score matching using this port
  have not been verified by this change.
- Original code and weights remain read-only. Further cluster work requires
  current private reservation and job/artifact checks to prevent duplicate runs.
- Do not regenerate publication claims from this ledger alone. Recheck current
  Git/PR state, source versions and physical artifacts before the next cycle.
