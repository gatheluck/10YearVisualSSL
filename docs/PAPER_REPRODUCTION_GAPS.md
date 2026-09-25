# Paper reproduction gap ledger

Audit date: 2026-09-25, starting public baseline `552ca1e` (PR 194 merged).
The current manuscript, all 14 sheets of the supplied workbook, captured active
experiment configurations and the public implementation were inspected. The
workbook predates the latest manuscript; agreement is checked per experiment,
not inferred from matching names. Private source revisions, hashes and numerical
comparison evidence remain outside Git.

This ledger prioritizes implementation coverage. Extracted feature arrays,
model-provider availability, passing small tests and paper-score reproduction
are different statuses. Earlier extracted-method counts and partial Basic5
support do not imply complete experiment coverage.

| Priority | Gap and evidence | Next implementation boundary | Current status |
| --- | --- | --- | --- |
| 1 | Step-4 DINOv3 Tables 34–35 have several head layouts and a Gram training stage; the public trainer previously only ran independent heads and the core objective | Sharing ownership, projection gradients, EMA, checkpoint/export, weighted objectives, clean-crop Gram lifecycle | [Implemented components and limits](DINOV3_STEP4.md); CPU comparisons and stage-boundary tests, no full ImageNet score run |
| 2 | Step-4 continuation and joint-assignment coverage | H1S200 checkpoint split at epoch 200; H1CORE continuation without Gram; H1JA joint assignment/mass logic | Component paths now implemented with version-1 checkpoint resume and reduced reference parity; full distributed/BF16 training and native checkpoint conversion still pending |
| 3 | IDv2 Step-4 configurations use two-view/multicrop bank behavior beyond the public path | Capture-aligned bank initialization, update order, sample identities and loss/update parity before end-to-end integration | Outstanding |
| 4 | Step-3 Basic5 has incomplete model/task/adaptation coverage | Extend the [provider support matrix](BASIC5_VISION_PROVIDERS.md), integrate remaining family-specific forward/gradient/optimizer paths in groups | Recent provider support remains partial; it is not whole-workbook reproduction |
| 5 | Extend and full evaluation result provenance are incomplete | Port supported task protocols, require actual checkpoint/config/data identities, then aggregate repeated runs | Outstanding; a cached score or provider import is insufficient |

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
  distributed Sinkhorn or CUDA BF16 behavior. Full 300-epoch training and paper
  accuracy have not been verified by this change.
- Original code and weights remain read-only. Further cluster work requires
  current private reservation and job/artifact checks to prevent duplicate runs.
- Do not regenerate publication claims from this ledger alone. Recheck current
  Git/PR state, source versions and physical artifacts before the next cycle.
