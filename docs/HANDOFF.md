# Cross-Mac project handoff

For ASIS, CTRL and legacy numbered labels, see the [manuscript terminology guide](PAPER_TERMINOLOGY.md).

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

Snapshot: 2026-09-19, after PR #183 merged. This document is a recovery entry
point, not a live dashboard. Read [agent instructions](../AGENTS.md) and
[repository rules](../CLAUDE.md), then refresh the actual Git/PR/cluster state.
The older [Basic5 crop handoff](HANDOFF_BASIC5_B.md) is historical; its branch,
delivery status and permission assumptions do not describe today's workflow.

### 2026-10-04: Extended structured-task integration

PR 218 merged as `a79f63f`; all 115 checks succeeded and main matched its tested
source tree. Existing video-submodule changes remain unrelated and preserved.
Read-only current sources confirm the target camera/question/pose components;
other task revisions differ from the snapshot and are not treated as parity.

The [structured guide](EXTENDED_STRUCTURED.md) documents five-dataset LP/AP,
native annotation conversion, exact-population metrics and portable continuation.
Behavioral RED preceded heads/metrics, data handling, training, readout integration
and delivery. Localization's source/companion conflict and 3DSRBench's local fitted
holdout remain explicit. Released-weight/GPU/full-data scores, historical run
attribution and native checkpoint import are pending. No cluster compute or
original input was modified. Record final validation and CI in the PR and stop
for review; private evidence remains outside Git.

### 2026-10-04: Extended optical-flow execution

PR 217 merged as `5db0452`; all 115 checks succeeded. Pulled main matches the
validated `a89c141` tree. The unrelated dirty video submodule is preserved.
Current flow targets/builders/runtime match the capture; the matching model
class bodies also match, while other task edits in their files are not parity.

The [flow guide](EXTENDED_FLOW.md) covers paired LP/AP training for five spatial
providers, Sintel/Middlebury local partitions, explicit Spring manifests, strict
target validity, coordinate-aware metrics, distributed execution and portable
continuation. The source feature-difference/L1 path conflicts with the Unified
companion's correlation/robust-endpoint description. This is explicitly a
noncanonical component profile; historical score attribution remains pending.

Behavioral RED preceded numerical/data components, execution, metric vocabulary,
cross-split RGB leakage protection and documentation/CI delivery. Selected
source comparisons cover initialization, outputs, gradients, three AdamW updates,
data and metrics. Record final regression, mutations, mandatory gates and CI in
the PR. No cluster compute was submitted and no original inputs were modified.
Released-weight/GPU/full-data scores, FT, BlinkVision revisions, Spring native
membership and legacy checkpoint imports remain pending. Stop for PR review.

### 2026-10-04: Extended NYUv2 execution

PR 216 merged as `6de256b`; its CI checks succeeded and main matches the
validated `a053902` tree. The unrelated dirty video submodule is preserved.
Current sources were queried read-only; changes outside the depth path were
not mistaken for snapshot parity. Depth data/head/loss behavior was inspected
separately from the existing BasicFive port.

The [depth guide](EXTENDED_DEPTH.md) describes native split-file inputs,
five-provider LP/AP training, shared accumulation/distributed execution and
portable epoch continuation. Positive mapping precedes upsampling; loss uses
full log variance. Neither replaces the distinct BasicFive behavior.
Evaluation pools valid pixels without alignment. Empty populations fail rather
than inheriting a misleading zero score. Outputs explicitly remain noncanonical.

Behavioral RED preceded the new components, provider normalization, execution,
metric contract and documentation/CI delivery. Record final numerical, mutation,
whole-suite and CI outcomes in the PR. Released weights, CUDA/BF16/NCCL,
full-data scores and historical-run attribution remain unverified. Other depth
datasets and conflicting tasks remain pending. No cluster compute or original
input was modified; private evidence stays outside Git. Stop for PR review.

### 2026-10-03: Extended detection LP/AP execution integration

PR 215 merged as `100e523`; all 115 checks succeeded. Pulled main matches the
validated `1e63bce` tree. The existing dirty video submodule remains preserved.
Read-only current LP/AP trainers match their captured sources; catalog status
is execution readiness, not a completed experiment or score certification.

The [detection guide](EXTENDED_DETECTION.md) now covers LP/AP training, variable
image-size batches, captured SGD warmup/milestones, provider-specific short-tail
behavior, two-rank execution and portable epoch continuation. Frozen encoder
weights are omitted from checkpoints without dropping the trainable pyramid.
Missing LVIS federated metadata fails before model construction. Evaluation uses
original annotations and occurs once on rank zero. Earlier evaluation-only
configurations remain supported. Existing output directories are protected.

Behavioral RED preceded component APIs, ragged batching, encoder-excluding
checkpoints, CLI routing, required CI routing and preflight metadata checks.
A source comparison exposed a floating-point warmup operation-order mismatch;
a failing exact test preceded its correction. Public tests and private evidence
cover reduced model updates, source schedule/parameter groups and continuation.
Record final gate and CI outcomes in the PR; local passes do not certify CI.

The source/catalog geometry conflict, FT, GPU/NCCL, full-data/released-weight
runs and historical table provenance remain unresolved. No original code,
weights, snapshot or cluster job was modified. Private source/configuration
copies and evidence stay outside Git. Stop for PR review after delivery.

### 2026-10-02: Extended detection and instance evaluation components

PR 214 merged as `89ac6e3`; all 115 checks succeeded after CI restart. Pulled
main matches the tested `b6004d2` tree. Existing video submodule edits remain
unchanged. Read-only current sources and the rerun catalog were inspected;
`runnable` is not a completed experiment or a verified paper score.

[Detection components](EXTENDED_DETECTION.md) add explicit COCO/LIVECell/LVIS
inputs, transposed FPN Faster/Mask R-CNN compositions for the five frozen/AP
providers, and official original-annotation prediction evaluation. The prior
Basic5 compositions are unchanged. Unknown benchmark identities and invalid
masks are refused rather than approximated. LVIS's NumPy compatibility adapter
is local to the evaluator instance; no installed package is patched.

The first tests exercised missing component APIs; executable CLI RED separately
proved that merely importing the module did not deliver results. Further RED
covered CI routing, out-of-range mask probabilities and empty LVIS predictions.
Source parity, reduced real-provider updates and contract output checks remain
separate from full-data/GPU and score validation. Required final gates and CI
status belong in this task's PR evidence; stop for review after delivery.

Full Extended detection training/distributed/resume integration remains open.
The source's conditional 224/256 versus catalog 800/1333 geometry is explicitly
unresolved; the low-level factory requires a named profile and does not certify
its paper provenance. No compute jobs, original inputs, weights or snapshots
were changed. Evidence and current run configuration copies remain outside Git.

### 2026-10-01: Extended video classification integration

PR 213 merged as `2b45f2c`; all 115 checks succeeded. Pulled main matches the
previously tested `51e8c23` tree. Existing video submodule changes remain preserved.
This cycle adds HMDB51/UCF101 split-1 membership conversion, captured 16-frame
inputs and five-provider LP/AP through the shared accumulation, distributed and
epoch-continuation implementation. See [video support](EXTENDED_VIDEO.md) for
native decoder dependencies, source-specific readouts and scope boundaries.

Current original video paths and strict split builders match the snapshot;
changes elsewhere concern tracking. Actual rerun settings retain the 50-epoch
LP/AP recipes. The current catalog flags historical split contamination in
HMDB51 and selected UCF101 runs. Corrected rerun availability does not identify
which run supplied a submitted table value. Keep this result-provenance gap
pending; do not copy historical scores into verified portable results.

Behavioral tests preceded the new APIs; a separate behavioral RED exposed
symlinked annotation acceptance, and CI-routing RED preceded workflow changes.
Source comparisons cover reduced heads, updates, sampling, pixels and memberships.
The distributed worker uses an explicit importable module name; discovery-mode
RED/GREEN and mutation checks guard against a test that passes only when named.
The reduced provider/CLI/Gloo and Linux native-decoder checks are distinct from
released-weight CUDA/NCCL, authentic full-data and final-score reproduction.
Private source/configuration copies remain outside Git. No cluster compute job,
original data, checkpoint or capture snapshot was modified. Refresh the task PR
for final mandatory gates and CI; stop for review after delivery. Remaining
Extended families, full recipes and table/run provenance remain open.

### 2026-09-30: Extended portable epoch continuation

PR 212 merged as `63263e0`; all 115 checks succeeded. Pulled main matched the
previously tested `0f7ae2c` tree, and existing video submodule changes were
preserved byte-for-byte. This cycle adds shared epoch checkpoints and strict
continuation to image and semantic LP/AP, retaining five provider profiles and
their existing update rules. It does not import historical experimental
checkpoints or certify a reproduced paper cell. See
[continuation](EXTENDED_EXECUTION.md#epoch-checkpoints-and-continuation) for the
format, immutable-input/software assumptions and remaining CUDA boundaries.

Current original LP/AP training and revision helpers matched captured sources
in a read-only inspection. The source saves completed epoch/step and optimizer
state; the portable format additionally preserves per-rank random state and
rejects identity/state inconsistencies instead of silently restarting. Behavioral
RED preceded implementation and CI routing. Additional RED tests exposed empty
optimizer state, inconsistent learning-rate state and nonfinite optimizer state.
Completed training can also retry terminal evaluation in a new directory without
further optimizer updates, matching the original completed-epoch loop boundary.
No cluster compute job,
original data, weight or capture snapshot was changed. Private source evidence
stays outside Git. Refresh the task PR for final mutation/regression gates and CI;
stop for review after delivery. Other task families, unresolved source/protocol
identities, full recipes, released-weight/full-data execution and result-to-table
mapping remain incomplete. This entry supersedes earlier portable-resume gaps,
not the distinction between original experiments and public component support.

### 2026-09-30: Cars and MNIST native binary inputs

PR 211 merged as `9183575`; all 115 checks succeeded. Pulled main matched the
tested `7c0839f` tree, with existing video submodule changes preserved. This
cycle adds Cars MAT annotation conversion and MNIST IDX-to-lossless-PNG staging
for the existing five-provider image LP/AP path. Neither changes model recipes
nor certifies reproduced scores. Original inputs are read only; MNIST requires
a new external output directory and publishes its sample manifest last.

Captured and current original builders were compared separately on four splits
and 80 transformed fixture images each; labels and tensors match exactly after
common normalization. Strict RED preceded the converters and CI routing; a
second RED exposed incomplete-manifest publication during a simulated write
failure. Cars integer-label validation also has an observed RED/GREEN check.
An additional RED confirmed that parent-traversing output paths could create
an intermediate directory in the synthetic source tree; such output paths are
now refused before any write. No original experimental data was used in that test.
See [the input guide](EXTENDED_CLASSIFICATION.md) for layouts and the unresolved
MNIST transform/registry difference. The separate Flowers102 split disagreement,
other native input/task ports, resume and released-weight/full-score validation
remain pending. No compute job or original data/weight was changed. Private
source fingerprints remain outside Git. Refresh the task PR for final gate,
mutation and CI results; this entry is not a live completion record.

### 2026-09-30: Extended native classification membership

PR 210 merged as `6f01364`; all 115 checks succeeded, and pulled main matched
the tested `9641b11` tree. Preexisting video submodule changes were preserved.
The next grouped change adds Food-101, Pets, IP102 and MIT Indoor-67 native
annotation conversion to the existing image LP/AP sample contract. It does not
add providers, alter training recipes or certify full-data results. Both splits
are validated; IP102's separate validation split is checked and excluded. MIT
official lists override physical directory placement without uniform quotas.
See [the input guide](EXTENDED_CLASSIFICATION.md) for layouts, deliberate
prepared-copy differences and the unresolved Flowers102 train/validation conflict.

Current original files differed from Capture; both revisions were inspected and
executed on reduced fixtures, with ten matching split/label comparisons each.
Private source paths/hashes remain outside Git. Synthetic CLI counts and LP/AP
input/update tests are distinct from released-data/GPU/score reproduction.
No compute job, original dataset or weight was changed. Refresh the task PR for
final gates, mutation results and CI; stop for review after branch/PR delivery.

### 2026-09-29: Extended distributed execution

PR 209 merged as `29e9991`, with all 115 checks successful; main matched the
validated `2f60122` tree. Existing video submodule changes were preserved.
The next grouped change adds torchrun execution to both Extended image and
semantic LP/AP runners, across their existing five provider profiles. This
supersedes the prior checkpoint's single-process limitation, not its remaining
recipe, resume or numerical-reproduction limitations. Captured training sources
use padded DistributedSampler membership, epoch reseeding, per-microbatch DDP
synchronization, rank-specific Python/NumPy/Torch seed offsets and global
effective-batch LR scaling. The original preflight is not replayed, so this is
not a claim of complete historical random-transform trajectories. Rank-zero full-population
evaluation and probe-only export preserve the existing task/metric identities.
Small-population automatic batch changes are deliberately refused rather than
silently copied. See [execution](EXTENDED_EXECUTION.md) for ownership and failure
limits. Private source hashes and cluster access findings remain outside Git.
Local reduced Gloo parity is distinct from NCCL/BF16 released-weight and full-data
experiments. Refresh the implementation PR for final validation and delivery.

### 2026-09-29: Extended accumulation and precision

PR 208 merged as `eb352c2`, with all 115 checks successful; the pulled main tree
matched tested head `dc7a3ef`. Existing video submodule changes were preserved.
The next grouped cycle adds [shared execution](EXTENDED_EXECUTION.md) to Extended
image and semantic LP/AP across the five providers. This supersedes the earlier
entries' statement that these two runners only support one FP32 physical batch.
Current original training entrypoints were read-only inspected: all LP routes
discard accumulation tails, while AP differs by provider. Preserve those
differences and the observed optimizer-update/microbatch-horizon schedule clock;
do not silently reinterpret them as a conventional epoch-level schedule.
CPU tests cover accumulated updates, loss/gradient failures, clipping, frozen
encoders and explicit precision boundaries. CUDA released-weight parity, native
resume, distributed execution, remaining task families and run-to-table mapping
are still pending. No compute job or original input was changed. Private source
and run evidence stays outside Git. Refresh the task PR for final validation,
mutation, CI and delivery status; this checkpoint is not a claim of completion.

### 2026-09-29: offline push validation

The user requested a separate stacked PR above the dense-component branch,
leaving its running CI unchanged. Long whole-suite validation moves before
Git transport; a private, per-checkout receipt gates the actual pushed commit.
SSH remains the transport. Network-only retries reuse matching evidence;
changed inputs, failed validation, missing Torch and expired evidence refuse
publication. The former skip environment variable no longer bypasses the gate.
See [the workflow](PUSH_VALIDATION.md) for commands, limitations and recovery.
The full local suite remains distinct from the Linux CI matrix and GPU tests.
No scientific implementation, original input or cluster job is changed.
Refresh the task PR for final test counts and delivery/CI status.

### 2026-09-27: final-paper component port

September 28 follow-up after PR 206: all 115 checks succeeded and main was
fast-forwarded to `01a7375`, matching the tested `c9a750e` tree. Existing video
submodule changes were preserved. The next grouped cycle adds
[Extended semantic LP/AP](EXTENDED_SEGMENTATION.md), five explicit dense provider
profiles, pixel-population metrics and ADE/VOC/BDD native membership conversion.
Captured and current original dense heads/readers were compared read-only;
reduced initialization/output/gradient/three-update comparisons matched. Actual
recorded runs still use accumulation/BF16 not ported here. The runner explicitly
reports 224-grid component scores, not native-resolution or paper reproduction.
ADE's uncapped training crop is an explicit third profile, distinct from the
older pair loader's 512-pixel cap and the current fixed-224 builders.
SpaceNet metric and SUN RGB-D task contradictions remain pending. Remaining
Extended families, full recipes and native/GPU/result validation are next work.
No compute job or original input was changed. Consult the new PR for final
validation/CI/delivery state; private source/run evidence stays outside Git.

September 28 follow-up after PR 205: main was fast-forwarded to `48ae6d6`;
its tree matched tested head `473b901`, all 115 CI checks succeeded, and
preexisting submodule changes were preserved. The next grouped branch adds
[Extended image LP/AP](EXTENDED_CLASSIFICATION.md), explicit provider reader
declarations and official-membership converters for CUB-200-2011, DTD and
Aircraft. The shared image path is relevant to 24 of the 45 reported datasets;
it is not a claim of completed native dataset ports or score reproduction.
Current experimental sources and one recorded classification run were read
without changing originals. Some current small-image builders use the RGB
random-resized-crop path, so transform profiles remain explicit. Distributed
execution, accumulation, BF16, continuation, other native builders and the
remaining Extended task families are still pending. No compute was submitted.
Refresh the new PR for validation and delivery state.

September 28 follow-up: PR 204 merged as `8c1acc4`, with all 113 checks
successful. The next grouped cycle adds [C-RADIO](BASIC5_RADIO.md) and
[downstream final-run accounting](DOWNSTREAM_ACCOUNTING.md). All eight BasicFive
AP/FT families now have components, not full-paper numerical reproduction.
The C-RADIO source read from the current experimental workspace matched the
captured backbone. Its local loader, separate summary/spatial features, video
CPE, native detector and layer policy are tested; 14 fixture task routes execute.
An actual reduced official model matched unchanged reference wrapper methods
for outputs, gradients and three updates in private CPU tests. Released-weight
GPU tests and full-scale results remain pending. No cluster compute job was
submitted and no original checkpoint or extracted artifact was changed.

Accounting audits existing portable final-epoch artifacts, retains low/zero
scores and incomplete repeat counts, and uses sample standard deviation. It
does not convert historical schemas or infer table cells. Next priorities are
Extended's reported 45 datasets/five models, remaining complete BasicFive
recipes, distributed precision/continuation, and actual run-to-table mapping.
Keep ImageNet FT augmentation and source metric/reader conflicts pending.
Refresh the task PR for delivery/CI status rather than inferring success here.

CI follow-up, 2026-09-28: the first PR 204 run had 57 failed jobs out of
113. All 55 locked jobs and the Omega container reached a COCO integration
test without `pycocotools`; the RAEv2 container requested a smoke module whose
name differed from the method directory. The fully populated local environment
did not reproduce either environment/entrypoint boundary. The correction keeps
the pure Omega detector checks active in method environments, isolates the COCO
evaluation dependency, checks all task extras before starting the grouped routes,
and aligns the RAEv2 smoke with the actual container command. Fresh-process
absence/presence tests and dynamic smoke-module loading guard these boundaries;
the downstream CI job explicitly runs the complete-environment regression.
This changes test delivery, not scientific model behavior or dependency locks.
Both Linux/amd64 images were built locally and their locked environments and
component tests passed, with task-extra skips explicit. Fresh-process complete
environment controls execute those task tests without skips. This is component
validation, not released-weight/GPU or full-paper result validation.
Keep corrections in the existing PR and refresh its latest checks before merging;
the earlier local success does not establish success of the corrected CI run.

Follow-up after PR 203: main was fast-forwarded to `a6dd760`; its full tree
matched tested head `e985a3a`, and existing submodule edits were preserved.
The separate two-family task adds [K7](BASIC5_K7.md) and
[Omega](BASIC5_OMEGA.md), with 28 small-fixture LP/AP/FT routes and source-specific
video/readout/COCO-label behavior. Direct reference comparisons cover outputs,
gradients and three updates, not full paper results. C-RADIO, complete recipes,
Extended tasks, distributed training and run accounting remain in the
[current audit](FINAL_PAPER_COVERAGE.md). Originals and weights remain read-only;
no extraction is repeated. Refresh the task PR for current validation/delivery.

PR 202 was merged and main was fast-forwarded to `9d3b1ee`; its tree matched the
previously tested PR head, and all 109 reported CI checks succeeded. Unrelated
submodule edits were preserved. New work uses a separate branch and PR.

The user requests grouped, strict-TDD porting toward the final submitted paper,
with no silent resolution of source contradictions. The
[final coverage audit](FINAL_PAPER_COVERAGE.md) replaces earlier table numbering
and priorities. [Explicit reader/native detector profiles](BASIC5_NATIVE_PATHS.md)
cover 35 small-fixture routes across five existing providers. Six unchanged
reference reader/pyramid components matched initialization, outputs, gradients
and three updates in private CPU comparisons. Full-data/GPU scores remain
unverified; these are noncanonical components, not full-paper reproduction.

Read-only source comparison found three changed files among 22 comparable files
in the inspected subset: depth loss default and added evaluation diagnostics.
The current public depth loss already matches the revised default. Median-
aligned experimental metrics versus the final unaligned metric specification
remain a run-mapping question. No cluster jobs were submitted and originals,
weights and existing extracted features were unchanged. Private fingerprints
and comparisons remain outside Git. Validation counts and delivery state belong
to the task PR; refresh Git and CI rather than treating this note as live state.

### 2026-09-26: initial experiment protocol companions

The user supplied eight reconstructions of experiments preceding Unified LP/AP/FT
and authorized separating historical evidence from new unification proposals.
The [initial protocol companions](initial_protocols/README.md) retain detailed
recipes as attributed candidate specifications, with explicit historical and
proposal sections. RAEv2 readout, 4DFM resolution and VideoSSL schedule/seed
changes are not represented as completed reruns. Private machine paths, project
identifiers and job IDs are removed; originals and audit evidence remain private.

This is separate from the Extend-registry delivery branch and must not be added
to its ongoing PR. No runtime, original source, weights or experiments change.
Local ZIP tests verify delivery and selected evidence/proposal boundaries; full
score-to-run correspondence remains unverified. The existing submission ZIP must
not be described as containing these additions until explicitly rebuilt and
inspected. Preserve the supplied Unified companions and unrelated submodule work.

### 2026-09-26: supplementary v4 reconciliation

The supplied v4 adds manuscript terminology and a reviewer README, now retained
in [the mapping](PAPER_TERMINOLOGY.md) and the
[export template](templates/REVIEWER_README.md). Legacy filenames remain stable.
The user confirmed that the initial COCO experiments are object detection;
previous classification attributions are superseded. The supplied v4 detection
recipes are retained as reconstructions, not independently verified run settings.
Initial and current Unified specifications remain separate. The eight current
companions and three Extend JSON registries are already tracked and unchanged;
Charades and AVA remain distinct Extended tasks. Generated dependency indexes
remain export artifacts. No runtime or result is changed. The supplied v4 ZIP
is not rewritten by this repository change. Local tests check navigation and
protocol boundaries; score-to-run matching remains unverified.

## Verified baseline and what it means

- Default branch: `main`. PR #183 merge: `1c5ed6d66ad437db9d4fe4afb1f1e5cfe7a50776`.
- Merged code tree: `367ab0700dcc43b294b61dfbf692e60b209eb892`, identical to
  the tested PR head `c59b1b51900ea731f441ad9e3b10228fe45a6b24`.
- PRs #178--183 delivered protocol/result guards and partial Basic5 depth,
  temporal, attention, segmentation/detection and full-gradient components.
  All six are merged at this checkpoint. No implementation branch beyond #183
  was developed in this task; this handoff itself is a separate documentation PR.
- Four downstream CLIs support explicit frozen/attentive/finetune component
  execution. FT is limited to the verified differentiable timm ViT provider.
  See [current protocol boundaries](BASIC5_PROTOCOL.md) and
  [downstream execution](DOWNSTREAM.md). Components are noncanonical and
  nonrecordable; they do not reproduce every paper recipe or score.
- Before merge: ABCI torch whole suite 3,472 tests / 280 skips, exit 0;
  GPU suite 30 tests without skips; eight CPU/CUDA captured-composition checks
  matched outputs, gradients and two SGD updates. These use small fixtures,
  not full released-weight benchmarks. Final-tree focused run: 118 tests /
  seven CUDA-only skips; 19 new and 68 existing mutants detected. Whole-suite
  discovery preceded one final CI-guard test and documentation-only edit;
  executable production code was unchanged and the final tests ran separately.
- After merge: 30 protocol/language/CI tests passed without skips; complete
  tree identity checked. Full/GPU suites were not rerun for an identical tree.
  Recover CI results from GitHub; this document does not certify future CI.

## ImageNet validation features: do not lose the accounting

The last recorded collection is **47 of 51 methods: 34 delivered on ABCI plus
13 already held by the user**. This is a recorded 2026-09-18 delivery, not a new
live filesystem audit. See [the chronological progress record](STEP1_EXTRACTION_PROGRESS.md),
[feature sweep](FEATURE_SWEEP.md) and [weight acquisition](STEP1_WEIGHTS.md).
Use the latest dated section; early entries describe blockers since resolved.

The user's thirteen completed methods are DINOv2, Franca, AIMv2, BEiTv2, CAE,
Cosmos3 Super, Data2Vec2, EVA02, SigLIP, VAR, VideoMAE, V-JEPA2 AC and V-JEPA2.
Do not count these as missing because their arrays are absent from the shared
collection. AIM, CLIP, ImageGPT and SAM3 were the four unresolved extractions;
checkpoint/evaluation mapping needs fresh verification before further work.
CMC, PIRL, MSN, V-JEPA and LeJEPA were resolved and delivered in PR #177.

The agreed collection is `imagenet_val_l2`, with root README and consolidated
manifest, then per-method `features.npy`, `labels.npy`, `meta.json`, `result.json`.
Full results have 50,000 finite float32 L2 rows and sorted-WNID int64 labels,
1,000 classes with 50 images each; compare recorded hashes and actual metadata.
Feature extraction does not train an accuracy probe or establish canonical
Basic5 conformance. Never overwrite an existing profile with a different
resolution, preprocessing, readout or checkpoint identity.
The exact private destination and delivery evidence travel outside Git.

## Remaining work and decision boundaries

### Implementation follow-up (2026-09-19; separate from the checkpoint above)

The frozen/AP optimizer component adds opt-in task-specific SGD/AdamW,
weight decay and batch-scaled LR across the four existing task CLIs. It
preserves default legacy and FT execution, records the realized settings, and
rejects unsupported FT/distributed use. See
[the optimizer component](BASIC5_PROTOCOL.md#opt-in-frozenap-optimizer-components-2026-09-19).
This addresses only optimizer construction and full-batch accounting, not
schedules, accumulation or canonical eligibility. The supplied protocol and
captured cosine endpoints disagree away from reference batch size; keep that
question pending. No score, seed-count or feature-identity conflict was resolved
by this implementation. Delivery and current CI status must be refreshed from
the task PR; the historical checkpoint above does not certify this change.

PR #185 was verified merged on 2026-09-20 at
`7ac4756fd4af4792275ae17e818a1315a7293ddd`; all 107 PR CI checks succeeded.
Its tree matched the tested PR head, and unrelated submodule edits were
preserved. This is a dated verification, not a claim about future CI.

The 2026-09-20 follow-up adds an explicit COCO frozen/AP schedule component:
500-update linear warmup and decay at epochs 8 and 11, with update accounting
and truncated-run reporting. See [scope and unresolved questions](BASIC5_PROTOCOL.md#coco-frozenap-schedule-component-2026-09-20).
The user reaffirmed implementation only where paper, workbook/protocol context
and captured code support it, using strict TDD; ambiguous interpretations stay
pending. Workbook scores are not replaced or used as synthetic-test targets.
Refresh the follow-up PR status before claiming delivery or validation.

CI correction on 2026-09-20: PR #186's initial run exposed an incomplete
dependency guard in its new schedule tests. All 42 failed method-lock logs
retrieved at diagnosis had the same six CLI assertion failures. Importing the
COCO runner did not establish that its runtime `timm`/`pycocotools` dependencies
were installed. The fix reuses the existing COCO guard for CLI tests only,
retains numerical tests in partial environments, and adds fresh-process
regressions for both missing dependencies and actual execution with complete
dependencies. The dedicated downstream job runs these regressions. Local and
GPU success of the initial implementation did not certify the method-lock
matrix; refresh PR checks for the correction's current outcome. Scientific
schedule behavior is unchanged.

PR #186 was verified merged on 2026-09-20 at
`d54d133d97a10e4e01340ce4834a93c46366c9c1`, with an identical tested code tree.
All 107 PR checks and five post-merge jobs succeeded. The final initial-failure
audit covered 52 method locks, extending the 42-log diagnosis above; all 52
had the same six CLI failures. This is a historical checkpoint.

The user requested sequential strict-TDD implementation of unambiguous gaps,
grouped for fewer reviews, stopping at decisions requiring user input. The
next bounded component implements ADE20K/NYUv2 attentive scheduling at batch 8
only; see [the current boundary](BASIC5_PROTOCOL.md#dense-ap-reference-batch-schedule-2026-09-20).
The same grouped change also adds captured ADE20K/NYUv2 FT color jitter,
whose strengths and factory selection agree with paper/protocol evidence.
Other-batch endpoints, accumulation clock semantics and FT parameter mappings
remain unresolved. Refresh the task PR and evidence for validation status.

The remaining priorities below still apply, with optimizer construction for
these eight frozen/AP component paths and opt-in COCO scheduling implemented.

1. Reconcile the old private audit/implementation plan with the now-merged
   #178--183 components before selecting work. The plan predates them.
2. Full Basic5 LP/AP/FT recipes still need verified optimizer parameter groups,
   layer-wise decay and zero-decay exceptions, effective-batch LR scaling,
   warmup/schedules, FT augmentation and accumulation, plus native video paths
   and model-specific capabilities/readouts. ImageNet AP/FT integration and
   canonical result eligibility require their own evidence. Do not merely
   remove the noncanonical flag.
3. Model candidates from the asset audit include CLIP-L, SigLIP2 variants,
   DINOv3-7B, C-RADIO, V-JEPA2.1 and VGGT-Omega. These are audit candidates,
   not confirmed readable checkpoints or completed ports. Preserve RAE K7 and
   other native/multi-layer profiles as distinct unresolved identities.
4. Broader 3D/4D, Extend datasets and Step-4 work remain separate. All spreadsheet
   tabs matter; a Basic5 implementation does not cover all paper experiments.
5. Numerical disagreements among the paper, spreadsheet and protocols are
   deferred by the user. Record conflicts with exact sources; do not silently
   choose a value. CapturePrivate behavior is a required reference, not an
   automatic resolution of a protocol contradiction.

## Transfer inventory: Git and private material

| Material | Transfer route | Restore/verification |
|---|---|---|
| This repository, docs, locks, pinned submodules | Clone from GitHub | Verify default branch, HEAD and hooks; submodules at pinned revisions |
| CapturePrivate repository | Separately authenticated private clone | Its default is `ops`; experiment sources are on `origin/snapshots`, not the tooling checkout |
| Papers, all workbook tabs, supplied LP/AP/FT protocols | Private transfer of `10YearVisualSSLAssets` | Hash comparison; do not publish the source documents here |
| ABCI execution settings, job/output/weight evidence | Private transfer of `$HOME/.local/state/10YearVisualSSL/abci/` | Read `README.md` and `execution.json`; verify reservation validity and real current paths |
| Audit, implementation plan, RED/GREEN/mutation/reference evidence | Private transfer of `assets-audit-20260918/` | Read `AUDIT.md`, `BASIC5_IMPLEMENTATION_PLAN.md`, source hashes, tab coverage and the latest task evidence |
| User-wide agent agreements | Reviewed copy/merge of `$CODEX_HOME/AGENTS.md` (normally under `$HOME/.codex`) | Preserve new-machine agreements; project essentials are also in this repository's AGENTS.md |
| Uncommitted local changes | Private binary patches plus full changed/untracked files and base revisions | Inspect and use `git apply --check` before any restoration; never reset automatically |
| GitHub, ABCI SSH, gated weight credentials | Reauthenticate/configure separately | Do not copy tokens, browser cookies, SSH private keys or entire Codex state into this repository |

At this checkpoint the port had nine modified YAML files inside `third_party/vjepa2`;
CapturePrivate had modified `docs/STATUS.md` and untracked
`docs/STEP1_WEIGHT_SURVEY_20260916.md`. They are unrelated user work, not part of
#183. Preserve them privately; cloning alone cannot restore them.
Keep both old checkouts intact until the new environment is verified.

The private handoff package is separate from this PR. It contains a restore map,
hash manifest, source documents, audit/execution evidence and local-change
backups. It excludes authentication stores and the actual feature/weight arrays.
Some older logs embed old absolute paths: treat them as historical evidence;
map to new paths in a new local record instead of rewriting those logs wholesale.
Virtual environments and compiled caches should be recreated for the new Mac.

## New Mac: restoration procedure

1. Install Git and the Codex app, sign in, and authenticate access to both GitHub
   repositories. Configure ABCI SSH separately using the owner's approved
   credentials and host verification. Do not disable host-key verification.
2. Use fresh destination directories for the following recipe. Set `PORT_DIR`
   and `CAPTURE_DIR` to absolute, nonexistent paths outside each other. The two
   remotes are `git@github.com:gatheluck/10YearVisualSSL.git` and
   `git@github.com:gatheluck/10YearVisualSSLCapturePrivate.git`, respectively.
   For example set the four shell variables in your current shell, then run:

<!-- handoff-fresh-clone -->
```sh
git clone --recurse-submodules "$PORT_REMOTE" "$PORT_DIR" && git -C "$PORT_DIR" config core.hooksPath .githooks && git clone "$CAPTURE_REMOTE" "$CAPTURE_DIR"
```

   This is a fresh-clone recipe: rerunning against populated destinations must
   refuse, preserving their contents. For existing clones inspect status and
   remotes first, then fetch and use fast-forward-only updates as appropriate;
   never delete a directory just to rerun setup. Do not use `submodule --remote`.
3. If this handoff PR is not yet merged, fetch and check out its branch
   `codex/mac-handoff-context` to read these files. Do not assume they are on
   `main` until its merge is verified. For subsequent development, branch from
   the appropriate verified base rather than editing main.
4. Transfer the private package directly using an encrypted channel or encrypted
   removable storage. Extract into a fresh Git-external directory. Verify its
   supplied SHA-256 manifest, then follow its restore map. Restore ABCI local
   state to the same home-relative location only if absent; compare/merge any
   existing state instead of overwriting it. Owner-only permissions are suitable
   for local private state. Add the private asset/evidence folders as accessible
   project context without adding them to Git.
5. Open the port repository as the main project folder in Codex. Make the private
   capture and restored assets/evidence available as additional context. Read
   root AGENTS.md and CLAUDE.md explicitly on the first run. Do not depend on
   this chat's history or copy the entire application database/configuration.
   Confirm plugins/authentication on the new machine; browser login does not
   prove Git CLI or connector authentication.
6. Recreate Python using `.python-version` and the existing dependency locks.
   [Platform setup](PLATFORMS.md), [GPU rules](GPU.md) and the workflow define
   dependencies. Do not transplant a venv or replace the Linux CUDA locks with
   unverified Mac packages. CPU/tooling skips on a Mac are not GPU validation.
   Read `.githooks/pre-commit` and `.githooks/pre-push`; enable hooks as above and
   do not bypass them. Existing `tests/run-tests.sh` is the base regression gate.
   After committing, use `python3 bin/validate-push.py push --remote origin`;
   the whole Torch suite runs offline before Git transport.
7. Verify ABCI access read-only first: account, private destination, source and
   checkpoint readability, reservation status and any already running jobs.
   Never resubmit a job merely because the Mac changed. Compute tests use only
   the verified reservation, separate task workspaces and read-only inputs.
   Paths and runtime versions in `execution.json` may be historical; inspect
   the newer task evidence before choosing the current test environment.
8. Restore unrelated patches only after reviewing them and checking their exact
   base revisions. Do not push changes to vendor upstreams or capture snapshots.
   Check current automations on both machines and avoid duplicate monitoring or
   job submissions. This handoff does not copy/activate automations or authorize
   changing their settings.

## Resume prompt

Paste the following into a new Codex task rooted in the cloned port, after
supplying the private package's actual local location:

> Resume this project from AGENTS.md, CLAUDE.md and docs/HANDOFF.md. Read the
> restored private transfer map, ABCI execution state, asset audit, implementation
> plan and latest task evidence. Inspect Git/PR state and preserve local changes.
> PR #183 was merged at the recorded checkpoint; verify today's state. Distinguish
> 47 recorded extracted methods (34 delivered plus 13 user-held) from four
> unresolved ones, and partial Basic5 components from complete canonical recipes.
> Compare intended changes with CapturePrivate snapshot code and all relevant
> workbook tabs/paper/protocols. Keep source disagreements pending. First report
> verified status, missing transfer items and a grouped next implementation plan
> in Japanese. Once context and access are restored, continue unambiguous work
> with strict behavioral RED/GREEN and mutation tests, reserved-node GPU tests
> when needed, separate writable workspaces and unchanged originals. Use a new
> codex branch, commit, push and PR after validation; never merge automatically.
> Externalize evidence regularly. Keep private paths, accounts, reservation IDs,
> credentials, source documents and reference copies out of GitHub. Do not repeat
> completed extractions or jobs without checking the actual artifacts first.

## Validation limits

The handoff tests execute the clone recipe with real Git against isolated local
remotes, checking pinned submodule checkout, hooks and refusal to overwrite an
existing checkout. They also verify local document links. They do not test
GitHub authentication, transfer to another Mac, ABCI access from that Mac, or the
scientific truth of every prose statement. Those are explicit receiver checks.

Codex's documented worktree handoff moves between local checkouts/worktrees;
it is not the basis for this cross-machine restoration. See
[official worktree documentation](https://developers.openai.com/es-419/docs/environments/git-worktrees).

## Figure 2 expansion in progress (2026-09-21)

The user authorized immediate extraction of the additional Figure 2 profiles,
using up to eight reserved nodes while preventing duplicate jobs. See
[reference extraction](FEATURE_SWEEP.md#audited-reference-extraction-2026-09-21).
Refresh private execution state and the real queue/output directories before
resuming. Completed worker outputs are not equivalent to shared delivery, and
this expansion must not be added to the historical 47-method count without
checking distinct profile identities. The task's source mappings and execution
records remain outside Git. Review the task PR for its current validation state.

Correction to the older transfer inventory: the nine modified V-JEPA 2 YAML
paths were independently reproduced in a fresh checkout on the new Mac. They
are uppercase/lowercase path collisions on a case-insensitive filesystem,
not established intentional user edits. The pinned upstream contains both
spellings. Do not restore one spelling over the other or commit the checkout
artifact; the private case audit preserves the evidence. This correction does
not reclassify unrelated changes in other repositories.

The extraction wrapper has local behavioral and mutation coverage, plus full
validation-set runs on reserved GPUs. These runs reuse audited original
functions; they do not certify matching probe scores or full Basic5 conformance.
A delivery audit caught two duplicate retry outputs. All four file hashes for
each duplicate matched the already delivered profile; extra workspace copies
were retained separately. Existing shared artifacts were not overwritten.
Before any retry, check both active jobs and successful output directories;
a log observed before another worker finishes is not a stable retry decision.
The latest per-profile completion, delivery and unresolved-source ledger is in
private execution state. Refresh it before reporting collection totals.

## Step-3 reproduction expansion (2026-09-21)

The user requires publication-oriented coverage of all unambiguous paper,
workbook and captured-code differences, especially Step 3 onward. This is a
requirement, not a declaration that existing feature providers reproduce all
experiments. Preserve ambiguous source identities and report them separately.

The [new encoder integration](../methods/vjepa2_1/README.md) connects the pinned
image/video encoder to all four downstream component CLIs and adds explicit
provider capabilities for differentiable execution and the captured pyramid.
SSv2 now preserves native video tokens when the provider supplies them. The
existing image-provider frame averaging remains unchanged. The shared author
namespace preparation is reused by the older action-conditioned encoder.

The method/CompEval plan items remain incomplete: no complete ImageNet or
model-specific LP/AP/FT recipe or paper-score reproduction is claimed. The
older statement that every non-timm provider is frozen-only is superseded for
this verified opt-in provider. FT parameter grouping needs model-specific
mapping: captured families use different block/name rules; do not silently
apply one family's rules to all providers. Captured source and current data
availability must be refreshed before broadening the next group of ports.

The user explicitly authorized refreshing CapturePrivate. The new append-only
source snapshot is `c2d7b913077ffe96e8cfb978cf80062cc07db880`; originals and weights
remain unchanged. It exposed a dense-reader initialization/residual discrepancy,
now corrected after behavioral RED/GREEN. Independent comparisons with both
current shared and video-family readers matched initial parameters, outputs,
input/parameter gradients and three SGD updates exactly on CPU fixtures.
Query-reader architectures differ across captured families and remain pending;
distributed synchronization is outside this single-process component update.
No changed paper or workbook files were found in the snapshot comparison. This
does not establish that external documents are current or reproduce their scores.

CI follow-up (2026-09-22): the encoder image built and its runtime tests passed,
but a new workflow assertion failed because `.github` is deliberately excluded
from images. Gate only that repository assertion on workflow-directory presence.
A fresh-process regression removes git from PATH and makes workflows
absent, runs the method smoke, and requires at least nine executed tests. It
first reproduced the exact `KeyError`, then passed; skipping model coverage is
not the fix. The full local gate and image CI must be refreshed after this change.

## Grouped training coverage (2026-09-22, in progress)

PR #189 was verified merged at `4e6b92d6dd4618291ee3665497f5bc0194edfade`;
its tree matched the tested head and all 109 PR checks succeeded. Refresh later
CI and branch state before resuming. Preserve the unrelated case-colliding
submodule checkout described above.

The user requested a large grouped expansion under deadline, retaining strict
TDD, private reference evidence and review before merge. The follow-up branch
adds provider-owned FT groups across four tasks, twelve reference-batch
schedules, and online ImageNet LP/AP. The ImageNet FT model composition is
available for comparison but its full execution recipe is deliberately refused
until augmentation evidence is reconciled. See the
[current boundaries](BASIC5_PROTOCOL.md#broader-training-components-2026-09-22)
and [execution interface](DOWNSTREAM.md#extended-training-components-2026-09-22).

Capture snapshot `d8e82adc094ca68624d0f98bd6425c9ec976966f` supplied current
optimizer, schedule and classification references. Numerical source comparisons
passed on small CPU fixtures; no full-score, all-model or new GPU validation is
implied. The task's RED/GREEN, source hashes, mutation and delivery evidence must
be consulted for its final status. Remaining work includes ImageNet FT recipe
ambiguity, family-specific query readers, accumulation/distributed semantics,
additional model adapters, Extend and Step-4 experiments and real score/seed
reproduction. Do not mark these complete because component execution succeeds.

### Anonymous supplementary archive (2026-09-23)

The user requested an early, repeatable, test-driven submission ZIP workflow.
[The archive tool and guide](SUBMISSION_ARCHIVE.md) export committed sources and
parent-pinned submodule contents with a private allowlist, disclosure checks,
hash-bound replacements/approvals and deterministic ZIP output. Git metadata is
omitted; copyright notices are retained and cannot be silently anonymized.
Private identifying terms and audit reports belong outside Git. The example
policy is deliberately minimal and must not be mistaken for a complete export.

Fixture tests cover packaging and blocking behavior, including execution from an
unpacked archive. They do not establish that a real project submission is ready.
Before final delivery, review the full file selection, all identity terms,
license conflicts, approved binary/link contents and unpacked experiment runs.
Unknown identifiers and indirectly identifying prose still need human review.
This packaging task does not change scientific implementations or ABCI inputs.

Initial local validation: 29 archive tests passed; 29/29 archive mutants were
detected after adding a separate original-size test (a small replacement must
not bypass the input-size budget). The focused archive/repository guard set
passed 59 tests before two final test refinements. The base suite passed 3,572
tests (1,545 dependency/environment skips) before the malformed-policy refinement.
Final full-suite and CI results must be read from the task/PR evidence;
these counts do not certify a later commit or a real submission ZIP.
The repository argument must be the verified checkout root; allowing a nested
directory was shown to weaken output containment and is now rejected.


### Real submission trial follow-up (2026-09-24)

PR #191 was verified merged at `eb4ccbd34b7e7fefd34c5eaf71711787ff0a2f10`,
with all 109 PR checks successful. Its full-tree trial was blocked before ZIP
creation; it was never a reviewed submission artifact. The user requested an
actual anonymized archive, including an anonymous first-party holder label while
preserving third-party notices. The optional hash-bound root-license authorization
in [the archive guide](SUBMISSION_ARCHIVE.md) supports that narrow review variant.
This supersedes the earlier blanket statement that no notice can be anonymized:
general replacements and third-party changes remain prohibited.

The public README configuration examples also failed to create their advertised
JSON files; executable tests now cover file creation and configuration resolution.
Private archive selection, identifying terms, replacements, approvals, artifact
hashes and validation evidence stay outside Git. Consult current task evidence
for the real ZIP status; neither a fixture pass nor a broad source audit proves
submission readiness or paper-score reproduction. The dirty submodule checkout
remains unrelated and must be preserved.

### Broader vision-provider integration (2026-09-24)

Starting baseline: PR #192 merged, main `efebbd1a424e8f8ac1b6ee53ff72e2615aed6be7`.
The user requested the largest evidence-supported implementation gaps first,
strict TDD and grouped review. The follow-up integrates three local vision
families with explicit global/spatial features and FT policies; see the
[support matrix and limitations](BASIC5_VISION_PROVIDERS.md). It does not change
the anonymous ZIP workflow or certify previously built archives against new code.

Capture snapshot `d8e82adc094ca68624d0f98bd6425c9ec976966f` supplied the family
wrappers and optimizer policies. Reduced CPU comparisons covered outputs,
input gradients, three updates and all 129 fixture parameter policies. CLI
tests exercise 23 supported model/task/adaptation paths and enforce noncanonical
results. Consult current PR evidence for final suite, mutation and CI outcomes;
these observations do not establish released-weight/GPU or score reproduction.
No cluster jobs or original weights were changed. AP differences, two detection
boundaries, ImageNet FT augmentation, distributed execution, Extend and broader
Step-4 remain outstanding. Preserve the unrelated dirty video submodule.

The trial also found duplicated first-party attribution in the root README;
it now links to the authoritative LICENSE instead. Three Basic5 workflow-only
assertions attempted to read missing CI files in a Git-free export. They now
use the established checkout guard on those methods only; scientific tests and
invocation-parser controls remain active. A subprocess regression verifies both
Git-free skipping and failure for a checkout with missing workflow definitions.

The broad exported-suite trial also exposed dependency-check failures because
`.gitmodules` was absent: bundled upstream packages were classified as undeclared
index dependencies. The archive now generates a paths-only `upstream_sources.json`
and the existing shared scanner consumes it, preserving checks for genuinely
undeclared external imports. This is not a relaxed assertion or a new duplicate
file scanner. The index has no account URLs or revision IDs; invalid declarations
fail. Refresh artifact hashes and private delivery evidence after this fix.

### 2026-09-24: PR 193 method-environment CI correction

The new HF task-integration test initially ran when Transformers was present
but full downstream dependencies were absent. The `sam3` and `cosmos3_super`
locked jobs failed importing partially initialized task fixtures. The test now
checks both downstream fixture dependency flags before importing their helpers;
model-level tests remain enabled. A fresh-process regression with SciPy blocked
reproduces the original failure and verifies the explicit integration-only skip.
The full downstream environment must continue to execute all 23 integration
paths without skips. A separate `aimv2` container failure occurred fetching the
Docker Hub authentication token (connection reset), before the image build or
tests. This does not establish a model defect. See the PR for rerun outcomes;
local success alone does not establish the full CI matrix result.

### 2026-09-24: audited validation-cache staging

PR 193 merged at `20eb5e5`; its 109 PR jobs passed. The user next requested
ImageNet validation feature delivery for five additional figure models. Existing
reference caches must be checked before submitting duplicate inference. The
[cache delivery tool](REFERENCE_CACHE_DELIVERY.md) pins inputs, restores explicit
sample indices and validates canonical L2 output. It does not establish that a
reference model has been ported or that its paper scores are reproduced. Actual
model variants, source/checkpoint hashes, jobs, GPU checks and delivery evidence
remain in private execution state. Figure row numbers have changed between
versions; never silently use an old row-number directory for a new model.

### 2026-09-25: Step-4 head and Gram component integration

Starting baseline: PR 194 merged, main `552ca1e`. The user requested a fresh
paper/workbook/capture audit and grouped strict-TDD implementation of the largest
confirmed gap. See the [prioritized ledger](PAPER_REPRODUCTION_GAPS.md) and
[DINOv3 Step-4 guide](DINOV3_STEP4.md) for the implemented boundary and limits.

Four projection layouts, weighted objectives and the optional fixed-clock Gram
stage now have CPU behavioral coverage, including real optimizer updates,
clean-crop targets, snapshot/refresh boundaries and backbone-only exports. Reduced
reference comparisons matched head initialization, outputs, gradients and three
SGD/EMA updates. Shared-prototype construction preserves the reference's repeated
codebook initialization without duplicate parameter ownership.
This does not establish full ImageNet, distributed, BF16 or paper-score parity.
H1S200, H1CORE continuation, H1JA, IDv2 banks and the remaining Basic5/Extend
coverage are still pending. The new manuscript's H1-09 also changes masking.

The default core path is preserved. Full captured training checkpoints are not
drop-in inputs, and unsupported resume now fails. No new cluster jobs were
submitted for this change. Preserve the unrelated dirty video submodule; private
source snapshots, run identities and comparison outputs stay outside Git.
Consult the PR for gate and mutation outcomes; local coverage is not CI evidence.

### 2026-09-25: Step-4 continuation follow-up

PR 195 merged at `23669d0`; all 109 PR checks succeeded. The user requested
the largest remaining confirmed gaps under the approaching deadline, with
strict TDD and consistent documentation. This follow-up connects H1S200 head
splitting, H1CORE fixed-clock continuation and H1JA weighted joint assignment.
See [checkpoint boundaries and examples](DINOV3_STEP4.md#continuation-and-joint-assignment-components).
This supersedes the preceding blanket continuation/resume gap for component
checkpoints; native distributed checkpoints and canonical scores remain pending.

Tests compare real interrupted/resumed CPU training, optimizer/head ownership,
Gram lifecycle and three-process uneven/empty-rank assignment. Captured joint
outputs and three post-split AdamW/EMA updates match exactly on reduced inputs.
Private sources and detailed evidence remain outside Git. No cluster jobs or
original weights were modified; the unrelated dirty video submodule is preserved.

The local CI dry planner previously exported the full repository and submodules
for every unexecuted row. It now exports once, still executes discovery and
expands every row. Real execution retains a fresh checkout per row. Behavioral
tests cover both branches; this does not skip scientific tests or CI jobs.
Consult the task PR for final regression, mutation and CI outcomes.

### 2026-09-25: IDv2 component follow-up

PR 196 merged at `61ce5f4`; all 109 PR checks succeeded. The earlier sam3
runner communication failure passed on rerun without code changes. Main was
fast-forwarded and its code tree matched the previously tested PR head.

The user reaffirmed broad, strict-TDD coverage under the deadline. Seven
[IDv2 components](IDV2_COMPONENTS.md) now connect the existing ViT backbone,
captured view/loss/bank behavior, AdamW updates, checkpoint continuation and
backbone export. The new component format restores NCE partition and prototype
state as well as RNG; it does not retrofit legacy checkpoints. This supersedes
the earlier blanket IDv2 implementation gap. Private reference comparisons and
CPU tests do not certify native DDP/BF16, full ImageNet scores or whole-workbook
reproduction. Next major groups remain Basic5 provider/adaptation integration,
Extend protocols, distributed training and result provenance. Keep scientific
source contradictions pending. No cluster jobs or original inputs were changed;
unrelated dirty submodule changes remain excluded. Refresh the task PR for final
validation and CI status; private evidence is stored outside Git.

### 2026-09-26: Basic5 trunk and final-merger follow-up

PR 197 merged at `1b98078`; all 109 PR checks succeeded. Main was updated
and its tree matched the tested IDv2 branch. The user again requested grouped,
strict-TDD work on the largest remaining supported reproduction gaps.

[SAM3 and Cosmos3 Basic5 components](BASIC5_PATCH_PROVIDERS.md) now expose
14 frozen/FT execution paths with explicit local loading, captured pixel/grid
handling, classification readouts and layer-decay policy. The final-merger
Cosmos3 representation is separate from the older patch-token extraction path;
do not overwrite or relabel those existing features. Tests exercise real tiny
models and task output contracts. Private comparisons execute unchanged captured
wrappers with loader-injected tiny models through three updates and all 75
parameter-group assignments. They do not certify released checkpoints, GPU,
distributed training or paper scores. See the PR for final gate/mutation results.

COCO geometry, AP readers and ImageNet FT recipe discrepancies remain pending.
Other provider families, Extend protocols and full-run result provenance remain
major work. No original inputs or cluster jobs were changed. The unrelated dirty
video submodule is preserved. Private source identities and numerical evidence
remain outside Git.

### 2026-09-26: submission finalization candidate

Baseline: PR 198 merged, main `ef6b1a8`. The user requested minimal submission
corrections without new scientific features or repeated CI cycles. The
[seven supplied protocols](submission_protocols/README.md) are preserved as
specifications, with the outstanding scope document and JSON registries explicitly
pending. Do not equate document delivery with executable coverage or score parity.

The NYUv2 evaluator now refuses empty/all-invalid evaluations and omits empty-mask
batches from component averages. The metric helper rejects an empty mask.
Valid-batch formulas, actual zero errors and training-loss behavior are preserved.
This deliberately supersedes the captured invalid-input zero fallback in the
interest of truthful evaluation reporting. Historical depth alignment and full
recipe differences remain unresolved; no new training or cluster job is implied.

The previous private archive policy predates recent runtime modules and guides.
Refresh its selection and review changed replacements/approvals before rebuilding.
Keep identity lists, source hashes, audit reports and ZIPs outside Git. A candidate
ZIP is not a complete supplementary bundle while named companion inputs remain
pending. Consult task/PR evidence for actual RED/GREEN, mutation, gate and archive
checks; this entry does not assert CI completion. Preserve unrelated dirty
submodule changes and stop for PR review.

### 2026-09-26: distinguish experimental sources from the submission port

The user requested a comprehensive wording review: incomplete integration into
this submission package must not suggest that original experimental code is
absent or that paper experiments were not performed. The new scope guide and
entry-point links distinguish inspected-but-unported reference code, package
validation, unverified source/result correspondence, genuine feature constraints,
and deliberately out-of-scope pretraining. Supplied protocol originals and
runtime status codes retain their meanings. Historical notes remain dated.

This is a documentation correction, not additional scientific support or score
verification. Anonymous README replacements and private archive policy need the
same wording update before regeneration. Record manual semantic review separately
from automated navigation, parsing and archive checks. Keep identifiers and
private reference evidence outside Git. This continues the existing finalization
PR so the user can review the package together; do not merge automatically.

### 2026-09-26: scope and replication companion received

The previously pending `00_scope_and_replication.md` is now included verbatim
in the [companion index](submission_protocols/README.md), completing delivery
of its eight Markdown documents. This supersedes the scope-document delivery
status above; Extend JSON registries remain pending. The document explicitly
labels its replication profiles as editorial proposals, not historical-run
evidence. No seed assignment, executable recipe or scientific behavior changed.
Update the private archive selection and its anonymous README from seven to
eight documents, verify original-byte delivery and rebuild from the new commit.
Keep the resulting ZIP and review evidence outside Git. Continue the same
finalization PR while open; record gates/CI separately and stop for review.

### 2026-09-26: matching Extend registries included

The user authorized the 81-configuration registries matching the supplied
Markdown. The earlier request to obtain missing registries was based on an
incomplete search: reference and operational copies already existed. The matching
reference registries are now tracked alongside the eight documents. Only dataset
paths change to `${DATA_ROOT}/<dataset-id>`; all non-path values are preserved.
The alternate 82-entry operational revision is not substituted. Source identities
and original paths remain in private evidence, never Git or the submission.

This supersedes earlier pending-registry delivery statements. It does not add
trainers, execute jobs or prove result-to-run correspondence. JSON paths require
explicit loader substitution. Refresh private archive selection, README status
and affected hash-bound approvals, then verify ZIP delivery and anonymity.
