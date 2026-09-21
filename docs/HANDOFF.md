# Cross-Mac project handoff

Snapshot: 2026-09-19, after PR #183 merged. This document is a recovery entry
point, not a live dashboard. Read [agent instructions](../AGENTS.md) and
[repository rules](../CLAUDE.md), then refresh the actual Git/PR/cluster state.
The older [Basic5 crop handoff](HANDOFF_BASIC5_B.md) is historical; its branch,
delivery status and permission assumptions do not describe today's workflow.

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
from images. Gate only that repository assertion with the existing checkout
predicate. A fresh-process regression removes git from PATH and makes workflows
absent, runs the method smoke, and requires at least nine executed tests. It
first reproduced the exact `KeyError`, then passed; skipping model coverage is
not the fix. The full local gate and image CI must be refreshed after this change.
