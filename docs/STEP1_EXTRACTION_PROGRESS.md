# Step-1 extraction progress

## Objective
Resolve actual Step-1 checkpoints on ABCI, record immutable weight identities and acquisition paths, and extract ImageNet-val features reproducibly without modifying original experiments.

## 2026-09-16: start
- Baseline: c3cec79. Existing driver has 51 providers; recorded sweep covers 13 download-backed targets.
- Private evidence: Capture docs/STEP1_WEIGHT_SURVEY_20260916.md. Candidate references are not final selections.
- Implement using failing tests first. Reuse fetch-weights for checksum verification; distinguish public download from user-supplied weights.
- First compatibility pilot: SimCLR v1 ResNet-50. Its original evaluator unwraps state_dict and exports encoder features. No claim of successful real-weight extraction yet.
- Keep full private filesystem paths and raw original results outside Git in local execution state.
- Pending: selected checkpoint hashes, export/load parity, isolated ABCI execution environment and dataset access, batch smoke, full val extraction.

## Reservation-only execution constraint (2026-09-16)
User requires all jobs to use the already reserved nodes through the designated submission method, without additional point consumption. No jobs have been submitted. User has supplied the reserved-node submission procedure. Concrete group/reservation/account identifiers are local-only and must not be committed. Never fall back to an ordinary queue. Implementation and read-only inspection may continue.

Private execution settings and raw evidence are persisted under `$HOME/.local/state/10YearVisualSSL/abci/` (outside the repository). Read them on resumption; do not repeat acquisition questions if already answered there.

## Implementation and preflight
- SimCLR native checkpoint hash recorded in method provenance; no public URL verified, so user-supplied acquisition is explicit.
- Native wrapper/prefix mapping uses existing adapter selection/loading; collisions are refused.
- TDD: fetch tests RED (3 failures) then GREEN (7 tests). Mapping tests RED (5 missing-tool errors) then GREEN. Remote tensor round-trip: 6 tests passed, no skips. Mutation checks: 3/3 killed.
- Reserved queue verified enabled/started. Pilot output will go to a separate workspace; original filesystem remains read-only.

## SimCLR real-weight pilot completed
- Reserved-node job finished with exit status 0; no standard queue was used.
- Original source filesystem was mounted read-only; only the separate task workspace was writable.
- Native checkpoint SHA-256: e945c6bd144640f174e90ef924890fd47280f522a10d20ee01054be234d83b13.
- Export SHA-256 in this run: 928e17611d41cbfb8f667dab9fbab657c98b850e40ed6d44af3fda66a4ad52b2.
- Native captured model versus exported port: four real val images, maximum absolute feature error 0.0. This is a four-image parity check, not a numerical proof for all inputs.
- Full val extraction: (50000, 2048) float32, all finite; 1000 classes with 50 images each, labels sorted; L2 norms 0.9999998212 to 1.0000001192.
- Class-order SHA-256: 70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15.
- Runtime: Python 3.10.20, torch 2.5.1+cu124, NVIDIA H200. This is the existing inspection/extraction runtime, not the port's current dependency lock; matching-lock reproduction remains unverified.
- Weights are lab-trained, with no verified public URL. The new local acquisition path verifies an independently obtained copy. Redistribution has not been arranged.
- Private job IDs, paths, full logs and execution settings are stored outside the repository; see the local-state pointer in CLAUDE.md.
- Other Step-1 targets remain pending selection and compatibility verification. Do not count the 30-method preliminary reference survey as completed exports.

## Whole-suite validation
- Initial full base suite failed: missing submodules, missing new-tool README entry and Git-free scans including the initial in-repository private state.
- Private state moved outside the repository; README entry added. Platform/method-name/layout checks: 30 passed. Pinned submodules were initialized. A subsequent full suite reached 3276 tests with one documentation-guard failure (1370 dependency-gated skips).

## Output integrity
- features.npy SHA-256: ad567a07f2d6867cd8cb04f43539ed137b67cc928e9df7c834679173a4c0078a
- labels.npy SHA-256: 186361171a139f8617d3ce725b06a01cbc9ae9eac739098b93dc6261ee7aacbe
- Export tool source SHA-256 used in the job: 409b558986a007ae563e859e5a79e4fb8f5d4c4f7fe3905435a04fcedac43529
- Extraction driver source SHA-256 used in the job: e4d2ff968195a11572dfaaca4aa3daf9e156dfe3418fa2e94b5b84a676438eaa
- Final local acquisition tests: 8 passed; native-weight mutations: 4/4 killed.

- Documentation guard: added failing controls for numeric artifact names, invalid suffix truncation and comment-wrapped arguments; fixed the parser without suppressing missing-section failures. Its 8 tests pass.
- Native-weight mutation checks now cover 6 faults; all 6 were detected.
- The pre-commit hook enforces a fresh whole-suite pass before the commit is created. Final gate results are recorded in the PR and private local execution log.

## Seven-model batch completed (2026-09-16)

Child branch: `codex/step1-multi-model-extraction`, based on PR #170 commit
`3152f9970bb30290c254f5f3d0bb9a79f730893f`. The next PR is deferred until
#170 merges and the child changes are rebased onto the resulting main branch.

| Model | Selected training endpoint | ImageNet val features |
|---|---:|---|
| InstDisc | 200 | 50,000 x 2,048 |
| MoCo v1 | 200 | 50,000 x 2,048 |
| MoCo v2 | 200 | 50,000 x 2,048 |
| SimCLR v2 | 800 | 50,000 x 2,048 |
| SimSiam | 100 | 50,000 x 2,048 |
| SwAV | 200 | 50,000 x 2,048 |
| SeLa | 400 | 50,000 x 2,048 |

The actual Step-1 evaluation records reference each selected checkpoint.
Checkpoint configuration confirms the scheduled endpoint and backbone shape;
selection does not rank validation accuracy. All seven exact source hashes are
now in per-method provenance, with explicit user-supplied acquisition.

For each model, captured reference model/evaluation source files were checked
against the live original by hash. A strict full-native model load was followed
by comparison with the actual feature provider on four real val images. Raw
features matched exactly (maximum absolute error 0.0) for all seven models.
The providers' own preprocessing was exercised, including SwAV's captured red
channel standard deviation of 0.228 and SeLa's ResNetV2 backbone.

All seven full-val arrays are finite float32 with shape (50000, 2048), sorted
labels, 1000 classes of 50 images and unit L2 norms within 1e-6. Label hashes
match the preceding SimCLR pilot. Source, export and output hashes are recorded
in `STEP1_BATCH_RESULTS_20260916.json`. Four-image parity does not prove equality
on every possible input. Execution used the existing torch 2.5.1+cu124 / H200
runtime; matching the repository's current dependency locks remains unverified.
Original files were mounted read-only; two reserved-node jobs wrote only to
separate workspaces. Private IDs, paths and raw records remain outside Git.

TDD: batch tests first failed before implementation; a further failing test
caught interrupted work being labelled successful. Batch tests now cover
preflight, hash format, mapping, per-model failure, launch errors, output
preservation, symlink escape, interruption and CLI refusal. Nine mutation
controls were all detected. Seven identity tests failed before their provenance
records were added. Initial whole-suite validation caught incorrect names for
method-specific test files; those were renamed to the established convention,
without weakening the guard. Final whole-suite results are kept in the local
execution log and the eventual PR.

### Remaining compatibility work

- BYOL: the selected native checkpoint is readable and hashed. Its recorded
  official evaluation uses bicubic interpolation; the current provider uses
  bilinear. Do not claim equivalent Step-1 extraction without resolving that
  protocol difference and testing it.
- Barlow Twins: the evaluated `resnet50.pth` is a bare torchvision backbone,
  whereas the adapter expects prefixed sequential-backbone keys. It contains
  no training config/epoch; the result record alone is insufficient to assert
  its pretraining endpoint. Mapping and endpoint provenance remain pending.
- The remaining methods require their own selection and compatibility audit;
  readable checkpoint references are not completed extractions.


## Three additional models completed (2026-09-16)

Branch `codex/step1-remaining-models` starts from PR #171 tip `6964d94`.
This groups three models in one delivery rather than one PR per model.

| Model | Native checkpoint epoch (zero-based) | Full val output |
|---|---:|---|
| Colorization | 299 | 50,000 x 512 |
| Visual CPC 2018 | 199 | 50,000 x 1,024 |
| BYOL | 999 | 50,000 x 2,048 |

All three native models loaded strictly from the checkpoints referenced by
the recorded Step-1 evaluations. Each port matched four real validation images
exactly on an H200 GPU (maximum raw feature error 0.0), after checking reference
source hashes against the live original. All full arrays are finite float32,
L2-normalized within 1e-6, with sorted labels and 1,000 classes of 50 samples.
Their labels.npy hashes match the preceding eight new extractions. Outputs
remain in separate local cluster workspaces; no weights or arrays are committed.
Hashes and dimensions are in `STEP1_NEXT_RESULTS_20260916.json`.

The BYOL mismatch noted above is now resolved: bicubic interpolation and ceiling
resize are tested against the native evaluation. Colorization needed exact
module-name mapping and native evaluation arithmetic precision. CPC's actual
preprocessing already matched; its misleading metadata description was corrected.
These exact checkpoint bytes remain user-supplied, with no verified public URL.

TDD evidence: mapping, checkpoint-identity and native-pixel tests were written
before implementation and observed failing. Remote tensor/pixel suite: 24 tests,
no skips, exit 0. Nine new mutation controls and six inherited controls were all
detected. Full local base suite: 3,308 tests, exit 0, 1,372 dependency-gated skips.
An earlier full run caught platform vocabulary in provenance and an ambiguous
inherited mutation anchor; both were fixed without weakening the checks.
The first GPU job stopped at reference verification because macOS archive metadata
files were transferred; the corrected transfer used a new workspace and the
retry completed with exit 0. Original files stayed read-only in both attempts.
The GPU runtime differs from current dependency locks; four-image parity does
not prove equality for all possible inputs. PR CI is separate validation.

### Collection count, distinct from native identity audit

The user already holds features for **13** methods: DINOv2, Franca, AIMv2,
BEiTv2, CAE, Cosmos3 Super, data2vec2, EVA02, SigLIP, VAR, VideoMAE, V-JEPA2 AC
and V-JEPA2. These count as extracted; missing local checkpoint identity is not
missing feature data. Together with the preceding eight and these three new
extractions, the collection is **24 of 51 providers**, leaving **27 uncollected**.
Do not rerun the user's existing thirteen merely because their native identity
has not been audited. Cross-checking sample identity against that older external
collection and creating the consolidated delivery manifest remain separate work.

Rotation Prediction is still blocked by a representation mismatch (native
conv5 global average: 256 dimensions; current port encoder: 4,096 dimensions).
Barlow Twins' full checkpoint requires a reviewed safe-loading solution for an
optimizer callable; its bare backbone's endpoint still needs proof. Neither is
counted as extracted here.

## Four transformer protocols (next grouped branch)

Branch `codex/step1-next-model-audit` starts at PR #172 tip `3f43397`.
MoCo v3, DINO, MAE and SimMIM now have checkpoint-specific export profiles;
see `STEP1_WEIGHTS.md`. Preserve `encoder.pt` together with `export.json`.
Ordinary encoder files keep their existing provider defaults. Exact checkpoint
bytes are pinned as user-supplied artifacts; no public download URL is verified.

The shared sidecar/CLI tests were observed RED before implementation, followed
by GREEN. The resize and DINO concatenation tests also failed before their
helpers existed, and the four provenance tests failed before records were
added. The initial provider rejection tests additionally exposed heavy model
construction/import collisions when the profile was ignored; they were improved
to require rejection before any model import. Positive tests now exercise the
actual provider entry points with tiny backbones, including default preservation,
MAE pool selection and DINO concatenation. Eighteen deliberate mutations were
all detected with passing baselines; these include provider wiring, CLS token
selection, resize overrides, sidecar identity and export/batch forwarding.
The tensor-enabled focused suite passed 48 tests without skips. The first full
base run found unregistered mutation measurements; the results are now recorded
in their specifications rather than weakening the completeness check.

GPU parity and full-val extraction are tracked separately from code support.
The reserved-node job uses a read-only mount for original files and an isolated
writable output directory. No weights, feature arrays, account identifiers,
private paths or raw execution logs belong in this repository.

### All four extractions completed (2026-09-16)

| Model | Native checkpoint epoch (zero-based) | Full val output |
|---|---:|---|
| MoCo v3 | 299 | 50,000 x 768 |
| DINO | 99 | 50,000 x 1,536 |
| MAE | 1,599 | 50,000 x 1,024 |
| SimMIM | 799 | 50,000 x 1,024 |

All four strictly loaded native checkpoints passed independent four-image H200
comparisons with maximum raw feature error **0.0**. Captured reference source
hashes matched the live originals. The reserved job exited 0. Each full output
is finite float32, L2-normalized within 1e-6, and has sorted labels with exactly
50 samples per class across 1,000 classes. All label-file hashes match the
preceding eleven new extractions. Output hashes and parity metadata are in
`STEP1_VIT_RESULTS_20260916.json`.

Collection progress is now **28/51 complete, 23 uncollected**, including all
thirteen user-confirmed existing methods. Do not repeat those extractions.
Data remains in the isolated cluster workspace pending the visualization
team's destination. The consolidated manifest and alignment with the user's
older collection still require separate checks. This GPU runtime uses torch
2.5.1+cu124 and differs from the repository's dependency locks; four-image
parity does not prove equality for every possible input. Parent PR #172 and
its CI remain separate from this grouped child branch.

Final base suite after registering mutation evidence: 3,330 tests, exit 0,
with 1,382 dependency-gated skips. Those skips are not GPU test passes; the
separate focused tensor suite and the full native/GPU extraction checks above
provide the validation for these four changes.

## Seven more native methods completed (2026-09-16)

Child branch `codex/step1-broad-remaining-models` starts from PR #173 tip
`c7c97e5`. The seven candidates are now implemented and fully extracted.

| Method | Stored checkpoint epoch | Full ImageNet val output |
|---|---:|---|
| VAE | 300 | 50,000 x 50 |
| DeepCluster | 499 | 50,000 x 4,096 |
| Split-Brain | 99 | 50,000 x 512 |
| BEiT | 299 | 50,000 x 768 |
| iBOT | 799 | 50,000 x 1,536 |
| I-JEPA | 299 | 50,000 x 1,280 |
| NEPA | 1599 | 50,000 x 768 |

Each checkpoint was safely loaded, hashed, exported and strictly checked by its
adapter. Captured reference source hashes matched the live original. All seven
providers matched native raw features exactly on four real val images on H200.
Every full array passed shape, float32, finite-value, unit-L2 and label checks:
50,000 samples, sorted labels, 1,000 classes with 50 images each. Label hashes
match all preceding new extractions. Evidence is in
`STEP1_BROAD_RESULTS_20260916.json`.

The first GPU comparison caught a real DeepCluster defect: the adapter discarded
Sobel tensors although native initialization changes those frozen values. Tests
now require exact preservation and refusal of missing frontend state. Existing
exports without those tensors must be regenerated. Split-Brain needed explicit
native float32 Lab arithmetic; its ordinary training conversion remains the same.
BEiT needed bilinear resizing, VAE needed latent size 50 and image size 224, and
iBOT needed the last four teacher blocks. I-JEPA and NEPA use explicitly selected
target/EMA states. The initial VAE verification harness passed an unsupported
constructor argument; it was corrected. The extraction environment lacked iBOT's
declared tensorboard dependency; it was added in a separate environment. Only
the four unsuccessful methods were retried. Their final GPU job exited 0.
BEiT's initially stale text description was corrected to match its actual
bilinear override; feature arrays were not changed.

TDD: shared config/prefix, batch forwarding, seven profile/identity, Sobel and
native-pixel tests were observed failing before the corresponding implementation.
The tensor-enabled focused suite passed 63 tests without skips, with an additional
Lab-mode rejection test verified by mutation. DeepCluster/Split-Brain regression:
96 tests, 12 dependency-gated skips, exit 0. Eight generic mutations and fifteen
method-profile mutations were detected, including the real export CLI, input
config immutability, actual Sobel tensors, native pixel precision and provider
wiring. Nine inherited mapping/pixel and nine prior transformer-profile mutations
also passed. An initial base-suite run failed solely because new mutation results
had not yet been recorded; the completeness guard was retained.

Collection: **35 of 51 extracted, 16 still uncollected**, including the user's
thirteen previously completed methods. The original code, weights and environments
remained read-only. Only isolated workspaces and the designated delivery directory
are writable. All jobs used the required reservation; no ordinary queue fallback.
These lab checkpoint bytes have no verified public download URL. The runtime
still differs from the repository's current locks, so matching-lock reproduction
and PR CI remain separate validation. Raw logs and concrete execution identifiers
remain outside Git. The grouped child PR follows the parent-merge/rebase workflow.

Final local base suite: **3,359 tests, exit 0, 1,400 dependency-gated skips**.
The shared visualization collection now contains **22** verified methods; its
51-record manifest preserves 13 extracted elsewhere and 16 not yet extracted.
All four files of each new method were hash-checked before/after copying, and
all previously delivered files retained their hashes. Group read/traverse access
was checked for every delivered path. The seven-method branch is grouped for one
PR after #173 merges; private execution logs record operational details.


## Five additional evaluated checkpoints (2026-09-16)

Child branch `codex/step1-final-readable-models` starts from PR #174 tip
`886266b`. A fresh read-only audit found eight readable candidate files among
16 uncollected methods. All eight safely deserialized; seven contain backbone
weights. The inspected AIM file contains a probe only, not the backbone.
Readable tensors alone do not prove that a provider can load them.

Five methods now passed strict native/export loading, source identity checks,
and exact raw-feature parity on four real validation images on H200. Each then
completed 50,000-image extraction with finite float32 unit-L2 features and sorted
int64 labels (1,000 classes of 50 images). All five label hashes match the
previous outputs. Evidence: `STEP1_REMAINING_RESULTS_20260916.json`.

| Method | Evaluated variant | Dimensions |
|---|---|---:|
| Context Prediction | Official-style final checkpoint, global step 1,000,000 | 4096 |
| Jigsaw | Full-image CFN spatial pool4, stored epoch 275 | 8192 |
| Rotation | Official conv5 before pool5, global average, stored epoch 49 | 256 |
| Jigsaw++ | Knowledge-transfer AlexNet, stored epoch 89 | 9216 |
| Barlow Twins | Evaluated bare ResNet-50, pretraining epoch unverified | 2048 |

Context Prediction uses the recorded official-style evaluation, not the separate
paper-target epoch-299 checkpoint. Barlow's result directory does not independently
establish the backbone's pretraining epoch. Neither variant was selected by ranking
validation accuracy. All five exact files are user-supplied; no public URLs were
verified. Existing port defaults remain available without the native sidecar.

TDD measured failures first: six profile tests, two state-mapping tests and five
checkpoint-identity tests. Profile checks exercise actual transformed pixels and
feature tensors, including rejecting exact-name decoys. Sixteen mutation controls
were detected with passing unmodified baselines. Whole-suite results are recorded
in private execution logs and the eventual grouped PR. Runtime locks remain a
separate CI validation; four-image parity is not a proof for every input.

The collection now has 40/51 extracted methods, including all 13 user-owned older
outputs. Eleven remain uncollected: Context Encoder and DINOv3 require native
architecture compatibility; AIM's backbone location still needs resolution;
CMC, PIRL, MSN, V-JEPA and LeJEPA have inaccessible known checkpoint paths;
CLIP, ImageGPT and SAM3 still need actual checkpoint/evaluation mapping. These
are current audit gaps, not assertions that extraction is impossible.
