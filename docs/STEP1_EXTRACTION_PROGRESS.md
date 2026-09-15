# Step-1 extraction progress

## Objective
Resolve actual Step-1 checkpoints on ABCI, record immutable weight identities and acquisition paths, and extract ImageNet-val features reproducibly without modifying original experiments.

## 2026-09-16: start
- Baseline: c3cec79. Existing driver has 51 providers; recorded sweep covers 13 download-backed targets.
- Private evidence: Capture docs/STEP1_WEIGHT_SURVEY_20260916.md. Candidate references are not final selections.
- Implement using failing tests first. Reuse fetch-weights for checksum verification; distinguish public download from user-supplied weights.
- First compatibility pilot: SimCLR v1 ResNet-50. Its original evaluator unwraps state_dict and exports encoder features. No claim of successful real-weight extraction yet.
- Keep full private filesystem paths and raw original results in the private repository.
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
