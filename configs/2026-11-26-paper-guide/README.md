# S2 Paper reproduction guide

## Purpose and acceptance

Guide a reader from the [submission scope](../../docs/SUBMISSION_SCOPE.md) to the current entry points for the submitted CVPR Figures 1–7 and supplement. Identify the inputs, the level of verification and unresolved discrepancies for each result. This documentation change introduces no scientific code.

## Evidence and changes

The [guide](../../docs/PAPER_REPRODUCTION_GUIDE.md) uses the submitted supplement, the November 25 publication audit and semantic checkpoint evaluation, and the repository's adaptation and representation reports. Their results cover different populations. The guide distinguishes historical score arithmetic, scoring saved predictions, inference from saved weights and training with the public implementation. The original records and inputs remained read-only. The submission scope now links to the guide. A navigation test and its mutation specification check the route.

## Local verification

Run commands from the code root with the downstream Python environment active.

1. RED and GREEN: run `python -m unittest discover -s tests -p test_paper_guide_navigation.py`. Before the documentation change, one test ran and failed at the missing link from the submission scope. After the change, it reached each figure section and opened each linked local file. Eight links with heading fragments were also checked against the destination headings.
2. Run `python bin/mutate.py --spec mutations/paper-guide-navigation.json`. All six mutations were detected, including a missing archive heading, a first command that requires upstream code, and an incorrect correlation count.
3. Run `python -m unittest tests.test_transfer_comparison tests.test_frontier`. All 18 tests passed. These use fixture scores and responses.
4. Run `OMP_NUM_THREADS=1 python -m unittest tests.test_paper_guide_navigation tests.test_section_e_links tests.test_control_comparison tests.test_figure3_lora_configs tests.test_figure3_comparison tests.test_figure5_native_semantics tests.test_representation_analysis_cli tests.test_representation_plot tests.test_representation_nested`. All 38 tests passed with no skips. They exercise local navigation and the component interfaces, including reading saved maps, report generation and rendering.
5. Inspect the guide's executable modules with `--help` in the downstream environment. Each listed option was available. Run `python -m downstream.figure3_lora_configs --paths configs/figure3-lora-paths.example.json --out new-configs` with a new directory. The generator wrote 45 JSON configurations. This operation requires no training data.
6. Check the guide, submission scope, stage record and local PR draft with the project's plain-language checker. Inspect the change for machine paths, run or sample identifiers, and private records before delivery.

## Review of the text delivered in the ZIP

Three new checks failed on the previous text: the archive README lacked the guide's return heading, the first test command selected four probes that require external source, and the Figure 7b count omitted one of the two correlations per task. After correction, `python -m unittest tests.test_paper_guide_navigation tests.test_reviewer_navigation tests.test_reviewer_protocol_index tests.test_section_e_links` passed all nine tests.

A temporary documentation ZIP placed the reviewer template at `code/README.md` alongside the guide and their direct local targets. After extraction, all 73 local links opened and all 18 heading fragments resolved. The two suite commands and three help commands were read from that extracted README. The first suite passed 19 tests without skips in a source-only export. All three help commands printed usage. The second suite passed 23 tests without skips in an export containing the pinned probe file. Both used an existing isolated Python 3.12 environment on macOS.

The delivery policy for the November 24 package predates this guide. Include the new guide and its linked `docs/BASIC5_NATIVE_PATHS.md` and `docs/FRONTIER_EVALUATION.md` when preparing the next package. The archive guide now calls for verifying this against the real ZIP. The temporary ZIP checks the proposed text and paths only. Verification of final file membership and the actual release artifact belongs to the distribution stage.

## Decisions and cost

Reviewed D1 (submitted-paper scope), D2 (originals and COCO metrics), D3 (separate work and validation) and D4 (delivery, cost and submodule constraints) at the start and before reporting. Local checks used existing dependencies. No paid jobs or model calls were needed. No change to the scientific decisions is proposed.

## Delivery and open questions

The guide was developed at b7b4ee9 and prepared for delivery on feb58d5, which includes the local Git fixture changes. D5 covers replacement of the fixtures. D6 permits pinned dependencies for project validation. The S2 validation worktree uses independent local copies of the already verified dependency sources and Git objects. All 20 commit IDs were checked against the inventory. The first complete pre-commit run detected one missing return link to the scope guide among 4,597 tests. After adding it, both scope-navigation tests passed. The PR records the pre-commit and pre-push gate results. Final release ZIP inspection is assigned to the distribution stage. This stage performed no new training, original-checkpoint inference or independent installation test.
