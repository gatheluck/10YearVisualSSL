# S3 independent checkout verification

## Acceptance criteria recorded before execution

Export commit `b7b4ee93110ab5278c3adc0e26965b136dab877a` with `git archive` to a new temporary directory. Verify that the export contains neither Git metadata nor files from pinned submodules. Create a fresh Python 3.12 environment without system site packages. Record the host platform, dependency source and installation outcome separately from runtime results.

Run the four README Quick verification test modules together and each of the three help commands. Record exit status, errors and skips. A skip does not exercise a component. Run the public Frontier evaluation fixture through its CLI and verify the report and refusal cases. Run the representation plotting fixture through its CLI and check the PNG, SVG and JSON output. Identify additional inputs needed to apply these routes to paper data. Record any failure even if a narrower command succeeds.

## Hypothesis and limits

Saved predictions should support evaluation without weights or a service call. A recorded representation analysis and its family labels should suffice to draw a diagnostic plot. Test whether the advertised quick checks need code outside this checkout. Source and dependency checks use local resources only. The Linux lock, historical scores and full training require separate evidence.

## Coordination and cost

The initial coordination record ended at D5 (replace Git fixtures that fetch submodules). This stage uses local CPU time and existing package caches with no new charges.

## Observations on macOS arm64

The source export contained no `.git` directory and no regular files at any of the 14 gitlink paths. Python 3.12.16 reported `include-system-site-packages = false` in the new environment. The first wheel-only, offline install failed because the cache lacked `lvis==0.5.3`. For the installation check only, the package source was expanded to public PyPI wheels. No dataset, model, private checkout or submodule was retrieved.

From the export root, with the new environment active, the equivalent install command is `uv pip install --index-url https://pypi.org/simple --only-binary :all: -r downstream/requirements.txt`. The measured run selected the environment with `--python` and used `--link-mode clone`. It installed 58 packages, including torch 2.15.0, torchvision 0.29.2, NumPy 2.5.5, SciPy 1.18.1 and Matplotlib 3.11.2. `python -m pip check` exited 0.

| Command from the export root | Local result |
| --- | --- |
| `python -m unittest tests.test_figure3_comparison tests.test_cvpr_extended_coverage tests.test_transfer_comparison tests.test_dpt_probe` | Exit 1. Of 23 tests, 19 passed and four DPT tests errored during probe construction. No skips. |
| `python -m unittest tests.test_figure3_comparison tests.test_cvpr_extended_coverage tests.test_transfer_comparison tests.test_dpt_probe.DenseProbe.test_missing_submodule_reports_actionable_error` | Exit 0. All 19 ran and passed with an empty user home. This measures the portion available from the export. |
| `python -m downstream.revised_tasks --help` | Exit 0, displayed config and output options. |
| `python -m downstream.revised_reporting --help` | Exit 0, displayed config and output options. |
| `python -m downstream.representation_analysis --help` | Exit 0, displayed config and output options. |
| `python -m unittest -v tests.test_frontier.FrontierEvaluation.test_cli_writes_report_without_overwriting_and_rejects_raw_bad_json` | Exit 0 before dependency installation. The CLI scored 500 synthetic ImageNet class responses in five JSON files, wrote the expected score and input hashes, refused an overwrite and rejected malformed responses. |
| `python -m unittest -v tests.test_extended_native_images.NativeImages.test_all_clue_variants_keep_test_and_submission_disclosure` | Exit 0. Six generated 11 by 9 RGB PNGs exercised split selection and byte preservation for five CLUE variants. |
| `python -m unittest -v tests.test_representation_plot` | Exit 0, four passed. Four synthetic model rows with two family labels and five task directions exercised the plotting CLI. Checks covered the scene coordinates, PNG signature, SVG labels, JSON contents and refusal of inconsistent inputs. |

## Inputs required beyond the checkout

All four Quick verification errors came from the same missing file, `third_party/geosubprobe/evals/models/probes.py`. The export contains only the parent repository's files. Its Git tree pins geosubprobe at `b2425a724050f5fb7d809f9e2d416e1715072dbe`. The errors affected dense output and update comparison, hook cleanup, geometry checks and state round-trip. The test for the missing dependency passed.

The Frontier scorer needs a labeled manifest and five aligned response files for an actual run. Plotting needs a completed analysis JSON and matching family labels. Neither fixture loads model weights. Recomputing the analysis from images and running a trained encoder additionally require the corresponding data, checkpoint and run configuration. See the [representation analysis inputs](../../docs/REPRESENTATION_ANALYSIS.md).

## README correction and delivery status

The owner subsequently assigned the root README correction to this stage. Before editing the README, the new navigation test failed on the original first command because it selected the whole DPT module. Executing that command directly from the README in the source export reproduced the four missing-source errors.

After editing, the same check extracted the first unittest invocation and the three help invocations from the revised README and ran them with the isolated Python environment in the export. The 19 selected tests passed with no skips and each help command exited 0 and printed usage. A repeat of the macOS pip installation command found all requirements satisfied in the environment originally installed with uv.

Under decision D6, the coordinator prepared all 20 pinned dependencies, including nested dependencies, and verified their commits. The coordinator reported that the second README command passed all 23 tests in the prepared macOS environment in 4.468 seconds with no skips.

A second S3 run used a fresh export and added only `evals/models/probes.py` from the exact geosubprobe commit recorded above, read from S0's local Git objects. The full unittest command extracted from the README passed all 23 tests in 2.734 seconds with no skips. This isolates the extra source needed for these four DPT checks. The README gives the path-specific setup command for a Git clone. Its flags were checked against the [official Git manual](https://git-scm.com/docs/git-submodule); S3 did not perform a network fetch.

The following focused checks passed:

- `python3 -m unittest tests.test_reviewer_navigation tests.test_reviewer_protocol_index tests.test_section_e_links` (six tests).
- `python3 bin/mutate.py --spec mutations/quick-verification.json` (three mutations: requiring upstream code in the first command, dropping DPT coverage from the full command and selecting an upstream branch in place of the pinned commit).

For full validation, the S3 branch incorporates the fixture changes merged in PR #287. All 20 Git revisions in its dedicated worktree matched the parent repositories' pins before the whole-suite checks. The source-only measurements above used a separate export. Whole-suite outcomes and CI status are recorded on the pull request.

## Unverified

Linux x86_64 lock installation, a fresh network checkout or published release archive, full datasets, historical response files, encoder inference, GPU execution, numerical reproduction of the paper and visual inspection of plot styling were outside this run.
