# S9 Public research README

## Purpose and acceptance

A reader can start at a manuscript figure, find its procedure and implementation, and run a small example on a source checkout. The public paper identity and citation must agree with the published project page. The submission exporter must still install the independent reviewer introduction.

## Sources and decisions

D13 calls for a new research entry point. D12 governs the public method paths and D3 requires serial whole-suite validation. The official [project page](https://gatheluck.github.io/10YearVisualSSL/) published the title, seven authors and a 2026 manuscript citation. Its arXiv control was disabled on November 27, 2026. The README links to the site's BibTeX and leaderboard. No public manuscript URL or venue has been inferred.

## Local verification

- Before rewriting the public page, four reader tests and the updated archive fixture failed on the old README. Run them with `python3 -m unittest tests.test_public_readme tests.test_submission_archive.TestSubmissionArchive.test_root_readme_can_reference_license_without_duplicating_notice`.
- The first command in the new README exited zero without console output. It generated 45 JSON configurations for three tasks, five encoders and seeds 0, 1 and 2. It used example paths and did not load data or weights.
- The ZIP fixture built a real archive with the hash-bound reviewer template as its replacement. Its extracted README bytes matched the template. The only external URLs in that text were the PyTorch and PyPI indexes. This exercise does not certify the release archive.
- The project page, ranking page, main workflow link and branch-specific badge returned HTTP 200. The ranking fragment existed and the badge parsed as SVG.
- The focused documentation, archive, method-table, layout and rights checks passed on the S8 worktree revision `fad6b7f` (62 tests). All seven figure rows linked to their own guide section and source or protocol. The operating guide contains the method index and training, configuration and output examples under the unnumbered paths.
- `python3 bin/mutate.py --spec mutations/public-readme.json` detected eight deliberate faults in figure navigation, local links, identity, badge, runnable example and reviewer template isolation.

The first full run on S8 caught a stale mutation that still targeted the method table in the root README. Repointing that mutation to the operating guide keeps the historical ordering guard executable. The three quick-verification mutations and the two existing archive/document mutations also detected their injected faults.

## Delivery and remaining checks

Local CPU time and existing environments only. No new compute charges. The task branch follows the prepared S8 change to avoid publishing an obsolete method index. Repeat the link checks against integrated main after S8 merges. Whole-suite and CI evidence belongs to the exact PR revision. The project owner retains merge and publication decisions.
