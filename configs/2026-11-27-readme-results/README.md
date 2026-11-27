# S11 Figure 3 and key results in the public README

## Purpose and decisions

Show a representative Figure 3 comparison and three findings before the figure index. Decisions D14 (manuscript terminology) and D15 (README results) apply. Decision D3 calls for a dedicated branch, behavioral checks before implementation, and serial whole-suite validation.

## Evidence

- Figure 3 and Section 6 identify the five Unified checkpoints and report classification top-1 accuracy (%). The chart uses only ImageNet-1K and SSv2, with linear probing (LP) and attentive probing (AP).
- All 20 chart means agree at two decimals across the supplied main manuscript's Figure 3, adopted Supplement Tables 22 and 23, and recomputation from the November 25 audit. The audit has three seed values per cell, with 17 cells from native result records and three from contemporaneous archival records. The source tables show mean and standard deviation. The chart shows mean only, as stated in the caption.
- Figure 2 and Sections 3–4 describe 113 historical ASIS configurations under family-specific protocols. For the 32 selected self-supervised methods, the ImageNet-1K endpoints round to 2.83% (VAE) and 84.47% (DINOv3 ViT-B/16), checked against Figure 2 and Supplement Table 10.
- Across the five Figure 3 checkpoints, ranking the reported means gives four reversed pairs out of ten on SSv2 and one on ImageNet-1K. DINOv3-7B and SigLIP 2-G give the example in Section 6.
- Figure 7 and Section 7 report pooled held-out rank correlations across seven model families. With patch and image agreement instead of image agreement alone, Spearman correlation changes from 0.19 to 0.75 for ADE20K and from 0.01 to 0.67 for NYUv2.

The [chart](../../docs/figures/figure3-classification.svg) is a new drawing generated from the [reported means](../../docs/figures/figure3-classification.csv). The manuscript, original figure, workbook and private run records are outside the change.

## Local verification

- The four new tests failed on the previous README because the chart, data, findings and regeneration instructions were absent. The same tests then passed. Run `python3 -m unittest -v tests.test_readme_results`.
- The focused results, public README, manuscript terminology, paper guide and anonymous archive checks passed (18 tests). The language and tool coverage checks passed (22 tests). The stage record, method index and Appendix E links checks passed (14 tests).
- The new mutation suite caught all seven changes to the cited findings, data, chart and caption. The existing public README and paper guide suites caught all 14 changes, including loss of anonymous reviewer isolation.
- Chrome rendered the README and embedded SVG with GitHub-like margins and image sizing at 320, 390 and 1280 CSS pixels. The chart rendered at 288, 358 and 420 CSS pixels, respectively. Visual checks covered all labels, scores, the outlined and solid legend, captions, and the transition to the figure index. GitHub-hosted rendering is pending PR delivery.

## Integration

The first GitHub check showed main at the S9 merge `7329a77` and no S10 PR. The branch uses the S10 local commit `a9db23b` to preserve manuscript terminology. Before opening this PR, check the current S10 branch and main again and compare the S11-only diff.

The initial S11 worktree lacked the pinned third-party checkouts. The license inventory's Git-object check reproduced the missing dependency failures (17 subtest failures and three errors). After S10's whole-suite run finished, S11 used independent copies of the already verified S2 Git objects. All 20 checkout revisions and the links in their parent indexes matched the inventory. All seven license inventory tests passed with those checkouts. The setup used no network retrieval.

The first populated pre-commit run executed 4,632 tests and reported one failure and 2,341 skips. The new renderer was missing from the operating guide's tool layout. The operating guide now lists it, and the layout check passes.
