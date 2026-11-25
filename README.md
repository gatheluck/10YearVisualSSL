# Vision Baselines research website

The local website follows the November manuscript, *Revisiting Vision Model Baselines in the Deep Learning Era*. Figure 3 (page 4) reports means and standard deviations across three runs for five checkpoints. LP, AP and LoRA each contribute 25 cells; DPT contributes 10. The two daggers denote shortened schedules. Values are transcribed at their printed precision. Independent reruns belong to a separate source version.

Open `paper-site/index.html` directly in a browser. For local HTTP preview, run the following command from the repository root and open http://127.0.0.1:8765/.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory paper-site
```

The primary leaderboard (leaderboard.html#leaderboard) presents Figure 1. For Figure 3, the selected protocol, task and sort order are shareable via the #unified fragment. Existing parameterized Figure 3 leaderboard links remain supported. Select a score to see its metric, procedure, source and schedule note. The CSV contains the displayed cells in display order.

## Local checks

Install the pinned browser tools with `npm ci` and `npx playwright install chromium`.
Run `npm test` for the branch layout and all website browser tests. Use `CHROME_PATH`
to select an installed Chrome executable.

## Development and release

Keep website source, figure data and browser tests on `codex/paper-website`.
Experiment implementations and training recipes live on
[`main`](https://github.com/gatheluck/10YearVisualSSL/tree/main).
Use a feature branch and a PR into the website branch for edits. With the repository
hooks enabled (`git config core.hooksPath .githooks`), commit, run `npm run verify:push`
and push the verified commit with `git push -u origin HEAD`.

Open a PR from `codex/paper-website` into `gh-pages` to release. PRs targeting
either branch run the site tests. On merge to `gh-pages`, Actions tests the merged
revision and publishes only `paper-site/`. Settings → Pages uses **GitHub Actions**.
The site URL is <https://gatheluck.github.io/10YearVisualSSL/>. Roll back with a
revert PR into `gh-pages`. Use merge commits to preserve release ancestry.

## Historical comparison

Open **History** from the leaderboard or jump to `leaderboard.html#history`. Switch between the printed overall ranking (Figure 1, 109 configurations) and scores by first release (Figure 2, 113 configurations). Year and benchmark filters apply to the chart and table. Selecting a table value opens evaluation details. ASIS uses family-specific protocols; Figure 3 uses the Unified protocols.

Figure 1 preserves the manuscript's printed overall and task ranks, including ties. The mean is the unweighted average of its five task ranks; all 109 means were checked against page 1. Figure 2 uses the v45 plot source whose PDF content stream matches the figure embedded on page 3 of the November manuscript. Its plotted-point counts in benchmark order are 112, 110, 112, 112 and 113. MAGE, JiT and Highway Networks have no plotted COCO detection score in that version; AlayaWorld v1.1 is explicitly excluded from its depth panel. Later figure revisions are separate evidence. The two figures retain their own cohorts; in particular the printed Highway Networks ranks are preserved independently of missing Figure 2 scores. Checkpoint-specific feature descriptions follow manuscript section 4 where available. Unreported per-run recipes are left unspecified.

Run `node tests/paper-history.cjs` alongside the home-page test with the same browser setup. Set `HISTORY_HOME` to a local HTTP preview URL to exercise navigation through the running server.

## Controlled VSSL comparison

Open **ASIS vs CTRL** at `leaderboard.html#controlled`. The five benchmark panels follow Figure 4 on page 5. The source is the v41 comparison table used by the plot whose PDF drawing stream matches the November manuscript. All 32 methods follow the printed legend order. The table retains the plot source’s precision, including near-zero detection scores. CTRL uses the final 300-epoch value. Differences are signed CTRL minus ASIS, with lower values favorable only for NYUv2 RMSE. The scatter uses identical scales on both axes within each benchmark. Expand **Evaluation conditions** for the shared training recipe and interpretation limits from section 5.

Run `node tests/paper-control.cjs` with the browser setup above. The test verifies all 160 pairs, differences and outcomes, navigation, protocol details and narrow-screen layout. Set `CONTROL_HOME` to test an existing local preview.

## Figure 1 on the leaderboard

The leaderboard leads with the task transfer rank matrix and overall standings. Below the standings, expand **Compare rank trajectories** to trace all 109 published configurations through their five benchmark ranks in manuscript order. Each vertical axis uses the same rank scale, with rank 1 at the top. Click a family in the legend to highlight its traces, then click it again to restore all seven. The link below the chart opens the exact ranks in the historical comparison. Study-wide experiment and configuration counts cover a broader population than the Figure 1 cohort.

The seven series colors follow the manuscript family palette. Cyan is reserved for navigation and emphasis. The two V-JEPA lines retain their separate configurations (ViT-H in VSSL and ViT-L in VideoSSL). Run `node tests/paper-overview.cjs` with the browser setup above to check first-screen matrix and standings visibility, the expandable chart and family highlighting.

## Main leaderboard

The Figure 1 heatmap shows all 545 published task ranks with a single scale. Columns keep the manuscript’s mean-rank order and ties. The selected column has a gold pointer, side markers and a gold model label, repeated horizontally above the plot. Hovering another label preserves the selection and the cell colors continue to encode rank. Each column carries its full configuration name beneath the five task ranks. Narrow views scroll the columns and their labels together while task names stay fixed. Select a label or column, or use the configuration selector to read exact values. The standings start with ten rows and support expansion and model-name search. Run `node tests/paper-rankboard.cjs` to check the interactive views and compare the complete standings against the manuscript.

## Model publications and reading paths

Model names in the Figure 1–4 tables link directly to their original source and open in a separate tab. Source links also appear in the selected configuration and the Figure 2/3 score details. The 102 distinct destinations in `sources.js` were opened and their titles checked against the figure sources on November 21, 2026. Shared backbones and checkpoint aliases can refer to the same paper. For Wan2.2 and LTX-2.3 the model authors cite a family paper. FLUX.2 links to Black Forest Labs' official technical release. The benchmark manuscript itself remains local.

The design review considered [Arena](https://arena.ai/leaderboard), [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models), [SWE-bench](https://www.swebench.com/), [ARC Prize](https://arcprize.org/leaderboard), [WorldScore](https://haoyi-duan.github.io/WorldScore/), [MMMU](https://mmmu-benchmark.github.io/) and [Hugging Face's evaluation hub](https://huggingface.co/docs/leaderboards/index). Arena and Artificial Analysis pair a ranking with a task-specific finding. SWE-bench and ARC Prize distinguish evaluation tracks. WorldScore and MMMU carry the reader from summaries to experimental examples and primary sources. Here, the readout story links rank reversals to the slope chart, historical views identify the best measured result in the selected cohort, and the controlled comparison displays one outcome tile per method in legend order. Model pretraining papers describe the models; benchmark scores remain attributed to the study.

Run `node tests/paper-sources.cjs` and `node tests/paper-sections.cjs` alongside the five existing browser checks. They cover source coverage, destinations, release vs paper identity, and benchmark-dependent graphics. The normal regression suite works offline.

### Paper overview and broad transfer

Open `index.html#geometry` for the study rationale and Figure 7's five printed task directions and held-out correlations. Its model coordinates are not reconstructed. The existing `index.html#leaderboard` remains a direct entry to the rankings.

Figure 5 at `leaderboard.html#figure-five` uses the five-model LP/AP source table accompanying the 40-setting figure whose PDF drawing stream matches page 6 (SHA-256 prefix 86120ecbc2da6f5f). It shows the pre-normalization scores. SpaceNet 2 and SUN RGB-D are omitted from data and rankings pending metric confirmation. A displayed first place includes all ties. Source labels identify plotted series, not verified run configurations. Select a task for evaluation details and model publication links. Run `node tests/paper-transfer.cjs` alongside the other browser checks.

The primary navigation has two destinations: Overview and Leaderboard. Individual benchmark comparisons and protocols are linked within the leaderboard. The overview’s Figure 7 section displays the original 1320 × 1520 image embedded on manuscript page 8, with same-tab access to the full resolution and an adjacent accessible numerical summary. The plotted coordinates are preserved as published.

Figure 5 appears inline on the overview and leaderboard. The browser draws the verified chart export as inline SVG with ten profiles and 400 selectable marks. Toggle LP/AP, select a task for its raw scores, or enlarge the chart. The corresponding source PDF drawing stream matches manuscript page 6. The full figure retains its 40 published settings; the linked explorer continues to withhold the two pending task definitions. On narrow screens the original figure pans horizontally.

In the web layout each model has an independent SVG viewport. The five cards reflow with screen width. The common legend stays above the plots, and the always-visible numbered task key selects a spoke and updates all five model scores. Original path geometry, marks and model ordering come from the same verified export.

At 1200 CSS pixels and wider the five plots share one row. The complete numbered task key appears below the plots. Each group swatch uses the same opaque palette color as its corresponding band on all five radars.

Each radar card reports the published first- and second-place counts over all 40 settings in a table, with one row per probe.

Figure 5's selected-task panel shows raw scores with their metric and direction, grouped by model and probe. Figure 1's selected-model ranks include a position tick from first (left) to last (right).

The task picker heads the live score panel. Immediately above the radar plots, the profile selector doubles as the line-and-marker legend and sits alongside the rank key and zoom control. This toolbar stays in view while scrolling the plots on narrow screens.

Overview (`index.html`) introduces the research question, Figure 1 rank matrix, Figure 7 rationale and Figure 5 results. Leaderboard (`leaderboard.html`) opens the interactive standings and links to every comparison, protocol and model source. Legacy index leaderboard fragments preserve their filters; `paper.html` forwards to the overview. Run `node tests/paper-wayfinding.cjs` to check both current and historical routes.

The Overview cover distinguishes the 113 historical configurations in Figure 2 from the 109 complete task rankings in Figure 1. The five core tasks include LoRA on five selected checkpoints. The 40 extended settings use frozen probes. The primary hero action opens the leaderboard. Figure 1 introduces the Basic Five and links readers to the Figure 7 rationale. A compact summary links the historical configurations and extended settings to their results and gives the study-wide total of 2,500+ training and evaluation experiments. The 40 dataset benchmarks cover five selected models and count ImageNet 1% and 10% separately. The arXiv paper entry is marked “Coming soon” until its official abstract URL is available. The left-aligned icon links point to the public code, per-method recipes and weight acquisition guide. Some checkpoints require user-supplied files. The overview automatically plays Figure 2 v46 in release-date order for ImageNet, ADE20K, NYUv2 and SSv2. Each nine-second chapter includes two seconds on the completed plot. Dots and the adjacent family legend use the seven family colors from the Figure 2 v46 palette. The teal staircase tracks the best observed score across families at each release date, grouping same-day releases. This overview aggregates the original figure’s family-specific curves. For each task and viewport, label positions and leader lines are planned using the completed plot. Names appear on their release dates at those fixed positions; later records never displace an earlier visible label. Lower depth RMSE appears higher. COCO awaits protocol review. Reduced motion opens a complete static plot. Run `node tests/paper-hero.cjs` to verify the coverage and responsive view.

All comparison controls share one leaderboard document. Jump links preserve the current filters. The former history, control and extended URLs redirect to their corresponding sections.

The Overview ends with a copyable BibTeX manuscript entry and its LaTeX citation key. When the public bibliographic record is available, update the entry and paper link together. Preserve the citation key for readers who have already used it.
