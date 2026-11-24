# Vision Baselines research website

The local website follows the November manuscript, *Revisiting Vision Model Baselines in the Deep Learning Era*. Figure 3 (page 4) reports means and standard deviations across three runs for five checkpoints. LP, AP and LoRA each contribute 25 cells; DPT contributes 10. The two daggers denote shortened schedules. Values are transcribed at their printed precision. Independent reruns belong to a separate source version.

Open `index.html` directly in a browser. For local HTTP preview, run the following command from the repository root and open http://127.0.0.1:8765/.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory paper-site
```

The primary leaderboard (#leaderboard) presents Figure 1. For Figure 3, the selected protocol, task and sort order are shareable via the #unified fragment. Existing parameterized Figure 3 leaderboard links remain supported. Select a score to see its metric, procedure, source and schedule note. The CSV contains the displayed cells in display order.

## Local checks

With Node.js, Playwright and its Chromium browser installed, run `node tests/paper-site.cjs` from the root. Set `CHROME_PATH` to use a local Chrome executable. The browser test exercises the offline page, all protocol exports, manuscript data digests, details, ordering, shareable state and narrow screens.

## Branch and delivery

Maintain the static site on the dedicated `codex/paper-website` branch. The deployable directory is `paper-site/`; keep the repository root, test fixtures, development notes and review manuscript outside any hosting artifact. All runtime assets use relative paths. A future GitHub Pages workflow can select this branch and publish only that directory after release approval. No publishing workflow is included.

## Historical comparison

Open **History** from the home page or open `history.html` directly. Switch between the printed overall ranking (Figure 1, 109 configurations) and scores by first release (Figure 2, 113 configurations). Year and benchmark filters apply to the chart and table. Selecting a table value opens evaluation details. ASIS uses family-specific protocols; Figure 3 uses the Unified protocols.

Figure 1 preserves the manuscript's printed overall and task ranks, including ties. The mean is the unweighted average of its five task ranks; all 109 means were checked against page 1. Figure 2 uses the v45 plot source whose PDF content stream matches the figure embedded on page 3 of the November manuscript. Its plotted-point counts in benchmark order are 112, 110, 112, 112 and 113. MAGE, JiT and Highway Networks have no plotted COCO detection score in that version; AlayaWorld v1.1 is explicitly excluded from its depth panel. Later figure revisions are separate evidence. The two figures retain their own cohorts; in particular the printed Highway Networks ranks are preserved independently of missing Figure 2 scores. Checkpoint-specific feature descriptions follow manuscript section 4 where available. Unreported per-run recipes are left unspecified.

Run `node tests/paper-history.cjs` alongside the home-page test with the same browser setup. Set `HISTORY_HOME` to a local HTTP preview URL to exercise navigation through the running server.

## Controlled VSSL comparison

Open **ASIS vs CTRL** or `control.html`. The five benchmark panels follow Figure 4 on page 5. The source is the v41 comparison table used by the plot whose PDF drawing stream matches the November manuscript. All 32 methods follow the printed legend order. The table retains the plot source’s precision, including near-zero detection scores. CTRL uses the final 300-epoch value. Differences are signed CTRL minus ASIS, with lower values favorable only for NYUv2 RMSE. The scatter uses identical scales on both axes within each benchmark. Expand **Evaluation conditions** for the shared training recipe and interpretation limits from section 5.

Run `node tests/paper-control.cjs` with the browser setup above. The test verifies all 160 pairs, differences and outcomes, navigation, protocol details and narrow-screen layout. Set `CONTROL_HOME` to test an existing local preview.

## Figure 1 on the opening screen

The opening screen leads with the task transfer rank matrix and overall standings. Below the standings, expand **Compare rank trajectories** to trace all 109 published configurations through their five benchmark ranks in manuscript order. Each vertical axis uses the same rank scale, with rank 1 at the top. Click a family in the legend to highlight its traces, then click it again to restore all seven. The link below the chart opens the exact ranks in the historical comparison. Study-wide experiment and configuration counts cover a broader population than the Figure 1 cohort.

The seven series colors follow the manuscript family palette. Cyan is reserved for navigation and emphasis. The two V-JEPA lines retain their separate configurations (ViT-H in VSSL and ViT-L in VideoSSL). Run `node tests/paper-overview.cjs` with the browser setup above to check first-screen matrix and standings visibility, the expandable chart and family highlighting.

## Main leaderboard

The Figure 1 heatmap shows all 545 published task ranks with a single scale. Columns keep the manuscript’s mean-rank order and ties. The selected column has a gold pointer, side markers and a gold model label, repeated horizontally above the plot. Hovering another label preserves the selection and the cell colors continue to encode rank. Each column carries its full configuration name beneath the five task ranks. Narrow views scroll the columns and their labels together while task names stay fixed. Select a label or column, or use the configuration selector to read exact values. The standings start with ten rows and support expansion and model-name search. Run `node tests/paper-rankboard.cjs` to check the interactive views and compare the complete standings against the manuscript.

## Model publications and reading paths

Model names in the Figure 1–4 tables link directly to their original source and open in a separate tab. Source links also appear in the selected configuration and the Figure 2/3 score details. The 102 distinct destinations in `sources.js` were opened and their titles checked against the figure sources on November 21, 2026. Shared backbones and checkpoint aliases can refer to the same paper. For Wan2.2 and LTX-2.3 the model authors cite a family paper. FLUX.2 links to Black Forest Labs' official technical release. The benchmark manuscript itself remains local.

The design review considered [Arena](https://arena.ai/leaderboard), [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models), [SWE-bench](https://www.swebench.com/), [ARC Prize](https://arcprize.org/leaderboard), [WorldScore](https://haoyi-duan.github.io/WorldScore/), [MMMU](https://mmmu-benchmark.github.io/) and [Hugging Face's evaluation hub](https://huggingface.co/docs/leaderboards/index). Arena and Artificial Analysis pair a ranking with a task-specific finding. SWE-bench and ARC Prize distinguish evaluation tracks. WorldScore and MMMU carry the reader from summaries to experimental examples and primary sources. Here, the readout story links rank reversals to the slope chart, historical views identify the best measured result in the selected cohort, and the controlled comparison displays one outcome tile per method in legend order. Model pretraining papers describe the models; benchmark scores remain attributed to the study.

Run `node tests/paper-sources.cjs` and `node tests/paper-sections.cjs` alongside the five existing browser checks. They cover source coverage, destinations, release vs paper identity, and benchmark-dependent graphics. The normal regression suite works offline.

### Paper overview and broad transfer

Open `paper.html` for the study rationale and Figure 7's five printed task directions and held-out correlations. Its model coordinates are not reconstructed. The existing `index.html#leaderboard` remains a direct entry to the rankings.

Figure 5 at `extended.html` uses the five-model LP/AP source table accompanying the 40-setting figure whose PDF drawing stream matches page 6 (SHA-256 prefix 86120ecbc2da6f5f). It shows the pre-normalization scores. SpaceNet 2 and SUN RGB-D are omitted from data and rankings pending metric confirmation. A displayed first place includes all ties. Source labels identify plotted series, not verified run configurations. Select a task for evaluation details and model publication links. Run `node tests/paper-transfer.cjs` alongside the other browser checks.

The primary navigation has three destinations: Overview, Leaderboard and Paper. Individual benchmark comparisons and protocols are linked within the leaderboard. The Paper page's Figure 7 section displays the original 1320 × 1520 image embedded on manuscript page 8, with same-tab access to the full resolution and an adjacent accessible numerical summary. The plotted coordinates are preserved as published.

Figure 5 appears inline on the overview/leaderboard, paper and broad-transfer pages. The browser draws the verified chart export as inline SVG with ten profiles and 400 selectable marks. Toggle LP/AP, select a task for its raw scores, or enlarge the chart. The corresponding source PDF drawing stream matches manuscript page 6. The full figure retains its 40 published settings; the linked explorer continues to withhold the two pending task definitions. On narrow screens the original figure pans horizontally.

In the web layout each model has an independent SVG viewport. The five cards reflow with screen width. The common legend stays above the plots, and the always-visible numbered task key selects a spoke and updates all five model scores. Original path geometry, marks and model ordering come from the same verified export.

At 1200 CSS pixels and wider the five plots share one row. The complete numbered task key appears below the plots. Each group swatch uses the same opaque palette color as its corresponding band on all five radars.

Each radar card reports the published first- and second-place counts over all 40 settings in a table, with one row per probe.
