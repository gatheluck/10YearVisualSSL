# Vision Baselines research website

The local website follows the November manuscript, *Revisiting Vision Model Baselines in the Deep Learning Era*. Figure 3 (page 4) reports means and standard deviations across three runs for five checkpoints. LP, AP and LoRA each contribute 25 cells; DPT contributes 10. The two daggers denote shortened schedules. Values are transcribed at their printed precision. Independent reruns belong to a separate source version.

Open `index.html` directly in a browser. For local HTTP preview, run the following command from the repository root and open http://127.0.0.1:8765/.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory paper-site
```

The selected protocol, task and sort order are shareable via the leaderboard fragment. Select a score to see its metric, procedure, source and schedule note. The CSV contains the displayed cells in display order.

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
