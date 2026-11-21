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
