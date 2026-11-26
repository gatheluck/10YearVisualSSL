const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href);
    const panel = page.locator(".rank-preview");
    assert.equal(await panel.getByRole("heading", { name: "Vision Model Rankings" }).count(), 1);
    assert.deepEqual(await page.locator("main > [id]").evaluateAll(nodes => nodes.map(n => n.id)),
      ["rankings", "figure-five", "geometry", "citation"]);
    const view = panel.getByLabel("Rank by");
    assert.equal(await view.inputValue(), "overall");
    assert.equal(await panel.getByLabel("Show ranks").count(), 0);
    const { ranks, scores } = await page.evaluate(() => ({ ranks: historicalRanks, scores: historicalScores }));
    const rows = panel.locator("tbody tr");
    for (const [index, value] of ["overall", "in1k", "coco", "ade20k", "nyuv2", "ssv2"].entries()) {
      await view.selectOption(value);
      const rank = r => index === 0 ? r.place : r.ranks[index - 1];
      const expected = ranks.filter(r => rank(r) <= 10).sort((a,b) => rank(a)-rank(b));
      assert.deepEqual(await rows.evaluateAll(nodes => nodes.map(row =>
        [...row.cells].map(cell => cell.textContent))), expected.map(r => {
        const score = r.name === "V-JEPA (ViT-H)" ? scores.find(s => s.id === 29)
          : r.name === "V-JEPA (ViT-L)" ? scores.find(s => s.id === 54)
            : scores.find(s => s.name === r.name);
        return [r.name + score.family, "#" + r.place,
          (r.ranks.reduce((a,b)=>a+b,0)/5).toFixed(1),
          ...r.ranks.map((place, task) => (score.scores[task] ?? "—") + "Rank " + place)];
      }));
      assert.equal(await rows.locator('a[href^="https://"]').count(), expected.length);
      assert.equal(await panel.locator("thead th[aria-sort=ascending]").count(), 1);
    }
    for (const width of [320, 1280]) {
      await page.setViewportSize({ width, height: 800 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.equal(await panel.getByText("Scroll to see all five scores").isVisible(), width === 320);
      const table = panel.getByRole("region", { name: "Top 10 benchmark results" });
      assert.equal(await table.evaluate(el => el.scrollWidth > el.clientWidth), width === 320);
    }
    await panel.getByRole("link", { name: "View full leaderboard" }).click();
    assert.ok(page.url().endsWith("/leaderboard.html#leaderboard"));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
