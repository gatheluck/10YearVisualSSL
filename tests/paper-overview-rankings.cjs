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
    const range = panel.getByLabel("Show ranks");
    assert.equal(await view.inputValue(), "overall");
    assert.equal(await range.inputValue(), "10");
    const source = await page.evaluate(() => historicalRanks);
    for (const cutoff of [10, 30]) {
      await range.selectOption(String(cutoff));
      for (const [index, value] of ["overall", "in1k", "coco", "ade20k", "nyuv2", "ssv2"].entries()) {
        await view.selectOption(value);
        const rank = r => index === 0 ? r.place : r.ranks[index - 1];
        const expected = source.filter(r => rank(r) <= cutoff).sort((a,b) => rank(a)-rank(b));
        const rows = panel.locator("tbody tr");
        assert.deepEqual(await rows.evaluateAll(nodes => nodes.map(row =>
          [...row.cells].map(cell => cell.textContent))), expected.map(r => [
          String(rank(r)), r.name, index ? String(r.place) : (r.ranks.reduce((a,b)=>a+b,0)/5).toFixed(1)
        ]));
        assert.equal(await rows.locator('a[href^="https://"]').count(), expected.length);
      }
    }
    await page.setViewportSize({ width: 320, height: 800 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
