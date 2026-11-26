const assert = require("node:assert/strict");
const { pathToFileURL } = require("node:url");
const path = require("node:path");
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ viewport: { width: 320, height: 640 } });
    const base = process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href;
    await page.goto(base);
    assert.deepEqual(await page.locator("header nav a").allTextContents(), ["Overview","Leaderboard"]);
    for (const file of ["leaderboard", "history", "control", "extended", "index"]) {
      await page.goto(new URL(file + ".html", base).href);
      const links = page.locator("header nav a");
      assert.deepEqual(await links.allTextContents(), ["Overview", "Leaderboard"]);
      assert.ok(await links.evaluateAll(items => items.every(item => {
        const r = item.getBoundingClientRect();
        return r.width > 0 && r.left >= 0 && r.right <= innerWidth;
      })));
    }
    await page.getByRole("link", { name: "View full leaderboard", exact: true }).click();
    const cells = page.locator("#rank-matrix .matrix-cell");
    assert.equal(await cells.count(), 545);
    assert.ok(await cells.evaluateAll(nodes => {
      const swatch = document.createElement("span");
      return nodes.every((cell, i) => {
        swatch.style.backgroundColor = taskRankColor(historicalRanks[Math.floor(i / 5)].ranks[i % 5]);
        return cell.style.backgroundColor === swatch.style.backgroundColor;
      });
    }));
    await page.goBack();
    const figure = page.locator("#geometry figure");
    assert.equal(await figure.count(), 1, "Figure 7 is presented alongside the analysis");
    const img = figure.locator("img");
    await img.evaluate(i => i.decode());
    assert.ok(await img.evaluate(i => i.naturalWidth >= 1000 && i.alt.includes("patch")));
    await figure.getByRole("link").click();
    assert.equal(await page.locator("img").evaluate(i=>i.naturalWidth),1320);
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
