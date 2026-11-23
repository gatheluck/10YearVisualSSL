const assert = require("node:assert/strict");
const { pathToFileURL } = require("node:url");
const path = require("node:path");
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ viewport: { width: 320, height: 640 } });
    const base = process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href;
    for (const file of ["index", "history", "control", "extended", "paper"]) {
      await page.goto(new URL(file + ".html", base).href);
      const links = page.locator("header nav a");
      assert.deepEqual(await links.allTextContents(), ["Overview", "Leaderboard", "Paper"]);
      assert.ok(await links.evaluateAll(items => items.every(item => {
        const r = item.getBoundingClientRect();
        return r.width > 0 && r.left >= 0 && r.right <= innerWidth;
      })));
    }
    const figure = page.locator("#geometry figure");
    assert.equal(await figure.count(), 1, "Figure 7 is presented alongside the analysis");
    const img = figure.locator("img");
    await img.evaluate(i => i.decode());
    assert.ok(await img.evaluate(i => i.naturalWidth >= 1000 && i.alt.includes("patch")));
    await figure.getByRole("link").click();
    assert.equal(await page.locator("img").evaluate(i=>i.naturalWidth),1320);
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
