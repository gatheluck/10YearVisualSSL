const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const { pathToFileURL } = require("node:url");
const path = require("node:path");
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage();
    const base = process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href;
    for (const file of ["index.html", "paper.html", "extended.html"]) {
      await page.goto(new URL(file, base).href);
      const feature = page.locator("#figure-five");
      assert.equal(await feature.locator("figure img").count(), 1, file + " shows the published chart inline");
      const image = feature.locator("img");
      await image.evaluate(img => img.decode());
      assert.ok(await image.evaluate(img => img.naturalWidth > 2000));
      await page.setViewportSize({width:390,height:844});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(await feature.locator("figure a").evaluateAll(links => links.map(link => link.getAttribute("href"))), ["figure-5.png", "figure-5.png"]);
      await feature.getByRole("link", {name:"Inspect task scores"}).click();
      await page.waitForURL("**/extended.html#matrix");
      assert.equal(await page.locator("#matrix tbody tr").count(), 38);
    }
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
