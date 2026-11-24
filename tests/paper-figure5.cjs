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
      assert.equal(await feature.locator("figure svg .model-panel").count(), 5, "Five live radar panels");
      assert.equal(await feature.locator(".profile").count(), 10);
      assert.equal(await feature.locator(".score-point").count(), 400);
      assert.equal(await feature.locator("figure img").count(), 0);
      await feature.getByLabel("Visible profiles").selectOption("LP");
      assert.equal(await feature.locator(".score-point:visible").count(), 200);
      assert.equal(await feature.locator(".profile:visible").count(), 5);
      await feature.getByLabel("Inspect task").selectOption("MPI Sintel");
      assert.match(await feature.locator("output").innerText(), /MPI Sintel.*DINOv3-7B/s);
      const mark=feature.locator('.score-point[data-dataset="ImageNet-10%"][data-probe="LP"][data-model="dino"]');
      await mark.dispatchEvent("click");
      assert.match(await feature.locator("output").innerText(), /ImageNet-10%.*84.71/s);
      await feature.getByLabel("Inspect task").selectOption("SUN RGB-D");
      assert.match(await feature.locator("output").innerText(), /pending/i);
      assert.equal(await feature.locator('.score-point[data-dataset="SUN RGB-D"][data-score]').count(),0);
      await page.setViewportSize({width:390,height:844});
      await feature.getByRole("button",{name:"Enlarge chart"}).click();
      assert.ok(await feature.locator(".transfer-pan").evaluate(e=>e.scrollWidth>e.clientWidth));
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await feature.getByRole("link", {name:"Inspect task scores"}).click();
      await page.waitForURL("**/extended.html#matrix");
      assert.equal(await page.locator("#matrix tbody tr").count(), 38);
    }
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
