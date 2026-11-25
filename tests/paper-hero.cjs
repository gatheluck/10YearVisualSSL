const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ reducedMotion: "no-preference" });
    await page.goto(process.env.HERO_URL || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href);
    const hero = page.locator(".paper-cover");
    const scope = await hero.locator("dl div").evaluateAll(items =>
      items.map(item => [item.querySelector("dd").textContent, item.querySelector("dt").textContent]));
    assert.deepEqual(scope, [
      ["113", "Historical model configurations"], ["2,500+", "Training and evaluation experiments"],
      ["5", "Core tasks with LoRA adaptation"], ["40", "Extended settings with frozen probes"],
    ]);
    const traces = hero.locator("svg polyline");
    assert.deepEqual(await traces.evaluateAll(lines => lines.map(line =>
      Array.from(line.points, point => point.y))), await page.evaluate(() => historicalRanks.map(model => model.ranks)));
    const plot = hero.locator(".rank-field svg");
    const frame = () => plot.screenshot({ animations: "allow" });
    const changes = async () => {
      const before = await frame();
      await page.waitForTimeout(400);
      return !before.equals(await frame());
    };
    await page.waitForTimeout(4100);
    assert.ok(await changes(), "The visible plot must still move after the opening");
    await hero.getByRole("button", {name:"Pause motion"}).click();
    assert.ok(!(await changes()), "Pausing freezes the actual plot");
    await hero.getByRole("button", {name:"Play motion"}).click();
    assert.ok(await changes());
    await page.emulateMedia({reducedMotion:"reduce"});
    await hero.getByRole("button", {name:"Play motion"}).waitFor();
    assert.ok(!(await changes()), "Reduced motion starts with a static plot");
    await hero.getByRole("button", {name:"Play motion"}).click();
    assert.ok(await changes(), "Readers may explicitly enable motion");
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({width, height:900});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      const last = await hero.locator("dd").last().boundingBox();
      assert.ok(last.y + last.height < 900, "Scope visible at " + width);
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
