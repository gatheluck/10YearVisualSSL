const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ reducedMotion: "no-preference" });
    // Returning readers may have the data script from before the shared color helper.
    await page.route("**/history-data.js", async route => {
      const fs = require("node:fs");
      const current = fs.readFileSync(path.resolve(__dirname, "../paper-site/history-data.js"), "utf8");
      await route.fulfill({contentType:"text/javascript", body:current.split("const taskRankColor =")[0]});
    });
    await page.goto(process.env.HERO_URL || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href);
    const hero = page.locator(".paper-cover");
    const scope = await hero.locator("dl div").evaluateAll(items =>
      items.map(item => [item.querySelector("dd").textContent, item.querySelector("dt").textContent]));
    assert.deepEqual(scope, [
      ["113", "Historical model configurations"], ["2,500+", "Training and evaluation experiments"],
      ["5", "Core tasks with LoRA adaptation"], ["40", "Extended settings with frozen probes"],
    ]);
    const story = hero.getByRole("figure", {name:/Same model, different ranks/});
    assert.equal(await story.count(), 1, "Introduce the comparison with a readable, labeled example");
    const ranks = story.locator("output");
    assert.deepEqual(await ranks.allTextContents(), ["#1", "#100"]);
    assert.match(await story.innerText(), /Segmentation/);
    assert.match(await story.innerText(), /Video recognition/);
    await story.getByRole("button", {name:"DINOv3-7B"}).click();
    assert.deepEqual(await ranks.allTextContents(), ["#9", "#3"]);
    await story.getByRole("button", {name:"RAEv2"}).click();
    assert.deepEqual(await ranks.allTextContents(), ["#1", "#100"]);
    assert.equal(await page.locator(".cover-matrix rect").count(),545);
    const plot = story.locator(".task-pair");
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
