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
    const models = hero.getByRole("link",{name:/113 historical model configurations/i});
    const settings = hero.getByRole("link",{name:/40 extended benchmark settings/i});
    assert.equal(await models.count(),1,"The first view must introduce the historical model collection");
    assert.equal(await settings.count(),1);
    const names = locator => locator.locator("i").evaluateAll(nodes => nodes.map(n => n.title));
    assert.deepEqual(await names(models), await page.evaluate(()=>historicalScores.map(m=>m.name)));
    assert.deepEqual(await names(settings), await page.locator(".task-choice option").allTextContents());
    assert.match(await settings.innerText(), /5 selected models/);
    for (const dataset of ["ImageNet","COCO","ADE20K","NYUv2","SSv2"]) assert.ok((await hero.innerText()).includes(dataset));
    assert.match(await hero.innerText(), /2,500\+/);
    assert.equal(await page.locator(".cover-matrix rect").count(),545);
    const plot = models.locator(".coverage-marks");
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
      const last = await settings.boundingBox();
      assert.ok(last.y + last.height < 900, "Scope visible at " + width);
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
