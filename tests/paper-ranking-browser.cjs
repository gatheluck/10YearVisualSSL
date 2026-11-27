const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { after, before, test } = require("node:test");
const { chromium } = require("playwright");

const base = process.env.BOARD_HOME ||
  pathToFileURL(path.resolve(__dirname, "../paper-site/leaderboard.html")).href;
let browser;
before(async () => {
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
});
after(async () => { await browser.close(); });

test("the first ranking view is compact and opens COCO conditions from a model with the keyboard", async () => {
  for (const width of [320, 390, 1280]) {
    const page = await browser.newPage({viewport:{width,height:850}});
    try {
      await page.goto(base);
      const rankings = page.locator("#leaderboard");
      const table = rankings.getByRole("table", {name:"Model rankings"});
      assert.equal(await table.count(), 1);
      assert.deepEqual(await table.locator("thead th").allTextContents(), ["Overall rank", "Model", "Mean rank"]);
      assert.equal(await rankings.locator("tbody tr").count(), 10);
      assert.equal(await rankings.getByRole("region", { name: "Selected model" }).isVisible(), false);
      assert.equal(await rankings.getByText("Task transfer rank matrix").isVisible(), false);
      const task = rankings.getByLabel("Rank by");
      await task.selectOption("coco");
      assert.deepEqual(await table.locator("thead th").allTextContents(), ["Task rank", "Model", "COCO 2017 score"]);
      const row = table.locator("tbody tr").first();
      assert.match(await row.innerText(), /#1.*DINOv3-7B.*40\.77 bbox AP/s);
      assert.ok(await table.evaluate(el => el.parentElement.scrollWidth <= el.parentElement.clientWidth));
      await row.getByRole("button", {name:"DINOv3-7B",exact:true}).press("Enter");
      const profile = rankings.getByRole("region", {name:"Selected model"});
      assert.equal(await profile.locator("dl > div").count(), 5);
      const coco = profile.getByRole("button", {name:/COCO 2017.*40\.77 bbox AP/});
      await coco.press("Enter");
      const modal = page.getByRole("dialog");
      assert.match(await modal.innerText(), /COCO 2017.*Bounding-box AP.*higher is better/s);
      assert.match(await modal.innerText(), /Figure 2 v45.*40\.77/s);
      assert.match(await modal.innerText(), /Original-setting evaluation \(ASIS\)/);
      assert.equal(await modal.getByRole("link", {name:"Figure 2 manuscript scores"}).getAttribute("href"), "#scores");
      await page.keyboard.press("Escape");
      assert.ok(await coco.evaluate(button => document.activeElement === button));
      assert.equal(await task.inputValue(), "coco");
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth === innerWidth));
    } finally { await page.close(); }
  }
});

test("every published rank is joined to its own v45 configuration and missing scores remain missing", async () => {
  const page = await browser.newPage({ viewport: { width: 320, height: 850 } });
  try {
    await page.goto(base);
    const ranks = page.locator("#leaderboard");
    await ranks.getByRole("button",{name:"Show all 109 models"}).click();
    const { ranked, v45 } = await page.evaluate(() => ({ ranked: historicalRanks, v45: historicalScores }));
    const ids = new Set();
    for (const model of ranked) {
      const score = model.name === "V-JEPA (ViT-H)" ? v45.find(row => row.id === 29)
        : model.name === "V-JEPA (ViT-L)" ? v45.find(row => row.id === 54)
          : v45.find(row => row.name === model.name);
      assert.ok(score);
      assert.equal(score.year, model.year);
      ids.add(score.id);
    }
    assert.equal(ids.size,109);
    const task = ranks.getByLabel("Rank by");
    assert.equal(await task.count(), 1, "Each task must be selectable from the primary rankings");
    for (const [index,key] of ["in1k","coco","ade20k","nyuv2","ssv2"].entries()) {
      await task.selectOption(key);
      assert.ok(await ranks.locator(".rank-scroll").evaluate(el => el.scrollWidth <= el.clientWidth));
      const displayed = await ranks.locator("tbody tr").evaluateAll(rows => rows.map(row => [
        row.cells[0].textContent, row.cells[1].querySelector("button").textContent,
        row.cells[2].textContent, Number(row.dataset.legendId),
      ]));
      const suffix = ["%"," bbox AP","%"," m","%"][index];
      assert.deepEqual(displayed, ranked.slice().sort((a,b) => a.ranks[index] - b.ranks[index]).map(model => {
        const score = model.name === "V-JEPA (ViT-H)" ? v45.find(row => row.id === 29)
          : model.name === "V-JEPA (ViT-L)" ? v45.find(row => row.id === 54)
            : v45.find(row => row.name === model.name);
        const value = score.scores[index];
        const label = value !== null ? value + suffix : model.name === "AlayaWorld v1.1" ? "Excluded" : "Unavailable";
        return ["#" + model.ranks[index], model.name, label, score.id];
      }));
    }
  } finally { await page.close(); }
});

test("comparisons open on demand and existing fragments restore the right panel", async () => {
  const page = await browser.newPage();
  try {
    await page.goto(base);
    const nav = page.getByRole("navigation", {name:"Leaderboard comparisons"});
    assert.equal(await page.locator("#history-task").isVisible(), false);
    assert.equal(await page.locator("#unified #csv").isVisible(), false);
    assert.equal(await page.locator("#controlled #benchmark").isVisible(), false);
    assert.equal(await page.locator("#figure-five select").isVisible(), false);
    const rank = page.locator("#leaderboard");
    await rank.getByLabel("Rank by").selectOption("coco");
    await rank.getByLabel("Find a model").fill("DINOv2");
    await nav.getByRole("link",{name:"History"}).press("Enter");
    assert.ok(await page.locator("#history-task").isVisible());
    await page.locator("#year").selectOption("2013");
    await nav.getByRole("link",{name:"Basic Five probes"}).click();
    await page.getByRole("button",{name:"Download CSV"}).waitFor();
    await page.locator("#history-task").waitFor({ state: "hidden", timeout: 3000 });
    await page.goBack();
    await page.locator("#history-task").waitFor();
    assert.equal(await page.locator("#year").inputValue(),"2013");
    await page.goBack();
    assert.equal(await rank.getByLabel("Find a model").inputValue(),"DINOv2");
    assert.equal(await rank.getByLabel("Rank by").inputValue(),"coco");
    for (const [target, selector] of [["#scores","#scores[aria-pressed=true]"],["#unified?protocol=LoRA&task=all","#tracks button[aria-pressed=true]"],["#rank-matrix","#rank-matrix"],["#rank-atlas","#rank-atlas"]]) {
      await page.goto(new URL(target, base).href);
      await page.locator(selector).waitFor();
      if (target.startsWith("#unified?")) {
        assert.equal(await page.locator(selector).innerText(), "LoRA");
      }
    }
  } finally { await page.close(); }
});
