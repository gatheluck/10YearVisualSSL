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
  for (const width of [320, 390, 768, 1280]) {
    const page = await browser.newPage({viewport:{width,height:850}});
    try {
      await page.goto(base);
      const heatmap = page.getByRole("figure").filter({has:page.locator("#rank-matrix")});
      assert.ok(await heatmap.isVisible(), "Figure 1 heatmap must be visible without opening a panel");
      assert.match(await heatmap.innerText(), /Columns.*models.*overall.*Rows.*tasks.*cell.*task rank/s);
      assert.match(await heatmap.innerText(), /Darker is better/);
      assert.ok(await heatmap.locator(".matrix-cell").first().evaluate(el => {
        const box = el.getBoundingClientRect();
        return box.top >= 0 && box.bottom <= innerHeight;
      }));
      const rankings = page.locator("#leaderboard");
      const table = rankings.getByRole("table", {name:"Model rankings"});
      assert.equal(await table.count(), 1);
      assert.deepEqual(await table.locator("thead th:visible").allTextContents(), ["Overall rank", "Model", "Mean rank", "Task ranks"]);
      assert.equal(await rankings.locator("tbody tr").count(), 10);
      assert.equal(await rankings.getByRole("region", { name: "Selected model" }).isVisible(), false);
      assert.match(await rankings.locator("#ranking-method").innerText(), /mean of the five published task ranks/i);
      const overallRow = table.locator("tbody tr").first();
      const published = await page.evaluate(() => historicalRanks[0].ranks);
      assert.deepEqual(await overallRow.locator(".task-ranks button strong").allTextContents(), published.map(String));
      assert.ok(await table.evaluate(el => el.parentElement.scrollWidth <= el.parentElement.clientWidth));
      await overallRow.getByRole("button", {name:/COCO 2017.*rank 1/i}).press("Enter");
      assert.match(await page.getByRole("dialog").innerText(), /Figure 2 v45.*40\.77/s);
      await page.keyboard.press("Escape");
      const task = rankings.getByLabel("Rank by");
      await task.selectOption("coco");
      assert.deepEqual(await table.locator("thead th:visible").allTextContents(), ["Task rank", "Model", "COCO 2017 score"]);
      assert.match(await rankings.locator("#ranking-method").innerText(), /Bounding-box AP.*higher is better/i);
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
    const displayedTaskRanks = await ranks.locator(".task-ranks-list").evaluateAll(groups =>
      groups.map(group => [...group.querySelectorAll("strong")].map(rank => Number(rank.textContent))));
    assert.deepEqual(displayedTaskRanks, ranked.map(model => model.ranks));
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

test("the heatmap shows five exact scores before interaction and updates them in place", async () => {
  for (const width of [320, 390, 768, 1280]) {
    const page = await browser.newPage({ viewport: { width, height: 850 }, hasTouch: width <= 390 });
    try {
      await page.goto(base);
      const visual = page.locator(".rank-visuals");
      const details = visual.getByRole("region", { name: "Heatmap model results" });
      const readout = visual.getByRole("status");
      assert.ok(await details.isVisible(), "The five scores must be visible before selecting or opening anything");
      assert.equal(await visual.getByRole("combobox", { name: "Inspect model" }).count(), 0);
      assert.equal(await visual.getByRole("button", { name: "Model results", exact: true }).count(), 0);
      assert.ok(await details.evaluate(panel => panel.previousElementSibling.matches("figure") &&
        Boolean(panel.compareDocumentPosition(document.querySelector("#leaderboard")) & Node.DOCUMENT_POSITION_FOLLOWING)));
      const reference = await page.evaluate(() => historicalRanks.slice(0, 2).map(rank => ({
        ...rank, scores: historicalScores.find(record => record.name === rank.name).scores,
      })));
      const suffixes = ["%", " bbox AP", "%", " m", "%"];
      const taskNames = ["ImageNet-1K", "COCO 2017", "ADE20K", "NYUv2", "SSv2"];
      const metrics = [/Top-1 accuracy/, /Bounding-box AP/, /mIoU/, /RMSE/, /Top-1 accuracy/];
      const checkProfile = async expected => {
        assert.equal(await details.locator("h3").innerText(), expected.name);
        assert.equal(await visual.locator("#matrix-selection").innerText(), expected.name);
        const scores = details.locator("dl > div");
        assert.equal(await scores.count(), 5);
        for (const [index, name] of taskNames.entries()) {
          const score = scores.nth(index);
          assert.match(await score.locator("dt").innerText(), metrics[index]);
          assert.ok((await score.locator("dt").innerText()).startsWith(name));
          assert.equal(await score.getByRole("button").innerText(), expected.scores[index] + suffixes[index]);
          assert.equal(await score.locator("dd small").innerText(), "Rank " + expected.ranks[index]);
        }
        assert.ok(await details.evaluate(panel => {
          const frame = panel.getBoundingClientRect();
          return [...panel.querySelectorAll("h3,dt,dt span,dd button")].every(element => {
            const box = element.getBoundingClientRect();
            return box.left >= frame.left && box.right <= frame.right &&
              element.scrollWidth <= element.clientWidth + 1 && parseFloat(getComputedStyle(element).fontSize) >= 12;
          });
        }), "Model and metric labels and their values fit at " + width);
      };
      await checkProfile(reference[0]);
      const profileNode = await details.elementHandle();
      const cell = visual.locator(".matrix-column").nth(1).locator(".matrix-cell").nth(1);
      await (width <= 390 ? cell.tap() : cell.click());
      assert.equal(await readout.count(), 1, "A cell selection needs a local, announced readout");
      const expected = reference[1];
      assert.match(await readout.innerText(), new RegExp(expected.name + ".*COCO 2017.*[Rr]ank " + expected.ranks[1] + "\\b", "s"));
      for (const item of [cell, readout]) {
        assert.ok(await item.evaluate(el => {
          const box = el.getBoundingClientRect();
          return box.top >= 0 && box.bottom <= innerHeight && box.left >= 0 && box.right <= innerWidth;
        }), "Both the selected cell and its readout are visible immediately at " + width);
      }
      assert.equal(await page.locator("#leaderboard #selected-model").isVisible(), false);
      assert.ok(await details.evaluate((panel, original) => panel === original, profileNode));
      await checkProfile(expected);
      const score = details.getByRole("button", { name: /COCO 2017.*bbox AP/ });
      await score.press("Enter");
      assert.match(await page.getByRole("dialog").innerText(), /COCO 2017.*Bounding-box AP/s);
      await page.keyboard.press("Escape");
      assert.ok(await score.evaluate(el => el === document.activeElement));
      assert.equal(new URL(page.url()).hash, "");
    } finally { await page.close(); }
  }
});

test("the heatmap scrolls to keyboard cells and preserves the selected task on reload", async () => {
  const page = await browser.newPage({ viewport: { width: 320, height: 640 } });
  try {
    await page.goto(base);
    const matrix = page.locator("#rank-matrix");
    const first = matrix.getByRole("button").filter({ hasText: /^$/ }).first();
    assert.equal(await first.count(), 1, "Heatmap marks need keyboard-accessible controls");
    assert.equal(await matrix.locator("button[tabindex='0']").count(), 1);
    await first.focus();
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("Enter");
    const cell = page.locator(".matrix-column").nth(1).locator(".matrix-cell").nth(1);
    assert.ok(await cell.evaluate(el => el === document.activeElement));
    assert.equal(await cell.getAttribute("aria-pressed"), "true");
    const profile = page.getByRole("region", { name: "Heatmap model results" });
    assert.ok(await profile.isVisible(), "Keyboard selection keeps the score panel visible");
    assert.equal(await profile.locator("h3").innerText(), "DINOv2");
    await cell.press("End");
    await page.keyboard.press("Space");
    const last = page.locator(".matrix-column").last().locator(".matrix-cell").nth(1);
    assert.equal(await last.getAttribute("aria-pressed"), "true");
    assert.ok(await last.evaluate(el => {
      const cell = el.getBoundingClientRect();
      const viewport = el.closest(".matrix-viewport").getBoundingClientRect();
      return cell.left >= viewport.left && cell.right <= viewport.right;
    }));
    const lastModel = await page.evaluate(() => historicalRanks.at(-1).name);
    assert.equal(await profile.locator("h3").innerText(), lastModel);
    await page.reload();
    assert.equal(await last.getAttribute("aria-pressed"), "true");
    assert.equal(await profile.locator("h3").innerText(), lastModel);
    assert.match(await page.locator(".rank-visuals [role=status]").innerText(), /COCO 2017/);
    await matrix.getByRole("button", { name: /^Highway Networks · COCO 2017 · Rank/ }).click();
    assert.equal(await profile.getByRole("button", { name: /^COCO 2017 Unavailable/ }).innerText(), "Unavailable");
    await matrix.getByRole("button", { name: /^AlayaWorld v1\.1 · NYUv2 · Rank/ }).click();
    assert.equal(await profile.getByRole("button", { name: /^NYUv2 Excluded/ }).innerText(), "Excluded");
  } finally { await page.close(); }
});

test("ranking models open directly below the selected row and close without losing the filter or focus", async () => {
  for (const width of [320, 390, 768, 1280]) {
    const page = await browser.newPage({ viewport: { width, height: 750 } });
    try {
      await page.goto(base);
      const rankings = page.locator("#leaderboard");
      await rankings.getByLabel("Rank by").selectOption("coco");
      await rankings.getByLabel("Find a model").fill("dino");
      const row = rankings.locator("tbody tr[data-legend-id]").last();
      const model = row.locator("th button");
      const name = await model.innerText();
      await model.press("Enter");
      assert.equal(await model.getAttribute("aria-expanded"), "true", "Opening a model should expand its own row");
      const profile = rankings.getByRole("region", { name: "Selected model" });
      assert.equal(await profile.locator("h3").innerText(), name);
      assert.ok(await profile.evaluate(el =>
        el.closest("tr").previousElementSibling.querySelector("th button").textContent === el.querySelector("h3").textContent));
      assert.ok(await profile.locator("h3").evaluate(el => {
        const box = el.getBoundingClientRect();
        return box.top >= 0 && box.bottom <= innerHeight;
      }));
      await profile.getByRole("button", { name: "Close model profile" }).press("Enter");
      const restored = rankings.getByRole("button", { name, exact: true });
      assert.ok(await restored.evaluate(el => el === document.activeElement));
      assert.equal(await rankings.getByLabel("Find a model").inputValue(), "dino");
      assert.equal(await rankings.getByLabel("Rank by").inputValue(), "coco");
      await restored.press("Space");
      await profile.getByRole("button", { name: /COCO 2017/ }).press("Enter");
      await page.getByRole("dialog").getByRole("link", { name: "Figure 2 manuscript scores" }).press("Enter");
      assert.equal(new URL(page.url()).hash, "#scores");
      await page.goBack();
      assert.equal(new URL(page.url()).hash, "");
      assert.ok(await profile.isVisible());
      assert.equal(await rankings.getByLabel("Find a model").inputValue(), "dino");
      await rankings.getByLabel("Find a model").fill("no-match");
      assert.equal(await profile.isVisible(), false);
    } finally { await page.close(); }
  }
});
