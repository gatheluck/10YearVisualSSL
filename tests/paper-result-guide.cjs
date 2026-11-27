const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { after, before, test } = require("node:test");
const { chromium } = require("playwright");

const overview = process.env.BOARD_HOME ||
  pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href;
let browser;
before(async () => {
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
});
after(async () => { await browser.close(); });

test("Figure 2 links its two versions to their evaluation conditions", async () => {
  const page = await browser.newPage({ reducedMotion: "reduce" });
  try {
    await page.clock.install();
    await page.clock.pauseAt(new Date(Date.now() + 1500));
    await page.goto(overview);
    const film = page.locator(".history-film");
    assert.match(await film.locator("figcaption").innerText(), /Figure 2 v46/i);
    await film.getByRole("button", { name: "Play motion" }).click();
    await page.clock.runFor(16500);
    assert.match(await film.locator("h2").innerText(), /COCO 2017/);
    assert.match(await film.locator(".film-best").innerText(), /MAGE · 42\.71/);
    const evaluation = film.getByRole("link", { name: "Source and evaluation conditions" });
    assert.equal(await evaluation.getAttribute("href"), "leaderboard.html#historical-evaluation");
    await evaluation.click();
    const source = page.locator("#historical-evaluation");
    assert.match(await source.innerText(), /v45.*v46.*MAGE.*42\.71/s);
    await source.getByRole("link", { name: "View manuscript scores" }).click();
    assert.equal(new URL(page.url()).hash, "#scores");
    await page.locator("#history-task").selectOption("coco");
    const mage = page.locator("#history tbody tr").filter({ has: page.getByRole("link", { name: "MAGE", exact: true }) });
    await mage.getByRole("button", { name: "Unavailable" }).click();
    assert.match(await page.getByRole("dialog").innerText(), /Figure 2 v45.*MAGE.*bounding.box/is);
  } finally { await page.close(); }
});

test("changing the ranking keeps the selected score and model visible together", async () => {
  const page = await browser.newPage();
  try {
    for (const width of [320, 390, 1280]) {
      await page.setViewportSize({ width, height: 860 });
      await page.goto(overview);
      const panel = page.locator("#rankings");
      const table = panel.getByRole("region", { name: "Top 10 benchmark results" });
      for (const task of ["ssv2", "coco", "nyuv2", "overall"]) {
        await panel.getByLabel("Rank by").selectOption(task);
        const visible = await table.evaluate(region => {
          const row = region.querySelector("tbody tr");
          const cell = row.querySelector(".selected-rank");
          const model = row.querySelector("th");
          const edge = region.getBoundingClientRect();
          const score = cell.getBoundingClientRect();
          const name = model.getBoundingClientRect();
          return {
            clear: score.left >= name.right - 1 && score.right <= edge.right + 1 &&
              name.left >= edge.left - 1,
            result: cell.innerText,
            horizontalPageScroll: document.documentElement.scrollWidth > innerWidth,
          };
        });
        assert.ok(visible.clear, task + " score and model at " + width + "px");
        assert.match(visible.result, task === "overall" ? /^#1$/ : /Rank 1/);
        assert.equal(visible.horizontalPageScroll, false);
      }
      assert.deepEqual(await table.locator("thead [data-rank]").evaluateAll(nodes => nodes.map(node => node.dataset.rank)),
        ["overall", "in1k", "coco", "ade20k", "nyuv2", "ssv2"]);
    }
  } finally { await page.close(); }
});

test("the Overview introduces the research question and links a verified Figure 1 result", async () => {
  const page = await browser.newPage({ viewport: { width: 390, height: 860 } });
  try {
    await page.goto(overview);
    const cover = page.locator(".paper-cover");
    const question = cover.getByText(/Which vision models transfer best across tasks/);
    assert.ok(await question.isVisible());
    assert.ok(await question.evaluate(el => {
      const luminance = (channels) => channels.map(n => n / 255)
        .map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
        .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
      const foreground = getComputedStyle(el).color.match(/\d+/g).map(Number);
      const lightestCoverColor = [23, 75, 89];
      return (luminance(foreground) + 0.05) / (luminance(lightestCoverColor) + 0.05) >= 4.5;
    }), "The question has sufficient contrast throughout the cover gradient");
    const result = cover.getByRole("link", { name: /RAEv2.*first on ADE20K.*seventh overall/ });
    await result.click();
    assert.equal(new URL(page.url()).hash, "#rankings");
    const rae = page.locator("#rankings tbody tr").filter({ has: page.getByRole("link", { name: "RAEv2", exact: true }) });
    assert.match(await rae.innerText(), /#7/);
    assert.match(await rae.locator("td").nth(4).innerText(), /46\.1\s+Rank 1/);
  } finally { await page.close(); }
});

test("Figure 5 explains the probe labels and links to raw scores and pending settings", async () => {
  const page = await browser.newPage({ viewport: { width: 320, height: 860 } });
  try {
    for (const pageName of ["index.html", "leaderboard.html"]) {
      await page.goto(new URL(pageName, overview).href);
      const figure = page.locator("#figure-five");
      assert.match(await figure.locator(".transfer-facts").innerText(), /frozen-feature probing.*attentive probing/is);
      assert.match(await figure.locator(".radar-guide").innerText(), /scaled to the best.*for each task and probe/i);
      await figure.getByRole("link", { name: "Inspect task scores" }).click();
      assert.equal(new URL(page.url()).hash, "#matrix");
      assert.equal(await page.locator("#matrix tbody tr").count(), 38);
      await page.locator("#figure-five").getByRole("link", { name: "2 settings pending" }).click();
      assert.equal(new URL(page.url()).hash, "#holds");
      assert.match(await page.locator("#holds").innerText(), /SpaceNet 2.*SUN RGB-D/);
    }
  } finally { await page.close(); }
});
