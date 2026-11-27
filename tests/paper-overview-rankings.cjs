const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href);
    const panel = page.locator(".rank-preview");
    assert.equal(await panel.getByRole("heading", { name: "Vision Model Rankings" }).count(), 1);
    assert.deepEqual(await page.locator("main > [id]").evaluateAll(nodes => nodes.map(n => n.id)),
      ["rankings", "figure-five", "geometry", "citation"]);
    assert.equal(await panel.getByRole("img", { name: /Five rows of task ranks/ }).count(), 0);
    const caption = panel.locator("figcaption");
    const datasets = "ImageNet-1K, COCO 2017, ADE20K, NYUv2, SSv2";
    const intro = caption.locator("p").filter({ hasText: "Top 10 of 109 models" });
    assert.ok((await intro.innerText()).includes(
      "Top 10 of 109 models across the Basic Five datasets (" + datasets + ")."));
    assert.equal(await intro.locator("strong, b").count(), 0, "Dataset names use the surrounding text weight");
    assert.equal(await panel.getByText("What is the Basic Five?", { exact: true }).count(), 0);
    const reason = panel.getByRole("link", { name: "Why the Basic Five?" });
    assert.equal(await caption.getByRole("link", { name: "Why the Basic Five?" }).count(), 0);
    await page.setViewportSize({ width: 1280, height: 800 });
    const horizontal = await intro.evaluate(paragraph => {
      const text = document.createRange();
      text.selectNodeContents(paragraph);
      const tops = [...text.getClientRects()].map(rect => rect.top);
      return Math.max(...tops) - Math.min(...tops) <= 2;
    });
    assert.ok(horizontal, "At desktop width, the complete introduction fits on one line");
    for (const width of [320, 1280]) {
      await page.setViewportSize({ width, height: 800 });
      const table = await panel.getByRole("region", { name: "Top 10 benchmark results" }).boundingBox();
      assert.ok((await reason.boundingBox()).y >= table.y + table.height);
    }
    await reason.click();
    assert.equal(new URL(page.url()).hash, "#geometry");
    assert.equal(await panel.getByRole("link", { name: "Why the Basic Five?" }).count(), 1);
    await page.goBack();
    const view = panel.getByLabel("Rank by");
    assert.equal(await view.inputValue(), "overall");
    assert.equal(await panel.getByLabel("Show ranks").count(), 0);
    const { ranks, scores } = await page.evaluate(() => ({ ranks: historicalRanks, scores: historicalScores }));
    const rows = panel.locator("tbody tr");
    const board = await browser.newPage();
    await board.goto(new URL("leaderboard.html", page.url()).href);
    await board.getByRole("button", { name: "Show all 109 models" }).click();
    const boardColors = await board.locator(".rank-scroll tbody th b").evaluateAll(labels =>
      Object.fromEntries(labels.map(label => [
        label.textContent, getComputedStyle(label, "::before").backgroundColor,
      ])));
    await board.close();
    for (const [index, value] of ["overall", "in1k", "coco", "ade20k", "nyuv2", "ssv2"].entries()) {
      if (index) await view.selectOption(value);
      const rank = r => index === 0 ? r.place : r.ranks[index - 1];
      const expected = ranks.filter(r => rank(r) <= 10).sort((a,b) => rank(a)-rank(b));
      assert.deepEqual(await rows.evaluateAll(nodes => nodes.map(row =>
        [...row.cells].map(cell => cell.textContent))), expected.map(r => {
        const score = r.name === "V-JEPA (ViT-H)" ? scores.find(s => s.id === 29)
          : r.name === "V-JEPA (ViT-L)" ? scores.find(s => s.id === 54)
            : scores.find(s => s.name === r.name);
        return [r.name + score.family, "#" + r.place,
          (r.ranks.reduce((a,b)=>a+b,0)/5).toFixed(1),
          ...r.ranks.map((place, task) => (score.scores[task] ?? "—") + "Rank " + place)];
      }));
      assert.equal(await rows.locator('a[href^="https://"]').count(), expected.length);
      const activeHeader = panel.locator("thead th[aria-sort=ascending]");
      assert.equal(await activeHeader.count(), 1);
      const activeColor = await activeHeader.evaluate(cell => getComputedStyle(cell).backgroundColor);
      assert.notEqual(activeColor, "rgba(0, 0, 0, 0)");
      const activeColumn = index === 0 ? 1 : index + 2;
      for (const cells of await rows.evaluateAll(nodes => nodes.map(row =>
        Array.from(row.cells, cell => getComputedStyle(cell).backgroundColor)))) {
        assert.equal(cells[activeColumn], activeColor, value + " highlights every cell in the active column");
        assert.equal(cells.filter(color => color === activeColor).length, 1);
      }
      const swatches = await rows.locator("th .preview-family").evaluateAll(labels => labels.map(label => {
        const marker = getComputedStyle(label, "::before");
        return [label.parentElement.querySelector("a").textContent, marker.backgroundColor, marker.width, marker.height];
      }));
      assert.deepEqual(swatches, expected.map(row => [row.name, boardColors[row.name], "8px", "8px"]));
    }
    for (const width of [320, 1280]) {
      await page.setViewportSize({ width, height: 800 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.equal(await panel.getByText("Scroll to see all five scores").isVisible(), width === 320);
      const table = panel.getByRole("region", { name: "Top 10 benchmark results" });
      assert.equal(await table.evaluate(el => el.scrollWidth > el.clientWidth), width === 320);
      const continuation = table.locator("tfoot");
      assert.equal(await continuation.count(), 1, "The results end with a continuation row");
      assert.match(await continuation.innerText(), /^…\s+See all 109 model rankings/);
      await table.evaluate(el => { el.scrollLeft = el.scrollWidth; });
      const link = await continuation.getByRole("link").boundingBox();
      const region = await table.boundingBox();
      assert.ok(link.x >= region.x && link.x + link.width <= region.x + region.width);
      await table.evaluate(el => { el.scrollLeft = 0; });
    }
    await panel.getByRole("link", { name: "See all 109 model rankings" }).click();
    assert.ok(page.url().endsWith("/leaderboard.html"));
    assert.equal(await page.evaluate(() => window.scrollY), 0);
    await page.getByRole("button", { name: "Show all 109 models" }).click();
    assert.equal(await page.locator("#leaderboard tbody tr").count(), 109);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
