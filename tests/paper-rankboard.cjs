const assert = require("node:assert/strict"),
  path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  );
  try {
    const page = await browser.newPage();
    await page.goto(
      process.env.BOARD_HOME ||
        pathToFileURL(path.resolve(__dirname, "../paper-site/leaderboard.html")).href,
    );
    await page
      .getByRole("link", { name: "Explore leaderboard", exact: true })
      .click();
    const board = page.locator("#leaderboard");
    assert.match(
      await board.innerText(),
      /Figure 1/i,
      "The primary leaderboard must be Figure 1",
    );
    assert.equal(await page.locator(".rank-visuals .matrix-column").count(), 109);
    assert.equal(await page.locator(".rank-visuals .matrix-cell").count(), 545);
    const heatmap = page.locator(".rank-visuals");
    const selectedName = page.locator(".rank-visuals #matrix-selection");
    const columns = page.locator(".rank-visuals .matrix-column");
    const cellColors = () =>
      columns
        .nth(6)
        .locator(".matrix-cell")
        .evaluateAll((cells) =>
          cells.map((c) => getComputedStyle(c).backgroundColor),
        );
    const before = await cellColors();
    await columns.nth(6).locator(".matrix-cell").first().click();
    await columns.nth(7).hover();
    assert.equal(
      await selectedName.allTextContents().then((t) => t.join("")),
      "RAEv2",
      "A visible horizontal label must identify the selected model",
    );
    const marker = (col) =>
      col.evaluate((el) => ({
        tick: getComputedStyle(el.querySelector(".model-tick")).backgroundColor,
        pointer: getComputedStyle(el, "::before").content,
        base: getComputedStyle(el, "::after").content,
      }));
    const active = await marker(columns.nth(6)),
      hovered = await marker(columns.nth(7));
    assert.equal(active.tick, "rgb(255, 199, 0)");
    assert.notEqual(active.tick, hovered.tick);
    assert.notEqual(active.pointer, "none");
    assert.notEqual(active.base, "none");
    assert.equal(hovered.pointer, "none");
    assert.deepEqual(await cellColors(), before);
    await columns.first().locator(".matrix-cell").first().click();
    assert.equal((await marker(columns.nth(6))).pointer, "none");
    const names = page.locator(".rank-visuals .model-tick");
    assert.deepEqual(
      await names.allTextContents(),
      await page.evaluate(() => historicalRanks.map(model => model.name)),
      "Each heatmap column must have its published configuration name on the axis",
    );
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(
        await names.evaluateAll((ticks) =>
          ticks.every((t) => {
            const a = t.getBoundingClientRect(),
              b = t.parentElement
                .querySelector(".matrix-cell:nth-child(5)")
                .getBoundingClientRect();
            return (
              a.width > 0 &&
              a.height > 0 &&
              a.top >= b.bottom - 1 &&
              Math.abs((a.left + a.right - b.left - b.right) / 2) < 1
            );
          }),
        ),
        "Axis text must stay centered below the cells",
      );
      await columns.first().locator(".matrix-cell").first().press("End");
      await page.keyboard.press("Enter");
      const last = await names.last().boundingBox(),
        end = await page.locator(".rank-visuals .matrix-viewport").boundingBox();
      assert.ok(
        last.x >= end.x && last.x + last.width <= end.x + end.width + 1,
        "Selected model must be visible",
      );
      await columns.first().locator(".matrix-cell").first().click();
      await names.nth(1).click();
      assert.equal(await selectedName.innerText(), "DINOv2");
    }
    await columns.nth(6).locator(".matrix-cell").first().click();
    assert.match(await heatmap.locator("#matrix-model-profile").innerText(), /RAEv2/);
    assert.equal(await heatmap.locator("#matrix-model-profile .rank-rail").count(),5);
    assert.ok(await heatmap.locator("#matrix-model-profile .rank-rail").evaluateAll(rails =>
      rails.every(rail => Number.parseFloat(rail.style.getPropertyValue("--position")) ===
        (Number(rail.parentElement.querySelector("dd small").textContent.replace("Rank ", ""))-1)*100/108)));
    assert.deepEqual(
      await heatmap.locator("#matrix-model-profile dd small").allTextContents(),
      ["Rank 2", "Rank 5", "Rank 1", "Rank 8", "Rank 100"],
    );
    await columns.first().locator(".matrix-cell").first().click();
    assert.equal(await selectedName.innerText(), "DINOv3-7B");
    assert.match(await heatmap.locator("#matrix-model-profile").innerText(), /6\.2/);
    const rows = board.locator("tbody tr");
    assert.equal(await rows.count(), 10);
    await board
      .getByRole("button", { name: "Show all 109 models" })
      .click();
    assert.equal(await rows.count(), 109);
    // Reconstruct the same digest from the overall and five task views.
    const rowsByName = new Map((await rows.evaluateAll(nodes => nodes.map(row => [
      row.cells[1].querySelector("button").textContent,
      [row.cells[0].textContent, row.cells[1].querySelector("button").textContent, row.cells[2].textContent],
    ]))));
    for (const task of ["in1k", "coco", "ade20k", "nyuv2", "ssv2"]) {
      await board.getByLabel("Rank by").selectOption(task);
      for (const [name, rank] of await rows.evaluateAll(nodes => nodes.map(row => [
        row.cells[1].querySelector("button").textContent, row.cells[0].textContent.slice(1),
      ]))) rowsByName.get(name).push(rank);
    }
    await board.getByLabel("Rank by").selectOption("overall");
    const displayed = [...rowsByName.values()].map(values => values.join("|")).join("\n");
    assert.equal(
      require("node:crypto")
        .createHash("sha256")
        .update(displayed)
        .digest("hex"),
      "918c0bb5cd54857d98cbb9279d273d6d521abea2499a0118fc268ff70521ba71",
    );
    const query = board.getByLabel("Find a model");
    await query.fill("V-JEPA");
    assert.equal(await rows.count(), 4);
    await query.fill("xx-no-match");
    assert.equal(await rows.count(), 0);
    assert.match(await board.innerText(), /No matching models/);
    await query.fill("");
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    console.log(
      "Primary Figure 1 board, 545 ranks, exact profile, full list, search and mobile passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
