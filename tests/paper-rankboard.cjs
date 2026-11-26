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
    assert.equal(await board.locator(".matrix-column").count(), 109);
    assert.equal(await board.locator(".matrix-cell").count(), 545);
    const select = board.getByLabel("Inspect configuration");
    const selectedName = board.locator("#matrix-selection");
    const columns = board.locator(".matrix-column");
    const cellColors = () =>
      columns
        .nth(6)
        .locator("i")
        .evaluateAll((cells) =>
          cells.map((c) => getComputedStyle(c).backgroundColor),
        );
    const before = await cellColors();
    await select.selectOption({ label: "RAEv2" });
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
    await select.selectOption({ index: 0 });
    assert.equal((await marker(columns.nth(6))).pointer, "none");
    const names = board.locator(".model-tick");
    assert.deepEqual(
      await names.allTextContents(),
      await select.locator("option").allTextContents(),
      "Each heatmap column must have its published configuration name on the axis",
    );
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(
        await names.evaluateAll((ticks) =>
          ticks.every((t) => {
            const a = t.getBoundingClientRect(),
              b = t.parentElement
                .querySelector("i.matrix-cell:last-of-type")
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
      await select.selectOption({ index: 108 });
      const last = await names.last().boundingBox(),
        end = await board.locator(".matrix-viewport").boundingBox();
      assert.ok(
        last.x >= end.x && last.x + last.width <= end.x + end.width + 1,
        "Selected model must be visible",
      );
      await select.selectOption({ index: 0 });
      await names.nth(1).click();
      assert.equal(await select.inputValue(), "DINOv2");
    }
    await select.selectOption({ label: "RAEv2" });
    assert.match(await board.locator("#rank-profile").innerText(), /RAEv2/);
    assert.equal(await board.locator("#rank-profile .rank-rail").count(),5);
    assert.ok(await board.locator("#rank-profile .rank-rail").evaluateAll(rails =>
      rails.every(rail => Number.parseFloat(rail.style.getPropertyValue("--position")) ===
        (Number(rail.parentElement.querySelector("dd").textContent)-1)*100/108)));
    assert.deepEqual(
      await board.locator("#rank-profile dd").allTextContents(),
      ["2", "5", "1", "8", "100"],
    );
    await board.locator(".matrix-column").first().click();
    assert.equal(await select.inputValue(), "DINOv3-7B");
    assert.match(await board.locator("#rank-profile").innerText(), /6\.2/);
    const rows = board.locator("tbody tr");
    assert.equal(await rows.count(), 10);
    await board
      .getByRole("button", { name: "Show all 109 models" })
      .click();
    assert.equal(await rows.count(), 109);
    // Reference digest from the printed places, names, means and five ranks on page one.
    const displayed = await rows.evaluateAll((list) =>
      list
        .map((row) =>
          [...row.cells]
            .map((cell, i) =>
              i === 1 ? cell.querySelector("b").textContent : cell.textContent,
            )
            .join("|"),
        )
        .join("\n"),
    );
    assert.equal(
      require("node:crypto")
        .createHash("sha256")
        .update(displayed)
        .digest("hex"),
      "918c0bb5cd54857d98cbb9279d273d6d521abea2499a0118fc268ff70521ba71",
    );
    const query = board.getByLabel("Find a configuration");
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
