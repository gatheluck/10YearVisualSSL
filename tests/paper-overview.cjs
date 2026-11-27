const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const { pathToFileURL } = require("node:url");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  );
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 900 },
    });
    await page.goto(
      process.env.OVERVIEW_HOME ||
        pathToFileURL(path.resolve(__dirname, "../paper-site/leaderboard.html")).href,
    );
    for (const [width, height, minRows] of [
      [1440, 900, 3],
      [390, 844, 1],
      [320, 640, 0],
    ]) {
      await page.setViewportSize({ width, height });
      assert.equal(await page.locator("#rank-matrix").isVisible(), false);
      if (minRows) {
        const row = await page
          .locator("#leaderboard tbody tr")
          .nth(minRows - 1)
          .boundingBox();
        assert.ok(
          row.y + row.height <= height,
          "Leading standings must be visible before scrolling at " + width,
        );
      }
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
    }
    await page.getByText("Show the rank heatmap", { exact: true }).click();
    assert.ok(await page.locator("#rank-matrix").isVisible());
    await page.getByText("Compare rank trajectories", { exact: true }).click();
    const lines = page.locator("#rank-atlas polyline");
    assert.equal(
      await lines.count(),
      109,
      "The opening chart must show the 109 Figure 1 configurations",
    );
    const series = await lines.evaluateAll((els) =>
      els.map((e) => ({
        name: e.querySelector("title").textContent,
        color: getComputedStyle(e).stroke,
        points: Array.from(e.points, (p) => p.y),
      })),
    );
    assert.equal(new Set(series.map((s) => s.color)).size, 7);
    const dino = series.find((s) => s.name.startsWith("DINOv3-7B"));
    assert.equal(dino.color, "rgb(0, 179, 176)");
    assert.deepEqual(dino.points, [1, 1, 9, 17, 3]);
    const filter = page.getByRole("button", { name: "VideoSSL", exact: true });
    await filter.click();
    assert.equal(await filter.getAttribute("aria-pressed"), "true");
    const active = await lines.evaluateAll((els) =>
      els
        .filter((e) => Number(getComputedStyle(e).opacity) > 0.15)
        .map((e) => e.querySelector("title").textContent),
    );
    assert.ok(active.some((s) => s.startsWith("V-JEPA (ViT-L)")));
    assert.ok(!active.some((s) => s.startsWith("V-JEPA (ViT-H)")));
    await filter.click();
    assert.equal(
      await lines.evaluateAll(
        (es) =>
          es.filter((e) => Number(getComputedStyle(e).opacity) > 0.15).length,
      ),
      109,
    );
    await page.getByRole("link", { name: "Explore all 109 rankings" }).click();
    await page.waitForURL("**/leaderboard.html#history", { waitUntil: "load" });
    assert.equal(await page.locator("#history tbody tr").count(), 109);
    console.log(
      "Opening chart, paper ranks, family colors, filtering, responsive first view and history navigation passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
