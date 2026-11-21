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
        pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href,
    );
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
    for (const [width, height] of [
      [1440, 900],
      [390, 844],
      [320, 640],
    ]) {
      await page.setViewportSize({ width, height });
      await page.evaluate(() => window.scrollTo(0, 0));
      const chart = await page.locator("#rank-atlas").boundingBox();
      assert.ok(
        chart.y + chart.height / 2 < height,
        "Chart must be visible on first screen at " + width,
      );
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
    }
    await page.getByRole("link", { name: "Explore all 109 rankings" }).click();
    assert.equal(await page.locator("tbody tr").count(), 109);
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
