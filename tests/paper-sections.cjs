const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const { pathToFileURL } = require("node:url");
const path = require("node:path");
(async () => {
  const browser = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  );
  try {
    const page = await browser.newPage();
    await page.goto(
      pathToFileURL(path.resolve(__dirname, "../paper-site/leaderboard.html")).href,
    );
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(
      await page
        .getByRole("link", { name: /Explore historical results/ })
        .isVisible(),
      "Both experiment paths must stay visible on mobile",
    );
    const story = page.locator(".readout-story");
    assert.equal(
      await story.count(),
      1,
      "A figure and its finding must share one readout story",
    );
    await story.getByRole("button", { name: "COCO", exact: true }).click();
    assert.equal(await story.locator("#reversal-count").innerText(), "0 / 10");
    await story.getByRole("button", { name: "SSv2", exact: true }).click();
    assert.equal(await story.locator("#reversal-count").innerText(), "4 / 10");
    await page.getByRole("link", { name: /Explore paired results/ }).click();
    await page.waitForURL("**/leaderboard.html#controlled");
    for (const benchmark of ["ImageNet-1K", "NYUv2"]) {
      await page.locator("#controlled").getByLabel("Benchmark").selectOption(benchmark);
      const better = await page.locator("td.better").count();
      assert.equal(
        await page.locator("#outcome-strip .better").count(),
        better,
      );
      assert.equal(await page.locator("#outcome-strip i").count(), 32);
    }
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    console.log("Readout story and controlled results stay synchronized.");
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
