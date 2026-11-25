const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { pathToFileURL } = require("node:url");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  );
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(
      process.env.CONTROL_HOME ||
        pathToFileURL(path.resolve(__dirname, "../paper-site/leaderboard.html")).href,
    );
    const link = page.getByRole("link", { name: "ASIS vs CTRL", exact: true });
    assert.equal(await link.count(), 1, "The home page must link to Figure 4");
    await link.click();
    const task = page.getByLabel("Benchmark");
    const panels = [];
    for (const name of ["ImageNet-1K", "COCO", "ADE20K", "NYUv2", "SSv2"]) {
      await task.selectOption(name);
      const rows = page.locator("tbody tr");
      assert.equal(await rows.count(), 32);
      assert.equal(await page.locator("svg circle").count(), 32);
      panels.push(
        ...(await rows.evaluateAll((rs) =>
          rs.map((r) => [...r.cells].map((c) => c.textContent).join("|")),
        )),
      );
    }
    assert.equal(
      createHash("sha256").update(panels.join("\n")).digest("hex"),
      "117cc84e583cf4362915a334f6cbfd60f0ab8eca1a5dc0e03877ca1f42b39501",
    );
    await task.selectOption("NYUv2");
    assert.match(
      await page.getByRole("row", { name: /DINOv3/ }).innerText(),
      /\+0\.115\s+Worse/,
    );
    assert.match(
      await page.getByRole("row", { name: /Split-Brain/ }).innerText(),
      /-0\.203\s+Better/,
    );
    await page.getByText("Evaluation conditions", { exact: true }).click();
    const conditions = await page.locator("details[open]").innerText();
    for (const phrase of [
      "ViT-B/16",
      "unlabeled ImageNet-1K",
      "300 epochs",
      "downstream",
      "feature extraction",
    ])
      assert.ok(conditions.includes(phrase), phrase);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    assert.deepEqual(errors, []);
    console.log(
      "Figure 4: navigation, 160 pairs and changes, score direction, chart, protocols and mobile passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
