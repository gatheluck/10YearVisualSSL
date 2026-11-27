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
    const control = page.locator("#controlled");
    const task = control.getByLabel("Benchmark");
    assert.match(await control.locator(".summary").innerText(), /same 32 methods.*selected benchmark.*original.*controlled/is);
    assert.match(await control.locator("#chart-note").innerText(), /horizontal.*ASIS.*vertical.*CTRL.*one method.*diagonal.*tie/is);
    const panels = [];
    for (const name of ["ImageNet-1K", "COCO", "ADE20K", "NYUv2", "SSv2"]) {
      await task.selectOption(name);
      const rows = page.locator("#controlled tbody tr");
      assert.equal(await rows.count(), 32);
      assert.equal(await page.locator("#paired circle").count(), 32);
      for (const outcome of ["Better", "Tied", "Worse"]) {
        const expected = await rows.locator("td").getByText(outcome, {exact:true}).count();
        const counter = control.locator("#control-outcomes div").filter({hasText:outcome});
        assert.equal(await counter.count(), 1, outcome + " needs a readable count");
        assert.equal(await counter.getByRole("term").innerText(), outcome);
        assert.equal(await counter.getByRole("definition").innerText(), String(expected));
      }
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
      await page.locator("#controlled").getByRole("row", { name: /DINOv3/ }).innerText(),
      /\+0\.115\s+Worse/,
    );
    assert.match(
      await page.locator("#controlled").getByRole("row", { name: /Split-Brain/ }).innerText(),
      /-0\.203\s+Better/,
    );
    await page.locator("#controlled").getByText("Evaluation conditions", { exact: true }).click();
    const conditions = await page.locator("#controlled details[open]").innerText();
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
    assert.match(await control.locator("#control-direction").innerText(), /Lower RMSE is better/i);
    for (const width of [320, 390, 768, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(await control.evaluate(section => {
        const inside = (el, outer) => {
          const rect = el.getBoundingClientRect();
          return rect.left >= outer.left - 1 && rect.right <= outer.right + 1 &&
            rect.top >= outer.top - 1 && rect.bottom <= outer.bottom + 1;
        };
        const chart = section.querySelector("#paired");
        const summary = section.querySelector(".reading");
        const card = document.querySelector('[href="#controlled"][aria-label]');
        return [...chart.querySelectorAll("text,circle")].every(mark => inside(mark, chart.getBoundingClientRect())) &&
          [...summary.querySelectorAll("dt,dd,summary")].every(text => inside(text, summary.getBoundingClientRect())) &&
          [...card.children].every(el => inside(el, card.getBoundingClientRect())) &&
          section.getBoundingClientRect().right <= innerWidth;
      }), "Figure 4 and its summary fit at " + width);
    }
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
