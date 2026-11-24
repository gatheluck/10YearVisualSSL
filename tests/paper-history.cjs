const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

(async () => {
  const browser = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  );
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(
      process.env.HISTORY_HOME ||
        pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href,
    );
    const link = page.getByRole("link", { name: "History", exact: true });
    assert.equal(
      await link.count(),
      1,
      "Home page must link to historical comparisons",
    );
    await link.click();
    await page.waitForURL("**/history.html", { waitUntil: "load" });
    const rows = page.locator("tbody tr");
    const year = page.getByLabel("First release");
    const task = page.getByLabel("Benchmark");
    assert.equal(await rows.count(), 109);
    assert.match(await rows.first().innerText(), /DINOv3-7B/);
    assert.match(await rows.first().innerText(), /6\.2/);
    // Digest every displayed row in all five panels against the manuscript extraction.
    async function checkPanels(expected, points) {
      const panels = [];
      for (const [i, key] of [
        "in1k",
        "coco",
        "ade20k",
        "nyuv2",
        "ssv2",
      ].entries()) {
        await task.selectOption(key);
        assert.equal(await page.locator("#plot circle").count(), points[i]);
        panels.push(
          ...(await rows.evaluateAll((lines) =>
            lines.map((line) =>
              [...line.cells]
                .flatMap((cell) =>
                  cell.children.length
                    ? [...cell.children].map((child) => child.textContent)
                    : [cell.textContent],
                )
                .join("|"),
            ),
          )),
        );
      }
      assert.equal(
        require("node:crypto")
          .createHash("sha256")
          .update(panels.join("\n"))
          .digest("hex"),
        expected,
      );
    }
    await checkPanels(
      "ba4c87bc4908cc99f7fef3edc54dab0e99a6da8a9ab849d6ca32902141eb1ee1",
      [109, 109, 109, 109, 109],
    );
    await year.selectOption("2013");
    assert.match(await rows.filter({ hasText: "VAE" }).innerText(), /108/);
    await page.getByRole("button", { name: "Figure 2 · Scores" }).click();
    await year.selectOption("");
    assert.equal(await rows.count(), 113);
    await checkPanels(
      "5ac461522d3c5bd86c3106987645f78751766ceca998d2e14d90a5d7a222fc32",
      [112, 110, 112, 112, 113],
    );
    await task.selectOption("coco");
    const score = (name) =>
      rows
        .filter({ has: page.getByText(name, { exact: true }) })
        .getByRole("button");
    assert.match(await score("RAE").innerText(), /19\.16/);
    assert.match(await score("MAGE").innerText(), /Unavailable/);
    assert.match(await score("DA3").innerText(), /^0(\.0+)?$/);
    await score("RAE").click();
    const detail = page.getByRole("dialog");
    assert.match(await detail.innerText(), /encoder/i);
    assert.match(await detail.innerText(), /bounding.box/i);
    await page.keyboard.press("Escape");
    await task.selectOption("nyuv2");
    assert.match(await score("RAE").innerText(), /0\.666/);
    assert.match(await score("AlayaWorld v1.1").innerText(), /Excluded/);
    await year.selectOption("2012");
    assert.equal(await rows.count(), 1);
    assert.match(await rows.innerText(), /AlexNet/);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    assert.deepEqual(errors, []);
    console.log("Historical comparisons passed");
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
