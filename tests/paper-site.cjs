const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

// Digests of ordered (checkpoint, task, protocol, mean, SD, dagger) tuples
// independently transcribed from Figure 3 of the November manuscript.
const paperDigests = {
  LP: "7a4b03e759494bbfb9cd218d467134238dd52ed6e50e3c341dfbbd558a8be1ee",
  AP: "63776e4465b7e86f5dddd7bfe4c092e9160e129180e4ba9003730ce5676e7bea",
  DPT: "5c29b5480851862abaefc1697239245944ba63299c24fc30d827036f3190226f",
  LoRA: "ca1e833a8309c87909575985cafe8a78225ca0c56154078a04f3675813ba5772",
};

(async () => {
  const browser = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  );
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const url = pathToFileURL(
      path.resolve(__dirname, "../paper-site/leaderboard.html"),
    ).href;
    await page.goto(url);
    assert.equal(await page.locator("h1").count(), 1, "Research title missing");
    assert.match(
      await page.locator("h1").innerText(),
      /Visual transfer leaderboard/i,
    );
    await page.getByRole("link", { name: "Basic Five probes" }).click();
    await page.getByRole("button", { name: "COCO", exact: true }).click();
    assert.match(await page.locator("#finding").innerText(), /0 of 10/);
    await page.getByRole("button", { name: "SSv2", exact: true }).click();
    assert.match(await page.locator("#finding").innerText(), /4 of 10/);
    assert.equal(await page.locator("#unified tbody tr").count(), 5);
    let total = 0;
    for (const [protocol, n] of [
      ["LP", 25],
      ["AP", 25],
      ["DPT", 10],
      ["LoRA", 25],
    ]) {
      await page.getByRole("button", { name: protocol, exact: true }).click();
      assert.equal(await page.locator("#unified tbody button").count(), n);
      total += n;
      const download = page.waitForEvent("download");
      await page.getByRole("button", { name: "Download CSV" }).click();
      const csv = await fs.readFile(await (await download).path(), "utf8");
      assert.equal(csv.trim().split(/\r?\n/).length, n + 1);
      assert.match(csv, /source/);
      const actual = csv
        .trim()
        .split(/\r?\n/)
        .slice(1)
        .map((line) => {
          const fields = [...line.matchAll(/"((?:[^"]|"")*)"/g)].map((m) =>
            m[1].replaceAll('""', '"'),
          );
          assert.equal(fields.length, 10);
          assert.equal(fields[7], "3");
          return [
            fields[0],
            fields[1],
            fields[2],
            fields[5],
            fields[6],
            fields[8].startsWith("†"),
          ];
        });
      const digest = require("node:crypto")
        .createHash("sha256")
        .update(JSON.stringify(actual))
        .digest("hex");
      assert.equal(
        digest,
        paperDigests[protocol],
        "CSV differs from the manuscript",
      );
    }
    assert.equal(total, 85);
    await page.getByRole("button", { name: /87\.90/ }).click();
    assert.match(
      await page.locator("dialog[open]").innerText(),
      /shorter|shortened/i,
    );
    assert.match(await page.locator("dialog[open]").innerText(), /ImageNet-1K/);
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "DPT", exact: true }).click();
    await page.getByRole("button", { name: /Sort by NYUv2/ }).click();
    assert.match(await page.locator("#unified tbody tr").first().innerText(), /DINOv3/);
    await page.reload();
    assert.equal(await page.locator("#unified tbody button").count(), 10);
    await page.getByRole("button", { name: "LP", exact: true }).click();
    await page.getByRole("button", { name: "Sort by SSv2" }).click();
    assert.match(await page.locator("#unified tbody tr").first().innerText(), /SigLIP/);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    );
    assert.deepEqual(errors, []);
    if (process.env.SCREENSHOT_DIR) {
      for (const [w, h] of [
        [390, 844],
        [1440, 1000],
      ]) {
        await page.setViewportSize({ width: w, height: h });
        await page.goto(url);
        await page.screenshot({
          path: path.join(
            process.env.SCREENSHOT_DIR,
            "vision-site-" + w + ".png",
          ),
          fullPage: true,
        });
      }
    }
    console.log(
      "Passed: article, chart, 85 cells, details, CSV, ranking, deep link and mobile width.",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
