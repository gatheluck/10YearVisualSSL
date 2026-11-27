const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");
(async () => {
  const b = await chromium.launch(
    process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {},
  );
  try {
    const p = await b.newPage();
    const base =
      process.env.BOARD_HOME ||
      pathToFileURL(path.resolve(__dirname, "../paper-site/leaderboard.html")).href;
    await p.goto(base);
    await p
      .getByRole("button", { name: "Show all 109 models" })
      .click();
    async function verify(selector, n) {
      const links = p.locator(selector + " a");
      assert.equal(
        await links.count(),
        n,
        "Every model row has an original source",
      );
      assert.ok(
        await links.evaluateAll((ls) =>
          ls.every(
            (l) =>
              l.href.startsWith("https://") &&
              l.title.length > 0 &&
              l.target === "_blank" &&
              l.rel.includes("noopener"),
          ),
        ),
      );
    }
    await verify("#leaderboard tbody th", 109);
    await verify("#unified tbody th", 5);
    assert.equal(
      await p
        .locator("#leaderboard tbody")
        .getByRole("link", { name: "Original source for iGPT", exact: true })
        .getAttribute("href"),
      "https://proceedings.mlr.press/v119/chen20s.html",
    );
    await p.getByText("Show the rank heatmap", { exact: true }).click();
    await p.locator("#inspect-config").selectOption("FLUX.2");
    assert.match(
      await p.locator("#rank-profile a").getAttribute("title"),
      /official release/i,
    );
    for (const [file, count] of [
      ["history.html", 109],
      ["control.html", 32],
    ]) {
      await p.goto(new URL(file, base).href);
      await verify(file === "history.html" ? "#history tbody th" : "#controlled tbody th", count);
      if (file === "history.html") {
        await p.getByRole("button", { name: "Figure 2 · Scores" }).click();
        await verify("#history tbody th", 113);
      }
    }
    // Snapshot of the checked configuration-to-primary-source mapping.
    const entries = await p.evaluate(() =>
      Array.from(modelSources, ([n, r]) => n + "|" + r.href)
        .sort()
        .join("\n"),
    );
    assert.equal(
      require("node:crypto").createHash("sha256").update(entries).digest("hex"),
      "4099fb5def5c5e2fe1f52b1cf587f14e695b583f78ec77564565fedf88e5a805",
    );
    console.log("Source links verified in all figure cohorts.");
  } finally {
    await b.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
