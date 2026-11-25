const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const { pathToFileURL } = require("node:url");
const path = require("node:path");
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage();
    await page.goto(process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname, "../paper-site/leaderboard.html")).href);
    const paper = page.getByRole("link", { name: "Overview", exact: true });
    assert.equal(await paper.count(), 1, "Leaderboard links directly to the paper overview");
    await paper.click();
    assert.equal(await page.locator("#geometry tbody tr").count(), 5);
    assert.match(await page.locator("#geometry tbody tr").last().innerText(), /NYUv2.*71°.*0.01.*0.67/s);
    await page.getByRole("link", { name: "Explore Figure 5" }).click();
    const rows = page.locator("#matrix tbody tr");
    assert.equal(await rows.count(), 38);
    const first = rows.first();
    assert.match(await first.locator("td").first().innerText(), /84.71/);
    assert.match(await first.locator("td").first().innerText(), /1st/);
    await page.locator("#transfer-details").getByLabel("Probe").selectOption("AP");
    assert.match(await first.locator("td").first().innerText(), /83.33/);
    await page.getByLabel("Task family").selectOption("Geometry");
    assert.equal(await rows.count(), 3);
    await rows.last().getByRole("button").click();
    const detail = page.getByRole("dialog");
    assert.match(await detail.innerText(), /MPI Sintel.*EPE.*lower/s);
    assert.equal(await detail.locator("a[target=_blank]").count(), 5);
    await page.keyboard.press("Escape");
    await page.getByLabel("Task family").selectOption("All");
    assert.doesNotMatch(await rows.allTextContents().then(x=>x.join(" ")), /SpaceNet|SUN RGB-D/);
    assert.match(await page.locator("#holds").innerText(), /SpaceNet 2.*SUN RGB-D/);
    await page.setViewportSize({ width:390, height:820 });
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth));
    assert.equal(await page.getByRole("link",{name:"Overview",exact:true}).isVisible(), true);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode=1; });
