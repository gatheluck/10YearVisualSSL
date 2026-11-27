const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

const profiles = [
  ["Hirokatsu Kataoka", "https://hirokatsukataoka.net/"],
  ["Kohsuke Ide", "https://kohsukeide.github.io/"],
  ["Yoshihiro Fukuhara", "https://gatheluck.net/home/"],
  ["Shumpei Takezaki", "https://shumpei-takezaki.github.io/"],
  ["Nakamasa Inoue", "https://strdb.s.isct.ac.jp/html/100001739_en.html"],
  ["Yuki M Asano", "https://yukimasano.github.io/"],
  ["Christian Rupprecht", "https://www.cs.ox.ac.uk/people/christian.rupprecht/"],
];

const frontia = "https://frontia-ai.go.jp/";
const site = pathToFileURL(path.resolve(__dirname, "../paper-site") + path.sep);

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    for (const width of [320, 390, 1280]) {
      const page = await browser.newPage({
        viewport: { width, height: 900 },
        hasTouch: width < 1280,
        isMobile: width < 1280,
        javaScriptEnabled: false,
      });
      await page.goto(new URL("index.html", site).href);
      const byline = page.getByRole("list", { name: "Authors", exact: true });
      assert.deepEqual(await byline.getByRole("link").allTextContents(), profiles.map(([name]) => name));
      for (const [name, url] of profiles) {
        const link = byline.getByRole("link", { name, exact: true });
        assert.equal(await link.getAttribute("href"), url);
        const box = await link.boundingBox();
        assert.ok(box.x >= 0 && box.x + box.width <= width, `${name} is clipped at ${width}px`);
      }

      const [firstName, firstUrl] = profiles[0];
      const first = byline.getByRole("link", { name: firstName, exact: true });
      await page.route(firstUrl, route => route.fulfill({ body: firstName }));
      if (width < 1280) {
        await first.tap();
      } else {
        const decoration = await first.evaluate(el => getComputedStyle(el).textDecorationThickness);
        await first.hover();
        assert.notEqual(await first.evaluate(el => getComputedStyle(el).textDecorationThickness), decoration);
        await page.mouse.move(0, 0);
        await page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: "Leaderboard" }).focus();
        await page.keyboard.press("Tab");
        assert.ok(await first.evaluate(el => el === document.activeElement && el.matches(":focus-visible")));
        assert.notEqual(await first.evaluate(el => getComputedStyle(el).outlineStyle), "none");
        await page.keyboard.press("Enter");
      }
      await page.waitForURL(firstUrl, { timeout: 5000 });

      await page.goto(new URL("leaderboard.html", site).href);
      const credits = page.getByRole("link", { name: "Authors and related project" });
      await credits.click();
      assert.equal(page.url(), new URL("index.html#authors", site).href);
      const authors = page.getByRole("region", { name: "Authors and related project" });
      assert.ok(await authors.isVisible());
      assert.equal(await authors.getByRole("link", { name: "FRONTia", exact: true }).getAttribute("href"), frontia);
      assert.ok(await authors.getByText("Related project", { exact: true }).isVisible());
      assert.equal(await authors.locator("img").count(), 0);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.close();
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
