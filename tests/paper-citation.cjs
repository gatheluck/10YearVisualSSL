const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    await page.goto(process.env.OVERVIEW_URL || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href);
    const section = page.locator("main > section:last-child");
    assert.equal(await section.getByRole("heading", { name: "Citation" }).count(), 1);
    await page.getByRole("link", { name: "Cite this paper" }).click();
    assert.equal(new URL(page.url()).hash, "#citation");
    const entry = await section.locator("pre code").textContent();
    const title = await page.locator("h1").innerText();
    assert.ok(entry.includes(title.replace(/\s+/g, " ")));
    const authors = entry.match(/author\s*=\s*\{([^}]+)\}/)[1].split(/\s+and\s+/);
    assert.deepEqual(authors, await page.locator(".paper-authors li").allTextContents());
    assert.ok((await section.innerText()).includes("\\cite{" + entry.match(/@\w+\{([^,]+)/)[1] + "}"));
    await page.evaluate(() => Object.defineProperty(navigator, "clipboard", { configurable: true, value: {
      async writeText(text) { window.copiedText = text; }
    }}));
    const copy = section.getByRole("button", { name: "Copy BibTeX" });
    await copy.click();
    assert.equal(await page.evaluate(() => window.copiedText), entry);
    await page.evaluate(() => navigator.clipboard.writeText = async () => { throw new Error("Denied"); });
    await copy.click();
    assert.equal(await page.evaluate(() => getSelection().toString()), entry);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
