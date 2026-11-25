const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const { pathToFileURL } = require("node:url");
const path = require("node:path");
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage();
    const base = process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href;
    for (const file of ["index.html", "leaderboard.html"]) {
      await page.goto(new URL(file, base).href);
      const feature = page.locator("#figure-five");
      assert.equal(await feature.locator("figure svg .model-panel").count(), 5, "Five live radar panels");
      assert.equal(await feature.locator(".profile").count(), 10);
      assert.equal(await feature.locator(".score-point").count(), 400);
      assert.equal(await feature.locator("figure img").count(), 0);
      const values = feature.locator("output .model-result");
      assert.equal(await values.count(),5,"Present each model's measured scores together");
      assert.match(await feature.locator("output").innerText(),/top-1.*higher/i);
      assert.deepEqual(await values.first().locator("dd").allTextContents(),["84.71","83.33"]);
      const taskPanel = feature.locator(".task-readout");
      assert.equal(await taskPanel.getByLabel("Inspect task").count(),1,"The task picker belongs to the score panel");
      const picker = taskPanel.getByLabel("Inspect task");
      for (const width of [1280, 390, 320]) {
        await page.setViewportSize({width,height:850});
        await picker.selectOption("Cambridge Landmarks");
        const size = await picker.evaluate(el => ({
          font: parseFloat(getComputedStyle(el).fontSize),
          height: el.getBoundingClientRect().height,
          right: el.getBoundingClientRect().right
        }));
        assert.ok(size.font >= 16 && size.font <= 18, "Use a compact, readable selected value");
        assert.ok(await picker.evaluate(el => {
          const style = getComputedStyle(el);
          const context = document.createElement("canvas").getContext("2d");
          context.font = [style.fontWeight, style.fontSize, style.fontFamily].join(" ");
          return [...el.options].every(option => context.measureText(option.text).width <= el.clientWidth - 32);
        }), "Every dataset name fits beside the native disclosure arrow");
        assert.ok(size.height >= 44 && size.right <= width, "Preserve a usable touch target");
      }
      await picker.selectOption("ImageNet-10%");
      const profiles = feature.getByRole("group",{name:"Visible profiles"});
      for (const [name, probe] of [[/Attentive/, "AP"], [/Linear/, "LP"]]) {
        await profiles.getByRole("radio", {name}).check();
        const points = feature.locator(`.score-point[data-model="dino"][data-probe="${probe}"]`);
        const point = points.nth(1);
        await point.scrollIntoViewIfNeeded();
        await page.mouse.move(0, 0);
        const baseWidth = (await point.boundingBox()).width;
        const neighborWidth = (await points.first().boundingBox()).width;
        await point.hover();
        assert.ok((await point.boundingBox()).width > baseWidth * 1.15, "Hover enlarges the individual mark");
        assert.equal((await points.first().boundingBox()).width, neighborWidth);
        assert.equal(await picker.inputValue(), "ImageNet-10%");
        await page.mouse.move(0, 0);
        assert.equal((await point.boundingBox()).width, baseWidth);
        await point.click();
        assert.equal(await picker.inputValue(), await point.getAttribute("data-dataset"));
        await page.mouse.move(0, 0);
        assert.equal(await point.evaluate(el => getComputedStyle(el).transform), "none");
        await picker.selectOption("ImageNet-10%");
      }
      assert.equal(await values.locator("dd").count(),5);
      assert.equal(await feature.locator(".score-point:visible").count(), 200);
      assert.equal(await feature.locator(".profile:visible").count(), 5);
      await feature.getByLabel("Inspect task").selectOption("MPI Sintel");
      assert.match(await feature.locator("output").innerText(), /MPI Sintel.*DINOv3-7B/s);
      assert.match(await feature.locator("output").innerText(),/EPE.*lower/i);
      const mark=feature.locator('.score-point[data-dataset="ImageNet-10%"][data-probe="LP"][data-model="dino"]');
      await mark.dispatchEvent("click");
      assert.match(await feature.locator("output").innerText(), /ImageNet-10%.*84.71/s);
      await feature.getByLabel("Inspect task").selectOption("SUN RGB-D");
      assert.match(await feature.locator("output").innerText(), /pending/i);
      assert.equal(await feature.locator('.score-point[data-dataset="SUN RGB-D"][data-score]').count(),0);
      await page.setViewportSize({width:390,height:844});
      await feature.getByRole("button",{name:"Enlarge chart"}).click();
      assert.ok(await feature.locator(".transfer-pan").evaluate(e=>e.scrollWidth>e.clientWidth));
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await feature.getByRole("link", {name:"Inspect task scores"}).click();
      await page.waitForURL("**/leaderboard.html#matrix");
      assert.equal(await page.locator("#matrix tbody tr").count(), 38);
    }
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
