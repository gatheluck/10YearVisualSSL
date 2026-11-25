const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ reducedMotion: "no-preference" });
    // Returning readers may have the data script from before the shared color helper.
    await page.route("**/history-data.js", async route => {
      const fs = require("node:fs");
      const current = fs.readFileSync(path.resolve(__dirname, "../paper-site/history-data.js"), "utf8");
      await route.fulfill({contentType:"text/javascript", body:current.split("const taskRankColor =")[0]});
    });
    await page.goto(process.env.HERO_URL || pathToFileURL(path.resolve(__dirname, "../paper-site/index.html")).href);
    const hero = page.locator(".paper-cover");
    const models = hero.getByRole("link",{name:/113 historical model configurations/i});
    const settings = hero.getByRole("link",{name:/40 extended benchmark settings/i});
    assert.equal(await models.count(),1,"The first view must introduce the historical model collection");
    assert.equal(await settings.count(),1);
    assert.equal(await models.getAttribute("href"),"history.html");
    assert.equal(await settings.getAttribute("href"),"extended.html#figure-five");
    assert.match(await settings.getAttribute("aria-label"), /5 selected models/);
    assert.equal(+(await models.locator("b").textContent()),await page.evaluate(()=>historicalScores.length));
    assert.equal(+(await settings.locator("b").textContent()),await page.locator(".task-choice option").count());
    const resources = hero.getByRole("navigation",{name:"Reproduction resources"});
    const experiments = hero.getByText("2,500+",{exact:true});
    const positions = async () => Promise.all([models.locator("b"), settings.locator("b"), experiments].map(el=>el.boundingBox()));
    for (const width of [1440,610,390,320]) {
      await page.setViewportSize({width,height:900});
      const strip = await hero.locator(".study-totals").boundingBox();
      assert.ok(strip && strip.height <= 72,"Compact scope strip at "+width);
      const boxes = await positions();
      assert.ok(Math.max(...boxes.map(b=>b.y))-Math.min(...boxes.map(b=>b.y)) < 3, "Three study totals share a baseline at "+width);
      assert.ok(boxes[0].x < boxes[1].x && boxes[1].x < boxes[2].x);
    }
    assert.doesNotMatch(await hero.innerText(), /core benchmarks/i);
    for (const dataset of ["ImageNet","COCO","ADE20K","NYUv2","SSv2"]) assert.ok((await page.locator("#geometry").innerText()).includes(dataset));
    for (const [label,path] of [["Source code",""],["Training recipes","/tree/main/methods"],["Checkpoint guide","/blob/main/docs/STEP1_WEIGHTS.md"]]) {
      assert.equal(await resources.getByRole("link",{name:label}).getAttribute("href"),"https://github.com/gatheluck/10YearVisualSSL"+path);
    }
    assert.equal(await page.locator(".cover-matrix rect").count(),545);
    const plot = hero.locator(".history-film svg");
    assert.equal(await plot.count(), 1, "Overview must play the historical chart");
    await page.setViewportSize({width:1100,height:900});
    // Keep the entire drawing in the viewport across focus and play-state changes.
    await plot.evaluate(node => node.scrollIntoView({block:"center",behavior:"instant"}));
    const frame = () => plot.screenshot({ animations: "allow" });
    const changes = async () => {
      const before = await frame();
      await page.waitForTimeout(400);
      const after = await frame();
      if (before.equals(after)) return false;
      // Chromium may rerasterize a few antialiased edge pixels in a stationary SVG.
      return page.evaluate(async images => {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        const pixels = [];
        for (const source of images) {
          const image = new Image();
          image.src = source;
          await image.decode();
          canvas.width = image.width;
          canvas.height = image.height;
          context.drawImage(image,0,0);
          pixels.push(context.getImageData(0,0,image.width,image.height).data);
        }
        let changed = 0;
        for (let i=0; i<pixels[0].length; i+=4) {
          if (Math.abs(pixels[0][i+1] - pixels[1][i+1]) > 12) changed++;
        }
        return changed > canvas.height / 2;
      }, [before,after].map(buffer => "data:image/png;base64," + buffer.toString("base64")));
    };
    await page.waitForTimeout(4100);
    assert.ok(await changes(), "The visible plot must still move after the opening");
    await hero.getByRole("button", {name:"Pause motion"}).click();
    const frozen = await plot.innerHTML();
    assert.ok(!(await changes()), "Pausing freezes the actual plot");
    assert.equal(await plot.innerHTML(), frozen, "Every SVG coordinate must remain exactly fixed");
    await hero.getByRole("button", {name:"Play motion"}).click();
    assert.ok(await changes());
    await page.emulateMedia({reducedMotion:"reduce"});
    await hero.getByRole("button", {name:"Play motion"}).waitFor();
    assert.ok(!(await changes()), "Reduced motion starts with a static plot");
    await hero.getByRole("button", {name:"Play motion"}).click();
    assert.ok(await changes(), "Readers may explicitly enable motion");
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({width, height:900});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      const last = await settings.boundingBox();
      assert.ok(last.y + last.height < 900, "Scope visible at " + width);
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
