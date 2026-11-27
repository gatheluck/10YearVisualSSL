const assert = require("node:assert/strict");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");
(async()=>{
  const browser=await chromium.launch({executablePath:process.env.CHROME_PATH});
  try {
    const page=await browser.newPage();
    const base=process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname,"../paper-site/index.html")).href;
    for(const [file,active] of [["index","Overview"],["history","Leaderboard"],["control","Leaderboard"],["extended","Leaderboard"],["leaderboard","Leaderboard"]]){
      await page.goto(new URL(file+".html",base).href);
      assert.deepEqual(await page.locator("header nav [aria-current]").allTextContents(),[active]);
      for(const width of [390,1280]){
        await page.setViewportSize({width,height:850});
        for (const link of await page.locator("header nav a").all()) {
          const target=await link.boundingBox();
          assert.ok(target.height>=32,file+" navigation hit area at "+width);
        }
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),file+" viewport");
      }
    }
    await page.goto(base);
    await page.getByRole("navigation",{name:"Primary"}).getByRole("link",{name:"Leaderboard"}).click();
    await page.waitForURL("**/leaderboard.html");
    assert.equal(await page.locator("header nav [aria-current]").innerText(),"Leaderboard");
    const routes=page.getByRole("navigation",{name:"Leaderboard comparisons"});
    assert.ok(await routes.evaluate(el=>Boolean(el.compareDocumentPosition(document.querySelector("#figure-five"))&Node.DOCUMENT_POSITION_FOLLOWING)));
    for(const link of await routes.getByRole("link").all()) {
      const target=await link.getAttribute("href");
      assert.ok(target.startsWith("#"), "Comparison must be inside this document: "+target);
      assert.equal(await page.locator(target).count(),1);
      await link.click();
      await page.waitForURL(new URL(target,new URL("leaderboard.html",base)).href);
      await page.goBack();
    }
    for (const [oldRoute, fragment] of [
      ["history.html#scores", "#scores"], ["control.html", "#controlled"],
      ["extended.html#matrix", "#matrix"]
    ]) {
      await page.goto(new URL(oldRoute, base).href);
      await page.waitForURL("**/leaderboard.html" + fragment);
    }
    await page.goto(new URL("leaderboard.html#scores", base).href);
    assert.equal(await page.locator("#scores").getAttribute("aria-pressed"), "true");
    await page.locator("#ranking").click();
    await page.goto(new URL("leaderboard.html#scores",base).href);
    await page.locator('#scores[aria-pressed="true"]').waitFor({timeout: 3000});
    assert.equal(await page.locator("#scores").getAttribute("aria-pressed"), "true");
    const duplicates = await page.locator("[id]").evaluateAll(nodes => {
      const ids = nodes.map(node => node.id);
      return ids.filter((id, index) => ids.indexOf(id) !== index);
    });
    assert.deepEqual(duplicates, []);
    await page.locator("#year").selectOption("2013");
    const origin = await page.evaluate(() => performance.timeOrigin);
    await routes.getByRole("link", {name: "ASIS vs CTRL"}).click();
    await routes.getByRole("link", {name: "History", exact: true}).click();
    assert.equal(await page.locator("#year").inputValue(), "2013");
    assert.equal(await page.evaluate(() => performance.timeOrigin), origin);
    for(const fragment of ["#unified?protocol=LoRA&task=all","#experiments","#leaderboard"]) {
      await page.goto(new URL("index.html"+fragment,base).href);
      await page.waitForURL(new URL("leaderboard.html"+fragment,base).href);
    }
    await page.goto(new URL("paper.html#geometry",base).href);
    await page.waitForURL("**/index.html#geometry");
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
