const assert = require("node:assert/strict");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");
(async()=>{
  const browser=await chromium.launch({executablePath:process.env.CHROME_PATH});
  try {
    const page=await browser.newPage();
    const base=process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname,"../paper-site/index.html")).href;
    for(const [file,active] of [["index","Overview"],["history","Leaderboard"],["control","Leaderboard"],["extended","Leaderboard"],["paper","Paper"]]){
      await page.goto(new URL(file+".html",base).href);
      assert.deepEqual(await page.locator("header nav [aria-current]").allTextContents(),[active]);
      for(const width of [390,1280]){
        await page.setViewportSize({width,height:850});
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),file+" viewport");
      }
    }
    await page.goto(base);
    await page.getByRole("link",{name:"Explore leaderboard",exact:true}).click();
    await page.locator('header nav a[href="#leaderboard"][aria-current]').waitFor();
    assert.equal(await page.locator("header nav [aria-current]").innerText(),"Leaderboard");
    await page.locator(".opening-copy").getByRole("link",{name:"Read the study"}).click();
    assert.match(page.url(),/paper\.html$/);
    await page.getByRole("link",{name:"Overview",exact:true}).click();
    const routes=page.getByRole("navigation",{name:"Leaderboard comparisons"});
    assert.ok(await routes.evaluate(el=>Boolean(el.compareDocumentPosition(document.querySelector("#figure-five"))&Node.DOCUMENT_POSITION_FOLLOWING)));
    for(const link of await routes.getByRole("link").all()) {
      const target=await link.getAttribute("href");
      await link.click();
      await page.waitForURL(new URL(target,base).href);
      await page.goBack();
    }
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
