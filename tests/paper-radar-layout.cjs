const assert = require("node:assert/strict");
const {chromium} = require("playwright");
const {pathToFileURL} = require("node:url");
const path = require("node:path");
(async()=>{
 const b=await chromium.launch({executablePath:process.env.CHROME_PATH});
 try {
  const p=await b.newPage({viewport:{width:1280,height:900}});
  await p.goto(process.env.BOARD_HOME || pathToFileURL(path.resolve(__dirname,"../paper-site/index.html")).href);
  const section=p.locator("#figure-five");
  await section.screenshot({path:"/tmp/web-radar-desktop.png"});
  await p.setViewportSize({width:390,height:844});
  await section.screenshot({path:"/tmp/web-radar-mobile.png"});
  const cards=section.locator("article");
  assert.equal(await cards.count(),5,"Five independently reflowing model cards");
  assert.equal(await section.locator("svg #headline, svg .dataset-column").count(),0);
  assert.ok(await cards.first().evaluate(el=>el.clientWidth>=250));
  assert.ok(await section.locator(".transfer-pan").evaluate(el=>el.scrollWidth<=el.clientWidth));
  await section.getByText("Task key", {exact:false}).filter({hasNot:p.locator("summary")}).first().click();
  const buttons=section.locator("details button");
  assert.equal(await buttons.count(),40);
  await buttons.last().click();
  assert.match(await section.locator("output").innerText(),/CrowdPose/);
  await p.setViewportSize({width:1280,height:900});
  const rects=await cards.evaluateAll(els=>els.map(el=>el.getBoundingClientRect().top));
  assert.equal(rects[0],rects[1]);
  assert.ok(rects[4]>=rects[2]);
 } finally {await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
