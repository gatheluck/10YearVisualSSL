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
  const tables=cards.getByRole("table",{name:"Placements across 40 tasks"});
  assert.equal(await tables.count(),5,"Every model labels its task counts");
  const reference=[[[29,6],[20,12]],[[1,2],[0,1]],[[2,16],[9,17]],[[1,2],[3,2]],[[7,14],[8,8]]];
  for (const [index, table] of (await tables.all()).entries()) {
    assert.deepEqual(await table.getByRole("columnheader").allTextContents(),["Probe","1st place","2nd place"]);
    assert.deepEqual(await table.getByRole("rowheader").allTextContents(),["Linear (LP)","Attentive (AP)"]);
    assert.deepEqual((await table.getByRole("cell").allTextContents()).map(Number),reference[index].flat());
  }
  assert.equal(await section.locator("svg #headline, svg .dataset-column").count(),0);
  assert.ok(await cards.first().evaluate(el=>el.clientWidth>=250));
  assert.ok(await section.locator(".transfer-pan").evaluate(el=>el.scrollWidth<=el.clientWidth));
  const buttons=section.locator(".task-key button:visible");
  assert.equal(await buttons.count(),40,"Every task name is visible without opening a disclosure");
  assert.equal(await section.locator(".task-key details, details.task-key").count(),0);
  assert.equal(await section.locator(".axis-groups").count(),0,"Task legend appears below the charts only");
  const expected = ["rgb(0, 179, 176)", "rgb(160, 68, 230)", "rgb(255, 90, 95)", "rgb(132, 212, 71)", "rgb(255, 199, 0)", "rgb(0, 162, 255)"];
  const swatches = await section.locator(".task-key h4").evaluateAll(headings => headings.map(h => getComputedStyle(h).borderLeftColor));
  assert.deepEqual(swatches,expected);
  for (const card of await cards.all()) {
    const bands = await card.locator(".category-band").evaluateAll(paths =>
      paths.map(path => [getComputedStyle(path).fill, getComputedStyle(path).fillOpacity]));
    assert.deepEqual(bands, swatches.map(color => [color, "1"]));
  }
  await buttons.last().click();
  assert.match(await section.locator("output").innerText(),/CrowdPose/);
  for(const width of [1280,1440,1920]) {
    await p.setViewportSize({width,height:900});
    const rects=await cards.evaluateAll(els=>els.map(el=>el.getBoundingClientRect()));
    assert.equal(new Set(rects.map(r=>r.top)).size,1,"Five plots share a row at "+width);
    assert.ok(rects[4].right<=width);
  }
 } finally {await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
