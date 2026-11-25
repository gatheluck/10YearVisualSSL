const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch({executablePath:process.env.CHROME_PATH});
  try {
    const page = await browser.newPage({reducedMotion:"no-preference"});
    await page.clock.install();
    await page.clock.pauseAt(new Date(Date.now()+1500));
    for(const width of [390,1280]) {
    await page.setViewportSize({width,height:900});
    await page.goto(pathToFileURL(path.resolve(__dirname,"../paper-site/index.html")).href);
    const film = page.locator(".history-film");
    assert.equal(await film.count(),1,"Provide an autoplay history in the cover");
    const points = film.locator("circle");
    const palette = {
      "VSSL":"#00B3B0", "3D/4D Geometric Models":"#FF5A5F", "Vision Generative Models":"#A044E6",
      "VideoSSL":"#84D447", "Supervised Models":"#FFC700", "Multimodal World / Video Generative":"#2563EB",
      "VLM, MLLM and Other Models":"#00A2FF",
    };
    const legend = film.getByRole("list",{name:"Model families"});
    assert.equal(await legend.count(),1,"The family key belongs to this chart");
    const swatches = await legend.locator("li").evaluateAll(items=>items.map(item=>[item.textContent,getComputedStyle(item,"::before").backgroundColor]));
    const rgb = hex => "rgb("+hex.match(/../g).map(pair=>parseInt(pair,16)).join(", ")+")";
    assert.deepEqual(swatches,Object.entries(palette).map(([family,hex])=>[family,rgb(hex.slice(1))]));
    assert.doesNotMatch(await film.innerText(), /best across all configurations/i);
    for (const [name,count] of [["Image classification",112],["Segmentation",112],["Depth estimation",112],["Video classification",113]]) {
      assert.equal(await film.locator("h2").innerText(),name);
      assert.ok(await points.count()<10);
      let previous = {};
      for(let step=0;step<36;step++) {
        await page.clock.runFor(200);
        const current=await film.locator(".milestone-label").evaluateAll(nodes=>Object.fromEntries(nodes.map(n=>{
          const leader=n.previousElementSibling;
          return [n.textContent,[n.getAttribute("x"),n.getAttribute("y"),...["x1","y1","x2","y2"].map(attr=>leader.getAttribute(attr))]];
        })));
        for(const [model,coords] of Object.entries(previous)) assert.deepEqual(current[model],coords,model+" label and leader must stay fixed during "+name);
        previous=current;
      }
      assert.equal(await points.count(),count);
      const key={"Image classification":"in1k","Segmentation":"ade20k","Depth estimation":"nyuv2","Video classification":"ssv2"}[name];
      const families=await page.evaluate(task=>filmRecords.filter(r=>r[task]!==null).map(r=>r.family),key);
      assert.deepEqual(await points.evaluateAll(ns=>ns.map(n=>getComputedStyle(n).fill)),families.map(family=>rgb(palette[family].slice(1))));
      const dates = await points.evaluateAll(ns=>ns.map(n=>n.dataset.date));
      assert.deepEqual(dates,[...dates].sort());
      assert.ok(await film.locator("path.frontier").getAttribute("d"));
      const expectedBest = {"Image classification":"DINOv3-7B","Segmentation":"RAEv2","Depth estimation":"π³","Video classification":"V-JEPA"};
      const errors = await film.evaluate(node=>{
        const path=node.querySelector("path.frontier");
        const points=[...node.querySelectorAll("circle")].map(c=>[c.cx.baseVal.value,c.cy.baseVal.value]);
        const dates=[...new Set(points.map(p=>p[0]))];
        return dates.filter(x=>{
          let lo=0, hi=path.getTotalLength();
          for(let i=0;i<25;i++){
            const mid=(lo+hi)/2;
            if(path.getPointAtLength(mid).x<x+0.1) lo=mid; else hi=mid;
          }
          const expectedY=Math.min(...points.filter(p=>p[0]<=x).map(p=>p[1]));
          return Math.abs(path.getPointAtLength(hi).y-expectedY)>0.2;
        });
      });
      assert.equal(errors.length,0,"The staircase must meet the best visible score on every release date: "+name);
      const labels=await film.locator("svg .milestone-label").allTextContents();
      assert.ok(labels.length>=2 && labels.includes(expectedBest[name]),"Show historical leaders and the final leader inside the graph");
      if(name==="Depth estimation") assert.match(await film.innerText(),/lower is better/);
      await page.clock.runFor(1800);
    }
    assert.equal(await film.locator("h2").innerText(),"Image classification");
    }
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
