// The overview summarizes v46 scores with a running best across all families.
(() => {
  const film = document.querySelector(".history-film");
  const svg = film.querySelector("svg");
  const button = film.querySelector("button");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const familyColors = {
    "VSSL": "#00B3B0",
    "3D/4D Geometric Models": "#FF5A5F",
    "Vision Generative Models": "#A044E6",
    "VideoSSL": "#84D447",
    "Supervised Models": "#FFC700",
    "Multimodal World / Video Generative": "#2563EB",
    "VLM, MLLM and Other Models": "#00A2FF"
  };
  const familyKey = film.querySelector(".film-key");
  for (const [family,color] of Object.entries(familyColors)) {
    const item=document.createElement("li");
    item.textContent=family;
    item.style.setProperty("--family-color",color);
    familyKey.append(item);
  }
  const chapters = [
    ["in1k", "Image classification on ImageNet-1K", "Top-1 accuracy (%)"],
    ["coco", "Object detection on COCO 2017", "Figure 2 mAP · mixed protocols"],
    ["ade20k", "Semantic segmentation on ADE20K", "mIoU"],
    ["nyuv2", "Depth estimation on NYUv2", "RMSE"],
    ["ssv2", "Video classification on SSv2", "Top-1 accuracy (%)"],
  ];
  const start = Date.parse("2012-01-01"), end = Date.parse("2026-09-24");
  const sweep = 7000, chapterTime = 9000;
  let elapsed = reduced.matches ? sweep : 0, playing = !reduced.matches;
  let timer, previous, chapter = -1, width, height, records, lower, ceiling;
  let frontier, cursor, next, best, path, dots, labels, annotations;

  function element(tag, attrs, parent = svg, text = "") {
    const node = document.createElementNS(svg.namespaceURI, tag);
    for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,value);
    node.textContent = text;
    parent.append(node);
    return node;
  }
  const x = date => 38 + (width-55)*(date-start)/(end-start);
  const y = value => 12 + (height-43)*(lower ? value/ceiling : 1-value/ceiling);

  function openChapter(index) {
    chapter = index;
    const [key,title,metric] = chapters[index];
    lower = key === "nyuv2";
    records = filmRecords.filter(row => row[key] !== null).map(row => ({
      date:Date.parse(row.date), label:row.date, value:row[key], name:row.name, family:row.family,
    }));
    const step = lower ? 0.5 : 10;
    ceiling = Math.ceil(Math.max(...records.map(r=>r.value))/step)*step;
    film.querySelector("h2").textContent = title;
    film.querySelector(".film-metric").textContent = metric + (lower ? " · lower is better" : " · higher is better");
    width = Math.max(200,svg.clientWidth); height = svg.clientHeight;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const tick = {fill:"#b9d0d1","font-size":11};
    for (let i=0;i<=4;i++) {
      const value=ceiling*i/4;
      element("line",{x1:38,x2:width-17,y1:y(value),y2:y(value),stroke:"#34545b","stroke-dasharray":"3 5"});
      element("text",{x:31,y:y(value)+4,"text-anchor":"end",...tick},svg,String(+value.toFixed(2)));
    }
    for (const year of (width < 500 ? [2012,2016,2020,2026] : [2012,2014,2016,2018,2020,2022,2024,2026])) {
      element("text",{x:x(Date.UTC(year,0)),y:height-7,"text-anchor":"middle",...tick},svg,String(year));
    }
    frontier=element("path",{class:"frontier",fill:"none",stroke:"#00b3b0","stroke-width":2.5});
    dots=element("g",{});
    cursor=element("line",{y1:9,y2:height-30,stroke:"#bee7d9","stroke-opacity":0.6});
    labels=element("g",{});
    annotations=planLabels();
    next=0;best=null;path="";
    film.querySelector(".film-best").textContent="";
  }
  // Reserve space using the complete chapter at the current viewport size.
  // A record with no room for a name still contributes its point and step.
  function planLabels() {
    const daily=new Map();
    for(const r of records) {
      const other=daily.get(r.date);
      if(!other || (lower ? r.value<other.value : r.value>other.value)) daily.set(r.date,r);
    }
    const milestones=[];
    let leader=null;
    for(const r of daily.values()) {
      if(!leader || (lower ? r.value<leader.value : r.value>leader.value)) {
        milestones.push(r);
        leader=r;
      }
    }
    const planned=new Map(milestones.map(r=>[r,null]));
    const placed=[];
    for (const r of milestones.reverse()) {
      const px=x(r.date), py=y(r.value);
      const name=element("text",{class:"milestone-label","font-size":12,"font-weight":650,
        fill:"#effaf7",stroke:"#102d33","stroke-width":3,"paint-order":"stroke"},labels,r.name);
      const length=name.getComputedTextLength();
      let position;
      for (const offset of [-10,19,-26,35]) {
        const left=Math.min(width-length-4,Math.max(38,px+5));
        const base=py+offset;
        if (base<12 || base>height-32) continue;
        const box={left,right:left+length,top:base-12,bottom:base+3};
        if (placed.some(p=>p.left<box.right+5 && box.left<p.right+5 && p.top<box.bottom+2 && box.top<p.bottom+2)) continue;
        position={left,base};
        placed.push(box);
        break;
      }
      if (!position) {name.remove();continue;}
      name.setAttribute("x",position.left);
      name.setAttribute("y",position.base);
      const leader=element("line",{x1:px,y1:py,x2:Math.max(position.left,Math.min(px,position.left+length)),
        y2:position.base+(position.base>py?-11:3),stroke:"#bbd6d5","stroke-width":0.8},labels);
      const pair=document.createDocumentFragment();
      pair.append(leader,name);
      planned.set(r,pair);
    }
    return planned;
  }
  function render() {
    const index=Math.floor(elapsed/chapterTime)%chapters.length;
    if(index!==chapter) openChapter(index);
    const date=start+(end-start)*Math.min(1,(elapsed%chapterTime)/sweep);
    const at=x(date);
    while(next<records.length && records[next].date<=date) {
      const r=records[next++], px=x(r.date), py=y(r.value);
      element("circle",{cx:px,cy:py,r:3.1,fill:familyColors[r.family],"data-date":r.label},dots);
      if (annotations.has(r)) {
        best=r;
        path+=path ? `H${px}V${py}` : `M${px} ${py}`;
        const annotation=annotations.get(r);
        if(annotation) labels.append(annotation);
        film.querySelector(".film-best").textContent=r.name+" · "+r.value;
      }
    }
    frontier.setAttribute("d",best ? path+"H"+at : "");
    cursor.setAttribute("x1",at);cursor.setAttribute("x2",at);
    film.querySelector(".film-year").textContent=new Date(date).getUTCFullYear();
  }
  function tick() {
    const now=performance.now();
    elapsed+=now-previous;previous=now;render();
  }
  function schedule() {
    clearInterval(timer);
    if(playing && !document.hidden) {previous=performance.now();timer=setInterval(tick,50);}
    button.textContent=playing?"Pause motion":"Play motion";
    render();
  }
  button.hidden=false;
  button.onclick=()=>{
    if(!playing && reduced.matches && elapsed===sweep) {elapsed=0;chapter=-1;}
    playing=!playing;
    schedule();
  };
  reduced.addEventListener("change",()=>{playing=!reduced.matches;elapsed=playing?0:sweep;chapter=-1;schedule();});
  document.addEventListener("visibilitychange",schedule);
  new ResizeObserver(()=>{if(svg.clientWidth!==width || svg.clientHeight!==height){chapter=-1;render();}}).observe(svg);
  schedule();
})();
