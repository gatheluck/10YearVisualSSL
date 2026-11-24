{
// Each host page supplies a heading that fits its part of the paper.
const broadTransfer = document.getElementById("figure-five");
const figureHeading = broadTransfer.querySelector("h2");
figureHeading.insertAdjacentHTML("beforebegin", '<p class="eyebrow">Figure 5 · Broad transfer</p>');
figureHeading.insertAdjacentHTML("afterend", `
  <div class="transfer-facts">
    <p><strong>29<span>/40</span></strong>DINOv3-7B first-place finishes with linear or task-specific frozen probes.</p>
    <p><strong>20<span>/40</span></strong>DINOv3-7B first-place finishes with attentive probes.</p>
    <p><strong>6<span>/9</span></strong>Detection and segmentation settings led by RAEv2 with attentive probes.</p>
  </div>
  <figure>
    <div class="radar-tools">
      <label>Visible profiles <select class="probe-choice"><option>Both</option><option>LP</option><option>AP</option></select></label>
      <label>Inspect task <select class="task-choice"></select></label>
      <button type="button" aria-pressed="false">Enlarge chart</button>
    </div>
    <output aria-live="polite"></output>
    <div class="radar-key"><span>△ LP · dashed</span><span>● AP · solid</span><span>Outer ring = best on this task</span></div>
    <details class="task-key"><summary>Task key · 40 spokes</summary><div></div></details>
    <div class="transfer-pan" role="region" tabindex="0" aria-label="Five interactive radar charts and the task key">${transferVector}</div>
    <figcaption>Each spoke is one task. LP triangles and AP circles are scaled separately. Choose a task or click a mark to compare its raw scores.</figcaption>
  </figure>
  <div class="transfer-foot"><p>The original figure includes all 40 settings. The score explorer covers 38 while the SpaceNet 2 and SUN RGB-D task definitions are under review.</p>
  <a href="extended.html#matrix">Inspect task scores</a></div>`);

const source = broadTransfer.querySelector("svg");
const chart = source.parentElement;
const probeChoice = broadTransfer.querySelector(".probe-choice");
const taskChoice = broadTransfer.querySelector(".task-choice");
const scoreOutput = broadTransfer.querySelector("output");
const marks = [...chart.querySelectorAll(".score-point")];
const panels = [...chart.querySelectorAll(".model-panel")];
for (const name of new Set(marks.map(mark => mark.dataset.dataset))) {
  taskChoice.add(new Option(name));
}

// The paper arranges every panel in one wide SVG. Keep its plotted marks,
 // but give each model an independent viewport and readable HTML headings.
const key = broadTransfer.querySelector(".radar-key");
for (const entry of source.querySelectorAll(".rank-entry")) {
  const label = document.createElement("span");
  label.className = "rank-swatch";
  label.style.setProperty("--swatch", entry.querySelector("circle").getAttribute("fill"));
  label.textContent = entry.querySelector("text").textContent;
  key.append(label);
}
const modelNames = new Map();
const names = [...taskChoice.options].map(option => option.value);
const taskKey = broadTransfer.querySelector(".task-key div");
let group;
for (const entry of source.querySelectorAll(".category-label, .dataset-key")) {
  if (entry.matches(".category-label")) {
    group = document.createElement("section");
    const title = document.createElement("h4");
    title.textContent = entry.textContent;
    title.style.borderColor = entry.getAttribute("fill");
    group.append(title);
    taskKey.append(group);
  } else {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.textContent;
    const number = Number.parseInt(entry.textContent, 10);
    button.addEventListener("click", () => {
      taskChoice.value = names[number - 1];
      updateRadar();
      taskChoice.focus();
    });
    group.append(button);
  }
}
for (const panel of panels) {
  const card = document.createElement("article");
  const heading = document.createElement("h3");
  const [name, family, lp, ap] = panel.querySelectorAll(":scope > text");
  heading.textContent = name.textContent;
  modelNames.set(panel, name.textContent);
  const subtitle = document.createElement("small");
  subtitle.textContent = family.textContent;
  const stats = document.createElement("p");
  stats.textContent = [lp, ap].map(label => label.textContent.trim().replace(/\s*Best \/ Second-best:\s*/, " ")).join("   ·   ");
  stats.prepend("First / second   ");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const center = Number(name.getAttribute("x"));
  svg.setAttribute("viewBox", [center - 212, 254, 424, 424].join(" "));
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", heading.textContent + " transfer profile");
  for (const item of [name, family, lp, ap, panel.querySelector("rect")]) item.remove();
  svg.append(panel);
  card.append(subtitle, heading, svg, stats);
  chart.append(card);
}
source.remove();

function updateRadar() {
  const task = taskChoice.value;
  const probe = probeChoice.value;
  chart.querySelectorAll("[data-probe]").forEach(node => {
    node.style.display = probe === "Both" || node.dataset.probe === probe ? "" : "none";
  });
  for (const mark of marks) mark.classList.toggle("inspected", mark.dataset.dataset === task);
  const heading = document.createElement("strong");
  heading.textContent = task;
  scoreOutput.replaceChildren(heading);
  const entries = marks.filter(mark => mark.dataset.dataset === task);
  if (!entries[0].hasAttribute("data-score")) {
    scoreOutput.append("Task definition and metric pending.");
    return;
  }
  for (const panel of panels) {
    const item = document.createElement("span");
    const label = document.createElement("b");
    label.textContent = modelNames.get(panel);
    item.append(label);
    for (const protocol of ["LP","AP"]) {
      if (probe !== "Both" && probe !== protocol) continue;
      const mark = entries.find(node => panel.contains(node) && node.dataset.probe === protocol);
      item.append(" " + protocol + " " + mark.dataset.score);
    }
    scoreOutput.append(item);
  }
}
chart.addEventListener("click", event => {
  const mark = event.target.closest(".score-point");
  if (mark) { taskChoice.value = mark.dataset.dataset; updateRadar(); }
});
probeChoice.addEventListener("change", updateRadar);
taskChoice.addEventListener("change", updateRadar);
const zoom = broadTransfer.querySelector(".radar-tools button");
zoom.addEventListener("click", () => {
  const enlarged = chart.classList.toggle("enlarged");
  zoom.setAttribute("aria-pressed", String(enlarged));
  zoom.textContent = enlarged ? "Fit chart" : "Enlarge chart";
});
updateRadar();
}
