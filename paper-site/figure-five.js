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
    <div class="transfer-pan" role="region" tabindex="0" aria-label="Five interactive radar charts and the task key">${transferVector}</div>
    <figcaption>Each spoke is one task. LP triangles and AP circles are scaled separately. Choose a task or click a mark to compare its raw scores.</figcaption>
    <output aria-live="polite"></output>
  </figure>
  <div class="transfer-foot"><p>The original figure includes all 40 settings. The score explorer covers 38 while the SpaceNet 2 and SUN RGB-D task definitions are under review.</p>
  <a href="extended.html#matrix">Inspect task scores</a></div>`);

const chart = broadTransfer.querySelector("svg");
const probeChoice = broadTransfer.querySelector(".probe-choice");
const taskChoice = broadTransfer.querySelector(".task-choice");
const scoreOutput = broadTransfer.querySelector("output");
const marks = [...chart.querySelectorAll(".score-point")];
const panels = [...chart.querySelectorAll(".model-panel")];
for (const name of new Set(marks.map(mark => mark.dataset.dataset))) {
  taskChoice.add(new Option(name));
}

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
    label.textContent = panel.querySelector("text").textContent;
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
