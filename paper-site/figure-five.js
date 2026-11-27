{
// Each host page supplies a heading that fits its part of the paper.
const broadTransfer = document.getElementById("figure-five");
const figureHeading = broadTransfer.querySelector("h2");
const scoresPage = document.getElementById("matrix") ? "" : "leaderboard.html";
figureHeading.insertAdjacentHTML("beforebegin", '<p class="eyebrow">Figure 5 · Broad transfer</p>');
figureHeading.insertAdjacentHTML("afterend", `
  <div class="transfer-facts">
    <p><strong>29<span>/40</span></strong>DINOv3-7B leads with frozen-feature probing (LP)</p>
    <p><strong>20<span>/40</span></strong>DINOv3-7B leads with attentive probing (AP)</p>
    <p><strong>6<span>/9</span></strong>RAEv2 leads in detection and segmentation with attentive probing</p>
  </div>
  <figure>
    <div class="task-readout">
      <label class="task-heading">Inspect task <select class="task-choice"></select></label>
      <output aria-live="polite"></output>
    </div>
    <div class="radar-comparison">
      <p class="radar-guide">40 tasks clockwise from the top. Scores are scaled to the best of the five models for each task and probe. The outer ring marks that best score.</p>
      <div class="transfer-foot">
        <a href="${scoresPage}#matrix">Inspect task scores</a>
        <a href="${scoresPage}#holds">2 settings pending</a>
      </div>
      <div class="plot-toolbar">
        <fieldset class="profile-switch"><legend class="sr">Visible profiles</legend>
          <label><input type="radio" name="radar-probe" value="Both" checked><span>Both</span></label>
          <label><input type="radio" name="radar-probe" value="LP"><span><svg viewBox="0 0 32 16" aria-hidden="true"><path d="M0 8h32" stroke="#56616f" stroke-dasharray="4 2"/><path d="m16 3 5 9H11Z" fill="#56616f"/></svg>Linear (LP)</span></label>
          <label><input type="radio" name="radar-probe" value="AP"><span><svg viewBox="0 0 32 16" aria-hidden="true"><path d="M0 8h32" stroke="#267bb3"/><circle cx="16" cy="8" r="4" fill="#267bb3"/></svg>Attentive (AP)</span></label>
        </fieldset>
        <div class="radar-key" aria-label="Rank markers"></div>
        <button class="zoom-radar" type="button" aria-pressed="false">Enlarge chart</button>
      </div>
      <div class="transfer-pan" role="region" tabindex="0" aria-label="Five interactive radar charts">${transferVector}</div>
    </div>
    <section class="task-key" aria-label="Numbered task key"><h3>Task key</h3><div></div></section>
  </figure>`);

const source = broadTransfer.querySelector(".transfer-pan svg");
const chart = source.parentElement;
const profileSwitch = broadTransfer.querySelector(".profile-switch");
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
const names = [...taskChoice.options].map(option => option.value);
const taskKey = broadTransfer.querySelector(".task-key div");
let group;
for (const entry of source.querySelectorAll(".category-label, .dataset-key")) {
  if (entry.matches(".category-label")) {
    group = document.createElement("section");
    const title = document.createElement("h4");
    // The four datasets grouped under "Detection" use semantic mIoU.
    title.textContent = entry.textContent === "Detection" ? "Semantic segmentation" : entry.textContent;
    const category = entry.nextElementSibling.dataset.category;
    const band = source.querySelector(`.category-band[data-category="${category}"]`);
    title.style.borderColor = band.getAttribute("fill");
    group.append(title);
    taskKey.append(group);
  } else {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.textContent;
    const number = Number.parseInt(entry.textContent, 10);
    if (!marks.find(mark => mark.dataset.dataset === names[number - 1]).hasAttribute("data-score")) button.append(" · pending");
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
  const subtitle = document.createElement("small");
  subtitle.textContent = family.textContent;
  const stats = document.createElement("table");
  stats.className = "placement-counts";
  stats.innerHTML = `<caption>Placements across 40 tasks</caption>
    <thead><tr><th scope="col">Probe</th><th scope="col">1st place</th><th scope="col">2nd place</th></tr></thead>`;
  const rows = stats.createTBody();
  for (const label of [lp, ap]) {
    const [, probe, first, second] = label.textContent.trim().match(/^(LP|AP)\s+Best \/ Second-best:\s*(\d+) \/ (\d+)$/);
    const row = rows.insertRow();
    const method = document.createElement("th");
    method.scope = "row";
    method.textContent = probe === "LP" ? "Linear (LP)" : "Attentive (AP)";
    row.append(method);
    row.insertCell().textContent = first;
    row.insertCell().textContent = second;
  }
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
  const probe = profileSwitch.querySelector(":checked").value;
  chart.querySelectorAll("[data-probe]").forEach(node => {
    node.style.display = probe === "Both" || node.dataset.probe === probe ? "" : "none";
  });
  for (const mark of marks) mark.classList.toggle("inspected", mark.dataset.dataset === task);
  const heading = document.createElement("strong");
  heading.textContent = task;
  scoreOutput.replaceChildren(heading);
  const row = transferRows.find(entry => entry.task === task);
  if (!row) {
    scoreOutput.append("Task definition and metric pending.");
    return;
  }
  const metric = document.createElement("small");
  metric.textContent = row.metric + " · " + row.direction + " is better";
  heading.append(metric);
  const grid = document.createElement("span");
  grid.className = "model-results";
  scoreOutput.append(grid);
  transferModels.forEach((model, index) => {
    const item = document.createElement("span");
    item.className = "model-result";
    const label = document.createElement("b");
    label.textContent = model;
    const scores = document.createElement("dl");
    item.append(label, scores);
    for (const protocol of ["LP", "AP"]) {
      if (probe !== "Both" && probe !== protocol) continue;
      const pair = document.createElement("div");
      const method = document.createElement("dt");
      method.textContent = protocol;
      const value = document.createElement("dd");
      value.textContent = row[protocol][index];
      scores.append(pair);
      pair.append(method, value);
    }
    grid.append(item);
  });
}
chart.addEventListener("click", event => {
  const mark = event.target.closest(".score-point");
  if (mark) { taskChoice.value = mark.dataset.dataset; updateRadar(); }
});
profileSwitch.addEventListener("change", updateRadar);
taskChoice.addEventListener("change", updateRadar);
const zoom = broadTransfer.querySelector(".zoom-radar");
zoom.addEventListener("click", () => {
  const enlarged = chart.classList.toggle("enlarged");
  zoom.setAttribute("aria-pressed", String(enlarged));
  zoom.textContent = enlarged ? "Fit chart" : "Enlarge chart";
});
updateRadar();
}
