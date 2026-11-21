"use strict";
const benchmarks = [
  ["in1k", "ImageNet-1K", "Image classification", "Top-1 accuracy (%)"],
  ["coco", "COCO", "COCO 2017 object detection", "Bounding-box AP"],
  [
    "ade20k",
    "ADE20K",
    "Semantic segmentation",
    "Mean intersection over union (%)",
  ],
  ["nyuv2", "NYUv2", "Depth estimation", "RMSE (meters)"],
  ["ssv2", "SSv2", "Video classification", "Top-1 accuracy (%)"],
];
const notes = {
  RAE: "Encoder features.",
  RAEv2: "Diffusion generator features from block 29 at timestep 0.1.",
  "VGGT-Ω aggregator":
    "Multi-view aggregator patch tokens from the non-text-aligned checkpoint, without geometric prediction heads.",
  "VGGT-Ω scene":
    "Camera/register tokens from the non-text-aligned checkpoint, without geometric prediction heads.",
  "V-JEPA 2.1": "ViT-G video encoder.",
  "RAVLT-S": "RAVLT-S configuration.",
};
const el = (id) => document.getElementById(id);
const task = el("task"),
  year = el("year");
let scores = false;
function add(parent, tag, value) {
  const child = document.createElement(tag);
  child.textContent = value;
  parent.append(child);
  return child;
}
for (const [key, name] of benchmarks) task.add(new Option(name, key));
const years = [...new Set(historicalScores.map((r) => r.year))].sort(
  (a, b) => a - b,
);
for (const y of years) year.add(new Option(y, y));
function mean(row) {
  return (row.ranks.reduce((a, b) => a + b, 0) / row.ranks.length).toFixed(1);
}
function displayScore(row, i) {
  return (
    row.scores[i] ??
    (row.name === "AlayaWorld v1.1" && i === 3 ? "Excluded" : "Unavailable")
  );
}
function showDetails(row, i) {
  const [, name, description, metric] = benchmarks[i];
  el("source").textContent = scores
    ? "Figure 2 · ASIS score"
    : "Figure 1 · Published rank";
  el("detail-title").textContent = row.name;
  const details = el("details");
  details.replaceChildren();
  const field = (label, value) => {
    add(details, "dt", label);
    add(details, "dd", value);
  };
  field("Benchmark", name + " · " + description);
  field(
    "Metric",
    metric + (i === 3 ? " · lower is better" : " · higher is better"),
  );
  if (scores) {
    field("Figure 2 result", displayScore(row, i));
    field(
      "Configuration",
      row.family + " · first released " + row.year + " · legend #" + row.id,
    );
    if (notes[row.name]) field("Feature source", notes[row.name]);
    if (i === 4 && ["JiT", "MAGE"].includes(row.name))
      field("Video input", "The rerun uses the first eight frames.");
    if (row.scores[i] === null)
      field(
        "Coverage",
        "No numerical point is plotted for this configuration and task in the manuscript.",
      );
  } else {
    field(
      "Published ranks",
      "Overall #" +
        row.place +
        " · mean " +
        mean(row) +
        " · " +
        name +
        " #" +
        row.ranks[i],
    );
  }
  field(
    "Evaluation conditions",
    "ASIS uses released or reproduced checkpoints on shared benchmarks. Feature extraction and downstream training vary by family. The figures do not specify a full per-run training recipe.",
  );
  el("details").closest("dialog").showModal();
}
function draw(rows, i) {
  const plot = el("plot");
  plot.replaceChildren();
  function mark(tag, attrs, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    plot.append(node);
    return node;
  }
  const all = scores ? historicalScores : historicalRanks;
  const value = (row) => (scores ? row.scores[i] : row.ranks[i]);
  const values = all.map(value).filter((v) => v !== null);
  const lo = scores ? 0 : 1,
    hi = scores ? Math.ceil(Math.max(...values)) : 109;
  const lowBetter = !scores || i === 3;
  const x = (y) => 55 + (720 * (y - years[0])) / (years.at(-1) - years[0]);
  const y = (v) =>
    10 + 128 * (lowBetter ? (v - lo) / (hi - lo) : (hi - v) / (hi - lo));
  for (const v of [lo, hi]) {
    mark("line", { x1: 48, x2: 785, y1: y(v), y2: y(v), stroke: "#c3ccd4" });
    mark(
      "text",
      { x: 40, y: y(v) + 5, "text-anchor": "end", fill: "#4d5d62" },
      v,
    );
  }
  for (const yr of years.filter((v, j) => j % 3 === 0 || v === years.at(-1))) {
    mark(
      "text",
      { x: x(yr), y: 165, "text-anchor": "middle", fill: "#4d5d62" },
      yr,
    );
  }
  for (const row of rows) {
    const v = value(row);
    if (v === null) continue;
    const dot = mark("circle", {
      cx: x(row.year),
      cy: y(v),
      r: 3.6,
      fill: "var(--blue)",
      "fill-opacity": ".6",
    });
    const title = document.createElementNS(plot.namespaceURI, "title");
    title.textContent = row.name + " · " + row.year + " · " + v;
    dot.append(title);
  }
}
function render() {
  const i = benchmarks.findIndex(([key]) => key === task.value);
  const name = benchmarks[i][1];
  const all = scores ? historicalScores : historicalRanks;
  const rows = all.filter((r) => !year.value || r.year === Number(year.value));
  if (scores) rows.sort((a, b) => a.year - b.year || a.id - b.id);
  el("ranking").setAttribute("aria-pressed", String(!scores));
  el("scores").setAttribute("aria-pressed", String(scores));
  el("heading").textContent = scores ? "Historical scores" : "Overall ranking";
  el("count").textContent =
    rows.length + " of " + all.length + " configurations";
  el("context").textContent = scores
    ? "Ordered by first release. Select a score for its metric and evaluation conditions. Missing points are identified in the table."
    : "Ordered by the mean of the five published benchmark ranks. Ties and ranks refer to the full Figure 1 cohort, including when a release year is selected.";
  el("chart-title").textContent =
    name + (scores ? " scores" : " ranks") + " by release year";
  el("axis").textContent = scores
    ? benchmarks[i][3] + (i === 3 ? " ↓" : " ↑")
    : "Published task rank · 1 is best";
  draw(rows, i);
  const head = document.querySelector("thead"),
    body = document.querySelector("tbody");
  head.replaceChildren();
  body.replaceChildren();
  const tr = add(head, "tr", "");
  const labels = scores
    ? ["Configuration", "Release", name + " score"]
    : ["Configuration", "Release", "Overall", "Mean rank", name + " rank"];
  for (const label of labels) add(tr, "th", label).scope = "col";
  for (const row of rows) {
    const line = add(body, "tr", "");
    const cell = add(line, "th", "");
    cell.scope = "row";
    add(cell, "span", row.name);
    if (scores) add(cell, "small", "#" + row.id + " · " + row.family);
    add(line, "td", row.year);
    if (!scores) {
      add(line, "td", "#" + row.place);
      add(line, "td", mean(row));
    }
    const btn = add(
      add(line, "td", ""),
      "button",
      scores ? displayScore(row, i) : "#" + row.ranks[i],
    );
    btn.addEventListener("click", () => showDetails(row, i));
  }
}
el("ranking").onclick = () => {
  scores = false;
  render();
};
el("scores").onclick = () => {
  scores = true;
  render();
};
year.onchange = render;
task.onchange = render;
render();
