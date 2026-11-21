"use strict";
const tasks = [
  ["ImageNet-1K", "Image classification · Top-1 (%)"],
  ["COCO", "Object detection · Bounding-box AP"],
  ["ADE20K", "Semantic segmentation · mIoU (%)"],
  ["NYUv2", "Depth estimation · RMSE (m)"],
  ["SSv2", "Video classification · Top-1 (%)"],
];
const benchmark = document.getElementById("benchmark");
for (const [name] of tasks) benchmark.add(new Option(name, name));

function drawPairs(points, lower) {
  const svg = document.getElementById("paired");
  svg.replaceChildren();
  function mark(tag, attrs, label) {
    const n = document.createElementNS(svg.namespaceURI, tag);
    for (const [key, value] of Object.entries(attrs))
      n.setAttribute(key, value);
    if (label !== undefined) n.textContent = label;
    svg.append(n);
    return n;
  }
  const values = points.flatMap(([, a, c]) => [Number(a), Number(c)]);
  const low = lower ? Math.floor(Math.min(...values) * 10) / 10 : 0;
  const high = lower
    ? Math.ceil(Math.max(...values) * 10) / 10
    : Math.ceil(Math.max(...values));
  const pos = (v) => (180 * (v - low)) / (high - low);
  mark("polygon", {
    points: lower ? "90,190 270,190 270,10" : "90,190 90,10 270,10",
    fill: "#d5e9e1",
  });
  mark("path", { d: "M90 10V190H270", stroke: "#879a9d", fill: "none" });
  mark("line", {
    x1: 90,
    y1: 190,
    x2: 270,
    y2: 10,
    stroke: "#708885",
    "stroke-dasharray": "4 4",
  });
  for (const value of [low, high]) {
    mark(
      "text",
      { x: 90 + pos(value), y: 208, "text-anchor": "middle" },
      value,
    );
    mark("text", { x: 82, y: 195 - pos(value), "text-anchor": "end" }, value);
  }
  mark("text", { x: 180, y: 232, "text-anchor": "middle" }, "ASIS");
  mark(
    "text",
    { transform: "translate(40 100) rotate(-90)", "text-anchor": "middle" },
    "CTRL",
  );
  for (const [method, a, c] of points) {
    const dot = mark("circle", {
      cx: 90 + pos(Number(a)),
      cy: 190 - pos(Number(c)),
      r: 3.8,
      fill: "#2849b4",
      "fill-opacity": ".75",
    });
    const title = document.createElementNS(svg.namespaceURI, "title");
    title.textContent = method + " · " + a + " → " + c;
    dot.append(title);
  }
}
function render() {
  const i = benchmark.selectedIndex,
    [name, metric] = tasks[i],
    lower = name === "NYUv2";
  const points = controlPairs.map(([method, results]) => [
    method,
    ...results[i],
  ]);
  const body = document.querySelector("tbody");
  body.replaceChildren();
  let improved = 0,
    tied = 0;
  for (const [method, a, c, delta] of points) {
    const gain = Number(delta) * (lower ? -1 : 1);
    if (gain > 0) improved++;
    if (gain === 0) tied++;
    const outcome = gain > 0 ? "Better" : gain < 0 ? "Worse" : "Tied";
    const row = body.insertRow(),
      label = document.createElement("th");
    label.scope = "row";
    label.textContent = method;
    row.append(label);
    for (const v of [a, c, Number(delta) > 0 ? "+" + delta : delta, outcome])
      row.insertCell().textContent = v;
    row.lastElementChild.className = outcome.toLowerCase();
  }
  document.getElementById("chart-title").textContent = name + " transfer";
  document.getElementById("metric").textContent = metric;
  document.getElementById("finding").textContent =
    improved +
    " better, " +
    (points.length - improved - tied) +
    " worse" +
    (tied ? ", " + tied + " tied" : "") +
    ".";
  document.getElementById("rule").textContent =
    "Difference = CTRL − ASIS, in the metric’s original units. " +
    (lower
      ? "Lower RMSE is better; a negative difference is an improvement."
      : "Higher is better; a positive difference is an improvement.");
  drawPairs(points, lower);
}
benchmark.addEventListener("change", render);
render();
