"use strict";
(() => {
  const byName = new Map(historicalRanks.map(row => [row.name, manuscriptScoreForRank(row).family]));
  const svg = document.getElementById("rank-atlas");
  const legend = document.querySelector(".family-legend");
  function mark(tag, attrs, parent = svg) {
    const el = document.createElementNS(svg.namespaceURI, tag);
    for (const [key, value] of Object.entries(attrs))
      el.setAttribute(key, value);
    parent.append(el);
    return el;
  }
  for (const y of [1, 28, 55, 82, 109])
    mark("line", {
      x1: 0,
      x2: 400,
      y1: y,
      y2: y,
      class: "gridline horizontal",
    });
  for (const x of [0, 100, 200, 300, 400])
    mark("line", { x1: x, x2: x, y1: 1, y2: 109, class: "gridline" });
  const traces = [];
  const buttons = [];
  let selected = null;
  function highlight(family) {
    selected = selected === family ? null : family;
    for (const button of buttons)
      button.setAttribute("aria-pressed", String(button.value === selected));
    const count = traces.filter(
      (line) => line.dataset.family === selected,
    ).length;
    for (const line of traces)
      line.style.opacity = selected
        ? line.dataset.family === selected
          ? "1"
          : ".055"
        : ".72";
    document.getElementById("atlas-status").textContent = selected
      ? count + " highlighted"
      : "All 109 shown";
  }
  for (const [family, label, token] of rankingFamilies) {
    const button = document.createElement("button");
    button.type = "button";
    button.value = family;
    button.textContent = label;
    button.style.setProperty("--series", "var(--" + token + ")");
    button.setAttribute("aria-pressed", "false");
    button.onclick = () => highlight(family);
    legend.append(button);
    buttons.push(button);
  }
  for (const row of historicalRanks) {
    const family = byName.get(row.name);
    const token = rankingFamilies.find(([key]) => key === family)[2];
    const points = row.ranks.map((r, i) => [100 * i, r].join(",")).join(" ");
    const trace = mark("polyline", {
      points,
      stroke: "var(--" + token + ")",
      "data-family": family,
      "vector-effect": "non-scaling-stroke",
    });
    mark("title", {}, trace).textContent =
      row.name + " · " + row.ranks.join(" → ");
    traces.push(trace);
  }
  highlight(null);
  buildRankBoard(historicalRanks, byName, rankingFamilies);
})();
