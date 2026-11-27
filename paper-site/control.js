(() => {
  "use strict";
  const root = document.getElementById("controlled");
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
    const left = 48, top = 20, size = 220;
    const right = left + size, bottom = top + size;
    const pos = (v) => (size * (v - low)) / (high - low);
    mark("polygon", {
      points: lower
        ? `${left},${bottom} ${right},${bottom} ${right},${top}`
        : `${left},${bottom} ${left},${top} ${right},${top}`,
      fill: "#d5e9e1",
    });
    mark("path", { d: `M${left} ${top}V${bottom}H${right}`, stroke: "#879a9d", fill: "none" });
    mark("line", {
      x1: left,
      y1: bottom,
      x2: right,
      y2: top,
      stroke: "#708885",
      "stroke-dasharray": "4 4",
    });
    for (const value of [low, high]) {
      mark(
        "text",
        { x: left + pos(value), y: bottom + 16, "text-anchor": "middle" },
        value,
      );
      mark("text", { x: left - 6, y: bottom + 4 - pos(value), "text-anchor": "end" }, value);
    }
    mark("text", { x: left + size / 2, y: bottom + 35, "text-anchor": "middle" }, "ASIS");
    mark(
      "text",
      { transform: "translate(21 130) rotate(-90)", "text-anchor": "middle" },
      "CTRL",
    );
    for (const [method, a, c] of points) {
      const dot = mark("circle", {
        cx: left + pos(Number(a)),
        cy: bottom - pos(Number(c)),
        r: 3.8,
        fill: "var(--blue)",
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
    const body = root.querySelector("tbody");
    body.replaceChildren();
    const counts = { Better: 0, Tied: 0, Worse: 0 };
    for (const [method, a, c, delta] of points) {
      const gain = Number(delta) * (lower ? -1 : 1);
      const outcome = gain > 0 ? "Better" : gain < 0 ? "Worse" : "Tied";
      counts[outcome]++;
      const row = body.insertRow(),
        label = document.createElement("th");
      label.scope = "row";
      label.append(sourceLink(method));
      row.append(label);
      for (const v of [a, c, Number(delta) > 0 ? "+" + delta : delta, outcome])
        row.insertCell().textContent = v;
      row.lastElementChild.className = outcome.toLowerCase();
    }
    document.getElementById("control-chart-title").textContent = name + " transfer";
    document.getElementById("metric").textContent = metric;
    for (const [outcome, count] of Object.entries(counts)) {
      root.querySelector("#control-outcomes ." + outcome.toLowerCase() + " dd").textContent = count;
    }
    document.getElementById("control-direction").textContent =
      metric + " · " + (lower ? "Lower RMSE is better." : "Higher is better.") +
      " The shaded region marks better CTRL scores.";
    document.getElementById("rule").textContent =
      "Difference = CTRL − ASIS, in the metric’s original units. " +
      (lower
        ? "Lower RMSE is better; a negative difference is an improvement."
        : "Higher is better; a positive difference is an improvement.");
    drawPairs(points, lower);
  }
  benchmark.addEventListener("change", render);
  render();
})();
