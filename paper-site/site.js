"use strict";

const tasks = {
  "ImageNet-1K": {
    metric: "Top-1 (%)",
    direction: "↑",
    description: "Image classification on ImageNet-1K.",
  },
  COCO: {
    metric: "Box AP",
    direction: "↑",
    description:
      "COCO 2017 object detection, evaluated with bounding-box average precision.",
  },
  ADE20K: {
    metric: "mIoU",
    direction: "↑",
    description:
      "Semantic segmentation on ADE20K, evaluated with mean intersection over union.",
  },
  NYUv2: {
    metric: "RMSE (m)",
    direction: "↓",
    description:
      "Depth estimation on NYUv2. Lower root mean squared error in meters is better.",
  },
  SSv2: {
    metric: "Top-1 (%)",
    direction: "↑",
    description: "Video classification on Something-Something V2.",
  },
};
const protocols = {
  LP: [
    "Linear probe / task head",
    "The backbone is frozen. ImageNet uses a classifier on global features; SSv2 averages per-frame features. COCO uses a Faster R-CNN detection head. Segmentation and depth use a spatial predictor.",
  ],
  AP: [
    "Attentive probe",
    "The backbone is frozen. An attention reader or spatial adapter precedes the task head. On COCO the manuscript labels this route “Attn + head”.",
  ],
  DPT: [
    "Dense prediction transformer",
    "A dense prediction head reads four intermediate layers of the frozen backbone. Figure 3 reports this protocol for ADE20K semantic segmentation and NYUv2 depth estimation.",
  ],
  LoRA: [
    "Low-rank adaptation",
    "Rank-16 adapters in the attention projections and the task-specific head are trained. The original backbone weights stay fixed.",
  ],
};
const models = ["DINOv3-7B", "VGGT-Ω", "RAEv2", "V-JEPA 2.1-G", "SigLIP 2-G"];
const palette = [
  "var(--teal)",
  "var(--coral)",
  "var(--purple)",
  "var(--green)",
  "var(--sky)",
];
const byId = (id) => document.getElementById(id);
const entry = (model, task, protocol) =>
  results.find(
    (r) => r.model === model && r.task === task && r.protocol === protocol,
  );
const betterFirst = (a, b, task) =>
  (Number(a.mean) - Number(b.mean)) * (tasks[task].direction === "↓" ? 1 : -1);
const paperTasks = (protocol) =>
  Object.keys(tasks).filter((task) => entry(models[0], task, protocol));
const state = { protocol: "LP", task: "all", sort: "", reverse: false };
const dialog = document.querySelector('dialog[aria-labelledby="detail-title"]');

function chart(task) {
  const ranked = (protocol) =>
    models
      .map((model) => entry(model, task, protocol))
      .sort((a, b) => betterFirst(a, b, task));
  const lp = ranked("LP"),
    ap = ranked("AP");
  const rightRank = (model) => ap.findIndex((r) => r.model === model);
  let reversals = 0;
  lp.forEach((r, i) =>
    lp.slice(i + 1).forEach((s) => {
      if (rightRank(r.model) > rightRank(s.model)) reversals++;
    }),
  );
  document
    .querySelectorAll("[data-chart]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.chart === task)),
    );
  // The vertical scale is ordinal; numeric endpoints retain the published magnitudes.
  const lines = lp.map((r, i) => {
    const j = rightRank(r.model),
      left = 42 + i * 34,
      right = 42 + j * 34,
      color = palette[models.indexOf(r.model)];
    return (
      '<g><text x="0" y="' +
      (left + 4) +
      '" font-weight="600">' +
      r.model +
      "</text>" +
      '<text x="150" y="' +
      (left + 4) +
      '" text-anchor="end">' +
      r.mean +
      "</text>" +
      '<line x1="164" x2="303" y1="' +
      left +
      '" y2="' +
      right +
      '" stroke="' +
      color +
      '" stroke-width="2.4"/>' +
      '<circle cx="164" cy="' +
      left +
      '" r="4" fill="' +
      color +
      '"/><circle cx="303" cy="' +
      right +
      '" r="4" fill="' +
      color +
      '"/>' +
      '<text x="315" y="' +
      (right + 4) +
      '">' +
      ap[j].mean +
      '</text><text x="475" y="' +
      (right + 4) +
      '" text-anchor="end" font-weight="600">' +
      r.model +
      "</text></g>"
    );
  });
  byId("plot").innerHTML =
    '<svg viewBox="0 0 476 200" role="img" aria-labelledby="chart-title chart-desc" font-family="Arial,sans-serif" font-size="13" fill="#172329"><title id="chart-title">' +
    task +
    ' rank by readout</title><desc id="chart-desc">' +
    lp
      .map((r) => r.model + " " + r.mean + " to " + ap[rightRank(r.model)].mean)
      .join("; ") +
    '</desc><g fill="#53616c" font-size="11"><text x="0" y="16">LP · ' +
    tasks[task].metric +
    '</text><text x="475" y="16" text-anchor="end">AP · ' +
    tasks[task].metric +
    "</text></g>" +
    lines.join("") +
    "</svg>";
  byId("reversal-count").textContent = reversals + " / 10";
  byId("finding").textContent =
    reversals +
    " of 10 model pairs reverse order on " +
    task +
    " from LP to AP, based on reported means.";
}

function notes(r) {
  return r.short
    ? "† Shortened LoRA schedule for DINOv3-7B on " + r.task + "."
    : "";
}
function details(r) {
  byId("detail-title").textContent = r.model + " · " + r.task;
  byId("detail").innerHTML =
    '<p class="score">' +
    r.mean +
    " ± " +
    r.sd +
    "</p><p>" +
    tasks[r.task].metric +
    " " +
    tasks[r.task].direction +
    " · Mean ± SD over three runs</p><p>" +
    tasks[r.task].description +
    "</p><p><strong>" +
    r.protocol +
    " · " +
    protocols[r.protocol][0] +
    "</strong><br>" +
    protocols[r.protocol][1] +
    "</p><p>" +
    notes(r) +
    '</p><p class="source">Manuscript · Figure 3, page 4 · Unified evaluation</p>';
  byId("detail").append(sourceLink(r.model, true));
  dialog.showModal();
}
function setHash() {
  const p = new URLSearchParams({ protocol: state.protocol, task: state.task });
  if (state.sort) {
    p.set("sort", state.sort);
    if (state.reverse) p.set("reverse", "1");
  }
  history.replaceState(null, "", "#unified?" + p);
}
function visibleTasks() {
  return state.task === "all" ? paperTasks(state.protocol) : [state.task];
}
function orderedModels() {
  return state.sort
    ? [...models].sort(
        (a, b) =>
          betterFirst(
            entry(a, state.sort, state.protocol),
            entry(b, state.sort, state.protocol),
            state.sort,
          ) * (state.reverse ? -1 : 1),
      )
    : models;
}
function selectProtocol(protocol) {
  state.protocol = protocol;
  if (!paperTasks(protocol).includes(state.task)) state.task = "all";
  if (!visibleTasks().includes(state.sort)) {
    state.sort = "";
    state.reverse = false;
  }
}
function draw() {
  const columns = visibleTasks();
  document
    .querySelectorAll("#tracks button")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.textContent === state.protocol)),
    );
  const options = ["all", ...paperTasks(state.protocol)];
  byId("task").replaceChildren(
    ...options.map(
      (t) =>
        new Option(t === "all" ? "All tasks" : t, t, false, t === state.task),
    ),
  );
  const head = document.querySelector("#unified thead");
  head.innerHTML =
    '<tr><th scope="col">Checkpoint</th>' +
    columns
      .map((task) => {
        const sorted = state.sort === task;
        const ascending = (tasks[task].direction === "↓") !== state.reverse;
        return (
          '<th scope="col"' +
          (sorted
            ? ' aria-sort="' + (ascending ? "ascending" : "descending") + '"'
            : "") +
          '><button aria-label="Sort by ' +
          task +
          '" data-sort="' +
          task +
          '">' +
          task +
          "<small>" +
          tasks[task].metric +
          " " +
          tasks[task].direction +
          "</small></button></th>"
        );
      })
      .join("") +
    "</tr>";
  const body = document.querySelector("#unified tbody");
  body.innerHTML = orderedModels()
    .map(
      (model) =>
        '<tr><th scope="row"><span class="key" style="background:' +
        palette[models.indexOf(model)] +
        '"></span>' +
        sourceLink(model).outerHTML +
        "</th>" +
        columns
          .map((task) => {
            const r = entry(model, task, state.protocol);
            const best = models.every(
              (m) => betterFirst(r, entry(m, task, state.protocol), task) <= 0,
            );
            return (
              '<td><button data-model="' +
              model +
              '" data-task="' +
              task +
              '" class="' +
              (best ? "best" : "") +
              '" aria-label="' +
              model +
              " " +
              task +
              " " +
              r.mean +
              " plus or minus " +
              r.sd +
              ', details">' +
              r.mean +
              (r.short ? "†" : "") +
              " <small>± " +
              r.sd +
              "</small></button></td>"
            );
          })
          .join("") +
        "</tr>",
    )
    .join("");
  byId("rows-info").textContent =
    "5 checkpoints · " + columns.length * 5 + " scores";
  byId("track-note").textContent = {
    LP: "COCO uses a detection head on a frozen backbone (head-only).",
    AP: "COCO uses an attention adapter and a detection head.",
    DPT: "Four frozen feature layers are read on the two dense prediction tasks.",
    LoRA: "† The DINOv3-7B ImageNet-1K and SSv2 runs use shortened schedules.",
  }[state.protocol];
}
function exportCSV() {
  const rows = [
    [
      "model",
      "task",
      "protocol",
      "metric",
      "better",
      "mean",
      "sd",
      "runs",
      "note",
      "source",
    ],
  ];
  for (const model of orderedModels())
    for (const task of visibleTasks()) {
      const r = entry(model, task, state.protocol);
      rows.push([
        model,
        task,
        state.protocol,
        tasks[task].metric,
        tasks[task].direction === "↑" ? "higher" : "lower",
        r.mean,
        r.sd,
        3,
        notes(r),
        "Manuscript Figure 3, page 4",
      ]);
    }
  const csv =
    rows
      .map((row) =>
        row.map((v) => '"' + String(v).replaceAll('"', '""') + '"').join(","),
      )
      .join("\r\n") + "\r\n";
  const url = URL.createObjectURL(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "basic-five-" + state.protocol + "-" + state.task + ".csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

for (const [code, [name, description]] of Object.entries(protocols)) {
  const btn = document.createElement("button");
  btn.textContent = code;
  btn.onclick = () => {
    selectProtocol(code);
    draw();
    setHash();
  };
  byId("tracks").append(btn);
  const guide = document.createElement("details");
  guide.innerHTML =
    "<summary>" + code + " · " + name + "</summary><p>" + description + "</p>";
  byId("guides").append(guide);
}
document
  .querySelectorAll("[data-chart]")
  .forEach((btn) => (btn.onclick = () => chart(btn.dataset.chart)));
document.querySelector("#unified thead").onclick = (e) => {
  const btn = e.target.closest("[data-sort]");
  if (!btn) return;
  const task = btn.dataset.sort;
  state.reverse = state.sort === task && !state.reverse;
  state.sort = task;
  draw();
  setHash();
  document.querySelector('[data-sort="' + task + '"]').focus();
};
document.querySelector("#unified tbody").onclick = (e) => {
  const btn = e.target.closest("[data-model]");
  if (btn) details(entry(btn.dataset.model, btn.dataset.task, state.protocol));
};
byId("task").onchange = (e) => {
  state.task = e.target.value;
  state.sort = state.task === "all" ? "" : state.task;
  state.reverse = false;
  draw();
  setHash();
};
byId("csv").onclick = exportCSV;
byId("close").onclick = () => dialog.close();
const params = new URLSearchParams(location.hash.split("?")[1] || "");
if (Object.hasOwn(protocols, params.get("protocol")))
  state.protocol = params.get("protocol");
const available = paperTasks(state.protocol);
state.task = available.includes(params.get("task"))
  ? params.get("task")
  : params.get("task") !== "all" && matchMedia("(max-width:550px)").matches
    ? available[0]
    : "all";
if (visibleTasks().includes(params.get("sort"))) {
  state.sort = params.get("sort");
  state.reverse = params.get("reverse") === "1";
}
chart("SSv2");
draw();
if (/^#(?:leaderboard|unified)\?/.test(location.hash))
  byId("unified").scrollIntoView();
