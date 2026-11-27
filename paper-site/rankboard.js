"use strict";
function buildRankBoard(rows, familyByName, families) {
  const board = document.getElementById("leaderboard");
  const get = id => document.getElementById(id);
  const familyStyles = new Map(families.map(([key, label, token]) =>
    [key, { label, color: `var(--${token})` }],
  ));
  const benchmarks = [
    { key: "in1k", name: "ImageNet-1K", task: "Image classification", metric: "Top-1 accuracy (%)", suffix: "%" },
    { key: "coco", name: "COCO 2017", task: "Object detection", metric: "Bounding-box AP", suffix: " bbox AP" },
    { key: "ade20k", name: "ADE20K", task: "Semantic segmentation", metric: "Mean intersection over union (%)", suffix: "%" },
    { key: "nyuv2", name: "NYUv2", task: "Depth estimation", metric: "RMSE (m)", suffix: " m", lower: true },
    { key: "ssv2", name: "SSv2", task: "Video classification", metric: "Top-1 accuracy (%)", suffix: "%" },
  ];
  const mean = model =>
    (model.ranks.reduce((sum, rank) => sum + rank, 0) / model.ranks.length).toFixed(1);
  const select = get("inspect-config");
  const matrix = get("rank-matrix");
  const search = get("rank-search");
  const task = get("rank-task");
  const expand = get("rank-expand");
  const profile = get("selected-model");
  const dialog = get("rank-score-dialog");
  const modelByName = new Map(rows.map(row => [row.name, row]));
  const elements = new Map();
  let lastModelButton;
  const saved = window.history.state?.modelRankings;
  let all = saved?.all ?? false;
  search.value = saved?.query ?? "";
  if (Array.from(task.options).some(option => option.value === saved?.task)) task.value = saved.task;

  function remember() {
    window.history.replaceState({
      ...window.history.state,
      modelRankings: { all, query: search.value, task: task.value, model: profile.hidden ? null : select.value },
    }, "");
  }

  function add(parent, tag, text = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    parent.append(node);
    return node;
  }

  function scoreText(record, index) {
    const value = record.scores[index];
    if (value !== null) return value + benchmarks[index].suffix;
    return record.name === "AlayaWorld v1.1" && index === 3 ? "Excluded" : "Unavailable";
  }

  function showScore(model, index) {
    const benchmark = benchmarks[index];
    const record = manuscriptScoreForRank(model);
    const fields = get("rank-score-fields");
    get("rank-score-title").textContent = model.name + " · " + benchmark.name;
    fields.replaceChildren();
    const field = (label, value) => {
      add(fields, "dt", label);
      return add(fields, "dd", value);
    };
    field("Benchmark", benchmark.name + " · " + benchmark.task);
    field("Metric", benchmark.metric + (benchmark.lower ? " · lower is better" : " · higher is better"));
    field("Figure 1 published rank", "#" + model.ranks[index]);
    field("Figure 2 v45 manuscript score", scoreText(record, index));
    field("Figure 2 v45 legend entry", `#${record.id} · ${record.family}`);
    field("Evaluation conditions", "Original-setting evaluation (ASIS) uses released or reproduced checkpoints. Feature extraction and downstream evaluation vary by model family.");
    const scoreLink = add(field("Figures"), "a", "Figure 2 manuscript scores");
    scoreLink.href = "#scores";
    scoreLink.onclick = () => {
      const historyTask = get("history-task");
      historyTask.value = benchmark.key;
      historyTask.dispatchEvent(new Event("change"));
      dialog.close();
    };
    const rankLink = add(field(""), "a", "Figure 1 rank heatmap");
    rankLink.href = "#rank-matrix";
    rankLink.onclick = () => dialog.close();
    const evaluation = add(field("Source notes"), "a", "Source and evaluation conditions");
    evaluation.href = "#historical-evaluation";
    evaluation.onclick = () => dialog.close();
    field("Original model source").append(sourceLink(model.name, true));
    dialog.showModal();
  }

  function inspect(row, trigger = null, moveFocus = false) {
    select.value = row.name;
    get("matrix-selection").textContent = row.name;
    for (const [name, col] of elements) col.classList.toggle("picked", name === row.name);
    const panel = get("rank-profile");
    const record = manuscriptScoreForRank(row);
    panel.replaceChildren();
    const { label, color } = familyStyles.get(familyByName.get(row.name));
    panel.style.setProperty("--series", color);
    add(panel, "p", label + " · " + row.year).className = "family-name";
    add(panel, "h3").append(sourceLink(row.name));
    const value = add(panel, "p");
    value.className = "spotlight";
    add(value, "strong", "#" + row.place);
    add(value, "span", "overall · " + mean(row) + " mean rank");
    const metrics = add(panel, "dl");
    row.ranks.forEach((rank, index) => {
      const benchmark = benchmarks[index];
      const group = add(metrics, "div");
      add(group, "dt", benchmark.name);
      const detail = add(group, "dd");
      const button = add(detail, "button", scoreText(record, index));
      button.type = "button";
      button.setAttribute("aria-label", benchmark.name + " " + button.textContent + ", rank " + rank + ". View score details");
      button.onclick = () => showScore(row, index);
      add(detail, "small", "Rank " + rank);
      const rail = add(group, "span");
      rail.className = "rank-rail";
      rail.setAttribute("aria-hidden", "true");
      rail.style.setProperty("--position", ((rank - 1) * 100 / (rows.length - 1)) + "%");
    });
    lastModelButton = trigger;
    profile.hidden = false;
    if (moveFocus) profile.focus();
    remember();
  }

  for (const row of rows) {
    select.add(new Option(row.name, row.name));
    const col = add(matrix, "div");
    col.className = "matrix-column";
    col.title = row.name + " · overall #" + row.place;
    for (const rank of row.ranks) {
      const cell = add(col, "i");
      cell.className = "matrix-cell";
      cell.style.backgroundColor = taskRankColor(rank);
    }
    add(col, "span", row.name).className = "model-tick";
    col.onclick = () => inspect(row, select);
    elements.set(row.name, col);
  }
  select.onchange = () => {
    inspect(modelByName.get(select.value), select);
    const viewport = matrix.parentElement;
    const frame = viewport.getBoundingClientRect();
    const cell = elements.get(select.value).getBoundingClientRect();
    if (cell.left < frame.left || cell.right > frame.right) {
      viewport.scrollLeft += (cell.left + cell.right - frame.left - frame.right) / 2;
    }
  };
  const initial = modelByName.get(saved?.model) ?? rows[0];
  get("matrix-selection").textContent = initial.name;
  elements.get(initial.name).classList.add("picked");
  if (saved?.model && modelByName.has(saved.model)) inspect(initial);
  get("close-model").onclick = () => {
    profile.hidden = true;
    (lastModelButton?.isConnected ? lastModelButton : search).focus();
    remember();
  };

  function draw() {
    const body = board.querySelector("tbody");
    body.replaceChildren();
    const index = benchmarks.findIndex(benchmark => benchmark.key === task.value);
    const rank = row => index < 0 ? row.place : row.ranks[index];
    get("rank-order").textContent = index < 0 ? "Overall rank" : "Task rank";
    get("rank-value").textContent = index < 0 ? "Mean rank" : benchmarks[index].name + " score";
    const query = search.value.trim().toLowerCase();
    const matches = rows.filter(row => row.name.toLowerCase().includes(query))
      .sort((a, b) => rank(a) - rank(b));
    const shown = all || query ? matches : matches.filter(row => rank(row) <= 10);
    for (const row of shown) {
      const result = manuscriptScoreForRank(row);
      const tr = body.insertRow();
      tr.dataset.legendId = result.id;
      const { label, color } = familyStyles.get(result.family);
      tr.insertCell().textContent = "#" + rank(row);
      const name = add(tr, "th");
      name.scope = "row";
      name.style.setProperty("--series", color);
      const model = add(add(name, "b"), "button", row.name);
      model.type = "button";
      model.className = "inspect-model";
      model.onclick = () => inspect(row, model, true);
      const link = sourceLink(row.name);
      link.textContent = "↗";
      link.setAttribute("aria-label", "Original source for " + row.name);
      name.append(link);
      add(name, "small", label);
      const value = tr.insertCell();
      if (index < 0) {
        value.textContent = mean(row);
      } else {
        const button = add(value, "button", scoreText(result, index));
        button.type = "button";
        button.onclick = () => showScore(row, index);
      }
    }
    get("rank-count").textContent = shown.length
      ? shown.length + " of " + rows.length + " models"
      : "No matching models";
    expand.hidden = Boolean(query);
    expand.textContent = all ? "Show top 10" : "Show all 109 models";
    expand.setAttribute("aria-expanded", String(all));
  }
  expand.onclick = () => {
    all = !all;
    draw();
    remember();
  };
  for (const [input, event] of [[search, "input"], [task, "change"]]) {
    input.addEventListener(event, () => {
      draw();
      remember();
    });
  }
  draw();
}
