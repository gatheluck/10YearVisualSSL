"use strict";
function buildRankBoard(rows, familyByName, families) {
  const board = document.getElementById("leaderboard");
  const get = id => document.getElementById(id);
  const familyStyles = new Map(families.map(([key, label, token]) =>
    [key, { label, color: `var(--${token})` }],
  ));
  const benchmarks = [
    { key: "in1k", name: "ImageNet-1K", shortName: "ImageNet", task: "Image classification", metric: "Top-1 accuracy (%)", profileMetric: "Top-1 accuracy", suffix: "%" },
    { key: "coco", name: "COCO 2017", shortName: "COCO", task: "Object detection", metric: "Bounding-box AP", profileMetric: "Bounding-box AP", suffix: " bbox AP" },
    { key: "ade20k", name: "ADE20K", task: "Semantic segmentation", metric: "Mean intersection over union (%)", profileMetric: "mIoU", suffix: "%" },
    { key: "nyuv2", name: "NYUv2", task: "Depth estimation", metric: "RMSE (m)", profileMetric: "RMSE", suffix: " m", lower: true },
    { key: "ssv2", name: "SSv2", task: "Video classification", metric: "Top-1 accuracy (%)", profileMetric: "Top-1 accuracy", suffix: "%" },
  ];
  const mean = model =>
    (model.ranks.reduce((sum, rank) => sum + rank, 0) / model.ranks.length).toFixed(1);
  const matrix = get("rank-matrix");
  const search = get("rank-search");
  const task = get("rank-task");
  const expand = get("rank-expand");
  const viewport = matrix.parentElement;
  const readout = get("matrix-readout");
  const matrixProfile = get("matrix-model-profile");
  const resultRow = get("rank-profile-template").content.firstElementChild.cloneNode(true);
  const tableProfile = resultRow.querySelector("#selected-model");
  const dialog = get("rank-score-dialog");
  const modelByName = new Map(rows.map(row => [row.name, row]));
  const elements = new Map();
  let tableTrigger;
  let activeCell;
  let pickedCell;
  let pickedColumn;
  const saved = window.history.state?.modelRankings;
  let all = saved?.all ?? false;
  let tableModel = saved?.tableModel ?? null;
  let matrixRow = modelByName.get(saved?.matrixModel ?? saved?.model) ?? rows[0];
  let matrixTask = saved?.matrixTask ?? null;
  search.value = saved?.query ?? "";
  if (Array.from(task.options).some(option => option.value === saved?.task)) task.value = saved.task;

  function remember() {
    window.history.replaceState({
      ...window.history.state,
      modelRankings: {
        all, query: search.value, task: task.value, tableModel,
        matrixModel: matrixRow.name, matrixTask,
      },
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

  function renderProfile(container, row) {
    const panel = container.querySelector(".model-profile-body");
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
      const label = add(group, "dt", benchmark.name);
      add(label, "span", benchmark.profileMetric).className = "profile-metric";
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
  }

  function moveMatrixFocus(cell, focus = false) {
    if (activeCell) activeCell.tabIndex = -1;
    activeCell = cell;
    cell.tabIndex = 0;
    if (focus) cell.focus({ preventScroll: true });
    const frame = viewport.getBoundingClientRect();
    const box = cell.getBoundingClientRect();
    if (box.left < frame.left || box.right > frame.right) {
      viewport.scrollLeft += (box.left + box.right - frame.left - frame.right) / 2;
    }
  }

  function showMatrixSelection(row, index, reveal = false) {
    matrixRow = row;
    matrixTask = index;
    get("matrix-selection").textContent = row.name;
    get("matrix-task-result").textContent = index === null
      ? "" : benchmarks[index].name + " · Rank " + row.ranks[index];
    pickedColumn?.classList.remove("picked");
    pickedColumn = elements.get(row.name);
    pickedColumn.classList.add("picked");
    pickedCell?.setAttribute("aria-pressed", "false");
    pickedCell = index === null ? null : pickedColumn.children[index];
    pickedCell?.setAttribute("aria-pressed", "true");
    moveMatrixFocus(pickedCell ?? pickedColumn.firstElementChild);
    renderProfile(matrixProfile, row);
    if (reveal) readout.scrollIntoView({ block: "nearest" });
  }

  for (const [column, row] of rows.entries()) {
    const col = add(matrix, "div");
    col.className = "matrix-column";
    col.title = row.name + " · overall #" + row.place;
    for (const [index, rank] of row.ranks.entries()) {
      const cell = add(col, "button");
      cell.type = "button";
      cell.className = "matrix-cell";
      cell.tabIndex = -1;
      cell.setAttribute("aria-label", row.name + " · " + benchmarks[index].name + " · Rank " + rank);
      cell.setAttribute("aria-pressed", "false");
      cell.style.backgroundColor = taskRankColor(rank);
      cell.onclick = () => {
        showMatrixSelection(row, index, true);
        remember();
      };
      cell.onkeydown = event => {
        let x = column, y = index;
        switch (event.key) {
          case "ArrowLeft": x--; break;
          case "ArrowRight": x++; break;
          case "ArrowUp": y--; break;
          case "ArrowDown": y++; break;
          case "Home": x = 0; break;
          case "End": x = rows.length - 1; break;
          default: return;
        }
        event.preventDefault();
        if (x >= 0 && x < rows.length && y >= 0 && y < benchmarks.length) {
          moveMatrixFocus(matrix.children[x].children[y], true);
        }
      };
    }
    const tick = add(col, "span", row.name);
    tick.className = "model-tick";
    tick.onclick = () => {
      showMatrixSelection(row, null, true);
      remember();
    };
    elements.set(row.name, col);
  }
  showMatrixSelection(matrixRow, matrixTask);
  const scrollCue = document.querySelector(".matrix-scroll-cue");
  new ResizeObserver(() => {
    scrollCue.hidden = viewport.scrollWidth <= viewport.clientWidth + 1;
  }).observe(viewport);

  function closeTableProfile(restoreFocus = false) {
    const previous = tableTrigger;
    previous?.setAttribute("aria-expanded", "false");
    resultRow.remove();
    tableTrigger = null;
    tableModel = null;
    if (restoreFocus && previous?.isConnected) previous.focus();
  }

  function openTableProfile(row, trigger, reveal = false) {
    tableTrigger?.setAttribute("aria-expanded", "false");
    tableTrigger = trigger;
    tableModel = row.name;
    trigger.setAttribute("aria-expanded", "true");
    const parent = trigger.closest("tr");
    resultRow.cells[0].colSpan = parent.cells.length;
    parent.after(resultRow);
    renderProfile(tableProfile, row);
    if (reveal) tableProfile.querySelector("h3").scrollIntoView({ block: "nearest" });
  }

  resultRow.querySelector("#close-model").onclick = () => {
    closeTableProfile(true);
    remember();
  };
  tableProfile.onkeydown = event => {
    if (event.key === "Escape") {
      closeTableProfile(true);
      remember();
    }
  };

  function draw() {
    const body = board.querySelector("tbody");
    body.replaceChildren();
    let openRow;
    const index = benchmarks.findIndex(benchmark => benchmark.key === task.value);
    const rank = row => index < 0 ? row.place : row.ranks[index];
    const overall = index < 0;
    const benchmark = benchmarks[index];
    board.classList.toggle("overall-ranks", overall);
    get("rank-components").hidden = !overall;
    get("rank-order").textContent = overall ? "Overall rank" : "Task rank";
    get("rank-value").textContent = overall ? "Mean rank" : benchmark.name + " score";
    get("ranking-method").textContent = overall
      ? "Overall standings use the mean of the five published task ranks. Select a rank for its score and evaluation conditions."
      : benchmark.task + " · " + benchmark.metric + " · " + (benchmark.lower ? "Lower is better" : "Higher is better");
    const query = search.value.trim().toLowerCase();
    const matches = rows.filter(row => row.name.toLowerCase().includes(query))
      .sort((a, b) => rank(a) - rank(b));
    const shown = all || query ? matches : matches.filter(row => rank(row) <= 10);
    for (const row of shown) {
      const result = manuscriptScoreForRank(row);
      const tr = body.insertRow();
      tr.dataset.legendId = result.id;
      tr.setAttribute("role", "row");
      const { label, color } = familyStyles.get(result.family);
      const position = tr.insertCell();
      position.setAttribute("role", "cell");
      position.textContent = "#" + rank(row);
      const name = add(tr, "th");
      name.scope = "row";
      name.setAttribute("role", "rowheader");
      name.style.setProperty("--series", color);
      const model = add(add(name, "b"), "button", row.name);
      model.type = "button";
      model.className = "inspect-model";
      add(model, "span").className = "disclosure-mark";
      model.lastElementChild.setAttribute("aria-hidden", "true");
      model.setAttribute("aria-controls", "selected-model");
      model.setAttribute("aria-expanded", "false");
      model.onclick = () => {
        if (tableModel === row.name) closeTableProfile(true);
        else openTableProfile(row, model, true);
        remember();
      };
      if (tableModel === row.name) openRow = [row, model];
      const link = sourceLink(row.name);
      link.textContent = "↗";
      link.setAttribute("aria-label", "Original source for " + row.name);
      name.append(link);
      add(name, "small", label);
      const value = tr.insertCell();
      value.setAttribute("role", "cell");
      if (overall) {
        value.textContent = mean(row);
        const ranks = tr.insertCell();
        ranks.className = "task-ranks";
        ranks.setAttribute("role", "cell");
        const list = add(ranks, "div");
        list.className = "task-ranks-list";
        row.ranks.forEach((rank, taskIndex) => {
          const taskInfo = benchmarks[taskIndex];
          const button = add(list, "button");
          button.type = "button";
          button.setAttribute("aria-label", taskInfo.name + ", rank " + rank + ". View score details");
          add(button, "span", taskInfo.shortName ?? taskInfo.name);
          add(button, "strong", rank);
          button.onclick = () => showScore(row, taskIndex);
        });
      } else {
        const button = add(value, "button", scoreText(result, index));
        button.type = "button";
        button.onclick = () => showScore(row, index);
      }
    }
    if (openRow) openTableProfile(...openRow);
    else closeTableProfile();
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
