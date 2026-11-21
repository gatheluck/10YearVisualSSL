"use strict";
function buildRankBoard(rows, familyByName, families) {
  const board = document.getElementById("leaderboard");
  const get = (id) => document.getElementById(id);
  const familyStyles = new Map(
    families.map(([key, label, token]) => [
      key,
      { label, color: "var(--" + token + ")" },
    ]),
  );
  const benchmarks = ["ImageNet", "COCO", "ADE20K", "NYUv2", "SSv2"];
  const mean = (r) =>
    (r.ranks.reduce((sum, x) => sum + x, 0) / r.ranks.length).toFixed(1);
  const shade = (n) =>
    "hsl(177 " +
    (57 - ((n - 1) / 108) * 26) +
    "% " +
    (24 + ((n - 1) / 108) * 71) +
    "%)";
  const select = get("inspect-config"),
    matrix = get("rank-matrix"),
    search = get("rank-search"),
    expand = get("rank-expand");
  const elements = new Map();
  function add(parent, tag, text = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    parent.append(node);
    return node;
  }
  function inspect(row) {
    select.value = row.name;
    for (const [name, col] of elements)
      col.classList.toggle("picked", name === row.name);
    const panel = get("rank-profile");
    panel.replaceChildren();
    const { label, color } = familyStyles.get(familyByName.get(row.name));
    panel.style.setProperty("--series", color);
    add(panel, "p", label + " · " + row.year).className = "family-name";
    add(panel, "h3", row.name);
    const value = add(panel, "p");
    value.className = "spotlight";
    add(value, "strong", "#" + row.place);
    add(value, "span", "overall · " + mean(row) + " mean rank");
    const metrics = add(panel, "dl");
    row.ranks.forEach((r, i) => {
      const group = add(metrics, "div");
      add(group, "dt", benchmarks[i]);
      add(group, "dd", r);
    });
  }
  for (const row of rows) {
    select.add(new Option(row.name, row.name));
    const col = add(matrix, "div");
    col.className = "matrix-column";
    col.title = row.name + " · overall #" + row.place;
    for (const rank of row.ranks) {
      const cell = add(col, "i");
      cell.className = "matrix-cell";
      cell.style.backgroundColor = shade(rank);
    }
    add(col, "span", row.name).className = "model-tick";
    col.onclick = () => inspect(row);
    elements.set(row.name, col);
  }
  select.onchange = () => {
    inspect(rows.find((row) => row.name === select.value));
    const viewport = matrix.parentElement;
    const frame = viewport.getBoundingClientRect();
    const cell = elements.get(select.value).getBoundingClientRect();
    if (cell.left < frame.left || cell.right > frame.right)
      viewport.scrollLeft +=
        (cell.left + cell.right - frame.left - frame.right) / 2;
  };
  inspect(rows[0]);
  let all = false;
  function draw() {
    const body = board.querySelector("tbody");
    body.replaceChildren();
    const q = search.value.trim().toLowerCase(),
      matches = rows.filter((row) => row.name.toLowerCase().includes(q));
    const shown = all || q ? matches : matches.slice(0, 10);
    for (const row of shown) {
      const tr = body.insertRow(),
        { label, color } = familyStyles.get(familyByName.get(row.name));
      tr.insertCell().textContent = "#" + row.place;
      const name = add(tr, "th");
      name.scope = "row";
      name.style.setProperty("--series", color);
      add(name, "b", row.name);
      add(name, "small", label);
      tr.insertCell().textContent = mean(row);
      for (const rank of row.ranks) {
        const td = tr.insertCell();
        const num = add(td, "span", rank);
        num.className = "rank-chip";
        num.style.background = shade(rank);
        num.style.color = rank < 16 ? "white" : "#000";
      }
    }
    get("rank-count").textContent = shown.length
      ? shown.length + " of " + rows.length + " configurations"
      : "No matching configurations";
    expand.hidden = Boolean(q);
    expand.textContent = all ? "Show top 10" : "Show all 109 configurations";
    expand.setAttribute("aria-expanded", String(all));
  }
  expand.onclick = () => {
    all = !all;
    draw();
  };
  search.oninput = draw;
  draw();
}
