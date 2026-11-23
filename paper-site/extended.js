const probe = document.querySelector("#probe");
const family = document.querySelector("#family");
const body = document.querySelector("#matrix tbody");
const dialog = document.querySelector("dialog");
const modelColors = ["#00b3b0", "#a044e6", "#ff5a5f", "#00a2ff", "#ffc700"];

for (const group of new Set(transferRows.map(row => row.group))) {
  family.add(new Option(group));
}
transferModels.forEach((model, i) => {
  const th = document.createElement("th");
  th.scope = "col";
  th.style.borderTop = "5px solid " + modelColors[i];
  th.append(sourceLink(model));
  document.querySelector("#matrix thead tr").append(th);
});

function showDetails(row) {
  dialog.querySelector("h2").textContent = row.task;
  const protocol = probe.value === "AP" ? "Attentive probing" : "Linear or frozen-backbone probing with a task-specific head";
  const note = [34, 35].includes(row.number) ? " The source records a primary median pose error; its translation/rotation reduction is unverified." : "";
  dialog.querySelector("#conditions").textContent = protocol + " · " + row.metric + " · " + row.direction + " is better. One seed per setting." + note;
  const list = dialog.querySelector("#scores");
  list.replaceChildren();
  transferModels.forEach((model, i) => {
    const item = document.createElement("li");
    item.append(sourceLink(model), " · " + row[probe.value][i]);
    list.append(item);
  });
  dialog.showModal();
}

function render() {
  body.replaceChildren();
  const visible = transferRows.filter(row => family.value === "All" || row.group === family.value);
  document.querySelector("#count").textContent = visible.length + " settings · 5 models";
  for (const row of visible) {
    const tr = body.insertRow();
    const th = document.createElement("th");
    th.scope = "row";
    const button = document.createElement("button");
    button.textContent = row.task;
    button.addEventListener("click", () => showDetails(row));
    const label = document.createElement("small");
    label.textContent = row.metric + (row.direction === "lower" ? " ↓" : " ↑");
    th.append(button, label);
    tr.append(th);
    const values = row[probe.value];
    const best = (row.direction === "lower" ? Math.min : Math.max)(...values.map(Number));
    values.forEach((score, i) => {
      const td = tr.insertCell();
      td.textContent = score;
      if (Number(score) === best) {
        td.className = "winner";
        td.style.setProperty("--accent", modelColors[i]);
        const rank = document.createElement("small");
        rank.textContent = "1st";
        td.append(rank);
      }
    });
  }
}
probe.addEventListener("change", render);
family.addEventListener("change", render);
render();
