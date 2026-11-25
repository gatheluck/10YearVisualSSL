const preview = document.querySelector(".cover-matrix");
const cells = document.createDocumentFragment();
historicalRanks.forEach((model, column) => {
  model.ranks.forEach((rank, row) => {
    const cell = document.createElementNS(preview.namespaceURI, "rect");
    cell.setAttribute("x", column * 4);
    cell.setAttribute("y", row * 10);
    cell.setAttribute("width", "3.65");
    cell.setAttribute("height", "9");
    cell.setAttribute("fill", taskRankColor(rank));
    const title = document.createElementNS(preview.namespaceURI, "title");
    title.textContent = model.name + " · " + ["ImageNet","COCO","ADE20K","NYUv2","SSv2"][row] + " #" + rank;
    cell.append(title);
    cells.append(cell);
  });
});
preview.append(cells);

const taskPicker = document.getElementById("preview-task");
const limitPicker = document.getElementById("preview-limit");
const standings = document.querySelector(".preview-standings");
const rankKeys = ["in1k", "coco", "ade20k", "nyuv2", "ssv2"];

function showStandings() {
  const task = rankKeys.indexOf(taskPicker.value);
  const rank = model => task < 0 ? model.place : model.ranks[task];
  const leaders = historicalRanks.filter(model => rank(model) <= Number(limitPicker.value))
    .sort((a, b) => rank(a) - rank(b));
  standings.caption.textContent = taskPicker.selectedOptions[0].text + " · Top " + limitPicker.value + " including ties";
  standings.querySelector(".comparison-stat").textContent = task < 0 ? "Mean rank" : "Overall rank";
  const body = standings.tBodies[0];
  body.replaceChildren();
  for (const model of leaders) {
    const row = body.insertRow();
    row.insertCell().textContent = rank(model);
    const label = document.createElement("th");
    label.scope = "row";
    label.append(sourceLink(model.name));
    row.append(label);
    row.insertCell().textContent = task < 0
      ? (model.ranks.reduce((sum, value) => sum + value, 0) / model.ranks.length).toFixed(1)
      : model.place;
  }
}
taskPicker.addEventListener("change", showStandings);
limitPicker.addEventListener("change", showStandings);
showStandings();
