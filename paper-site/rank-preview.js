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
const standings = document.querySelector(".preview-standings");
const rankKeys = ["in1k", "coco", "ade20k", "nyuv2", "ssv2"];
const resultsByModel = new Map(historicalScores.map(result => [result.name, result]));
for (const result of historicalScores.filter(result => result.name === "V-JEPA")) {
  const checkpoint = result.family === "VSSL" ? "ViT-H" : "ViT-L";
  resultsByModel.set(`V-JEPA (${checkpoint})`, result);
}

function showStandings() {
  const task = rankKeys.indexOf(taskPicker.value);
  const rank = model => task < 0 ? model.place : model.ranks[task];
  const leaders = historicalRanks.filter(model => rank(model) <= 10)
    .sort((a, b) => rank(a) - rank(b));
  standings.caption.textContent = taskPicker.selectedOptions[0].text + " · Top 10 including ties";
  standings.querySelector("[aria-sort]")?.removeAttribute("aria-sort");
  standings.querySelector(`[data-rank="${taskPicker.value}"]`).setAttribute("aria-sort", "ascending");
  const body = standings.tBodies[0];
  body.replaceChildren();

  for (const model of leaders) {
    const result = resultsByModel.get(model.name);
    const row = body.insertRow();
    const label = document.createElement("th");
    label.scope = "row";
    label.append(sourceLink(model.name), document.createElement("small"));
    label.lastElementChild.textContent = result.family;
    row.append(label);
    row.insertCell().textContent = "#" + model.place;
    row.insertCell().textContent = (model.ranks.reduce((sum, value) => sum + value, 0) / model.ranks.length).toFixed(1);
    for (const [index, place] of model.ranks.entries()) {
      const cell = row.insertCell();
      cell.classList.toggle("ranked-task", task === index);
      const score = document.createElement("strong");
      score.textContent = result.scores[index] ?? "—";
      const taskRank = document.createElement("small");
      taskRank.textContent = "Rank " + place;
      cell.append(score, taskRank);
    }
  }
}
taskPicker.addEventListener("change", showStandings);
showStandings();
