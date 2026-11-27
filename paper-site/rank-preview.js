const taskPicker = document.getElementById("preview-task");
const standings = document.querySelector(".preview-standings");
const rankKeys = ["in1k", "coco", "ade20k", "nyuv2", "ssv2"];
const familyColors = new Map(rankingFamilies.map(([family, , token]) => [family, "var(--" + token + ")"]));
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
    const family = document.createElement("small");
    family.className = "preview-family";
    family.style.setProperty("--series", familyColors.get(result.family));
    family.textContent = result.family;
    label.append(sourceLink(model.name), family);
    row.append(label);
    const overallRank = row.insertCell();
    overallRank.textContent = "#" + model.place;
    overallRank.classList.toggle("selected-rank", task < 0);
    row.insertCell().textContent = (model.ranks.reduce((sum, value) => sum + value, 0) / model.ranks.length).toFixed(1);
    for (const [index, place] of model.ranks.entries()) {
      const cell = row.insertCell();
      cell.classList.toggle("selected-rank", task === index);
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
