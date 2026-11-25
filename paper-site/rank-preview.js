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

const story = document.querySelector(".task-story");
const modelButtons = story.querySelectorAll(".example-models button");
const rankOutputs = story.querySelectorAll("output");
function showTaskExample(button) {
  const model = historicalRanks.find(row => row.name === button.textContent);
  rankOutputs[0].textContent = "#" + model.ranks[2];
  rankOutputs[1].textContent = "#" + model.ranks[4];
  modelButtons.forEach(item => item.setAttribute("aria-pressed", String(item === button)));
}
modelButtons.forEach(button => button.addEventListener("click", () => showTaskExample(button)));
showTaskExample(modelButtons[0]);

const motionButton = document.querySelector(".motion-toggle");
const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
function setRankMotion(playing) {
  story.dataset.playing = String(playing);
  motionButton.textContent = playing ? "Pause motion" : "Play motion";
}
setRankMotion(!preference.matches);
motionButton.hidden = false;
motionButton.addEventListener("click", () => setRankMotion(story.dataset.playing !== "true"));
preference.addEventListener("change", () => setRankMotion(!preference.matches));
