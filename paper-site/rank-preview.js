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

const atlas = document.querySelector(".coverage-atlas");
function addCoverageMarks(selector, names) {
  const marks = document.createDocumentFragment();
  for (const name of names) {
    const mark = document.createElement("i");
    mark.title = name;
    marks.append(mark);
  }
  atlas.querySelector(selector).append(marks);
}
addCoverageMarks(".model-marks", historicalScores.map(model => model.name));
addCoverageMarks(".setting-marks", Array.from(document.querySelector(".task-choice").options, option => option.text));

const motionButton = document.querySelector(".motion-toggle");
const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
function setAtlasMotion(playing) {
  atlas.dataset.playing = String(playing);
  motionButton.textContent = playing ? "Pause motion" : "Play motion";
}
setAtlasMotion(!preference.matches);
motionButton.hidden = false;
motionButton.addEventListener("click", () => setAtlasMotion(atlas.dataset.playing !== "true"));
preference.addEventListener("change", () => setAtlasMotion(!preference.matches));
