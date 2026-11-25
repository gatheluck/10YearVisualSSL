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

// The published lower bound spans 25 complete groups of 100 runs.
for (let group = 0; group < 25; group++) {
  atlas.querySelector(".experiment-units").append(document.createElement("i"));
}
