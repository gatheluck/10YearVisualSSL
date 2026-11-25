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

// Draw the same five task ranks as the matrix in a parallel-coordinates view.
const rankField = document.querySelector(".rank-field svg");
const lines = rankField.querySelector(".rank-lines");
const highlights = rankField.querySelector(".rank-highlights");
historicalRanks.forEach(({ ranks }, index) => {
  const trace = document.createElementNS(rankField.namespaceURI, "polyline");
  trace.setAttribute("points", ranks.map((rank, index) => [index * 100, rank].join(",")).join(" "));
  lines.append(trace);
  if (index % 8 === 0) {
    trace.id = "rank-profile-" + index;
    const highlight = document.createElementNS(rankField.namespaceURI, "use");
    highlight.setAttribute("href", "#" + trace.id);
    highlights.append(highlight);
  }
});

const motionButton = document.querySelector(".motion-toggle");
const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
function setRankMotion(playing) {
  rankField.dataset.playing = String(playing);
  motionButton.textContent = playing ? "Pause motion" : "Play motion";
}
setRankMotion(!preference.matches);
motionButton.hidden = false;
motionButton.addEventListener("click", () => setRankMotion(rankField.dataset.playing !== "true"));
preference.addEventListener("change", () => setRankMotion(!preference.matches));
