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
