// Figure 1 model labels match the Figure 2 v45 legend except for these two
// encoders. In the legend both are called V-JEPA and have distinct IDs.
const v45IdByModel = new Map(
  historicalScores.filter(row => row.name !== "V-JEPA").map(row => [row.name, row.id]),
);
v45IdByModel.set("V-JEPA (ViT-H)", 29);
v45IdByModel.set("V-JEPA (ViT-L)", 54);

const v45ById = new Map(historicalScores.map(row => [row.id, row]));
function manuscriptScoreForRank(model) {
  return v45ById.get(v45IdByModel.get(model.name));
}
