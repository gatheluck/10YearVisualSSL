// Former index.html linked directly to leaderboard sections and filter states.
function followLegacyBoardLink() {
  const section = location.hash.slice(1).split("?")[0];
  if (["leaderboard","unified","experiments","protocols","readout-title","rank-atlas","rank-matrix"].includes(section))
    location.replace("leaderboard.html" + location.search + location.hash);
}
followLegacyBoardLink();
window.addEventListener("hashchange", followLegacyBoardLink);
