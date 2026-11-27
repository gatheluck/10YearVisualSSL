(() => {
  const panels = Array.from(document.querySelectorAll(".comparison-panel"));
  function revealFragment() {
    const fragment = location.hash.slice(1);
    const name = fragment === "" ? "" : fragment.split("?")[0];
    const target = document.getElementById(
      name === "leaderboard" && fragment.includes("?") ? "unified" : name,
    );
    const selected = target?.closest(".comparison-panel");
    if (!selected) {
      if (name === "" || name === "leaderboard" || name === "experiments") {
        for (const panel of panels) panel.open = false;
      }
      return;
    }
    for (const panel of panels) panel.open = panel === selected;
    for (let parent = target.parentElement; parent && parent !== selected; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
    }
    requestAnimationFrame(() => target.scrollIntoView());
  }

  window.addEventListener("hashchange", revealFragment);
  document.addEventListener("click", event => {
    const href = event.target.closest("a")?.getAttribute("href");
    if (href && href === location.hash) revealFragment();
  });
  revealFragment();
})();
