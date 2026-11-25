(() => {
  const section = document.getElementById("citation");
  const entry = section.querySelector("pre code");
  const button = section.querySelector("button");
  const status = section.querySelector('[role="status"]');
  button.hidden = false;
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(entry.textContent);
      status.textContent = "BibTeX copied.";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(entry);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Entry selected. Use your device’s Copy command.";
    }
  });
})();
