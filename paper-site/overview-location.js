function updateOverviewLocation() {
  const primary = document.querySelector('nav[aria-label="Primary"]');
  const overview = !location.hash || location.hash === "#overview";
  for (const link of primary.querySelectorAll('a[href^="#"]')) {
    if ((link.hash === "#overview") === overview) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
updateOverviewLocation();
window.addEventListener("hashchange", updateOverviewLocation);
