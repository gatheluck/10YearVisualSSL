// Each host page supplies a heading that fits its part of the paper.
const broadTransfer = document.getElementById("figure-five");
const figureHeading = broadTransfer.querySelector("h2");
figureHeading.insertAdjacentHTML("beforebegin", '<p class="eyebrow">Figure 5 · Broad transfer</p>');
figureHeading.insertAdjacentHTML("afterend", `
  <div class="transfer-facts">
    <p><strong>29<span>/40</span></strong>DINOv3-7B first-place finishes with linear or task-specific frozen probes.</p>
    <p><strong>20<span>/40</span></strong>DINOv3-7B first-place finishes with attentive probes.</p>
    <p><strong>6<span>/9</span></strong>Detection and segmentation settings led by RAEv2 with attentive probes.</p>
  </div>
  <figure>
    <div class="transfer-pan" role="region" tabindex="0" aria-label="Five model charts and the task key; scroll horizontally on small screens">
      <a href="figure-5.png" aria-label="Open the full-resolution Figure 5">
        <img src="figure-5.png" width="3200" height="846" alt="Five radar plots compare DINOv3-7B, VGGT-Ω, RAEv2, V-JEPA 2.1-G and SigLIP 2-G across forty numbered settings. Each spoke is one setting; the five plots use the same task order.">
      </a>
    </div>
    <figcaption><span>Read across models. On each spoke, the outer ring is the best of the five for that probe. LP and AP are scaled separately.</span>
      <a href="figure-5.png">Enlarge original figure ↗</a></figcaption>
  </figure>
  <div class="transfer-foot"><p>The original figure includes all 40 settings. The score explorer covers 38 while the SpaceNet 2 and SUN RGB-D task definitions are under review.</p>
  <a href="extended.html#matrix">Inspect task scores</a></div>`);
