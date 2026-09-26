# V-JEPA 2.1 downstream encoder integration

> **Legacy terminology:** numbered Step 1-4 labels in this file are historical
> code/experiment identifiers, not manuscript section numbers. See the
> [ASIS, CTRL and manuscript comparison map](../../docs/PAPER_TERMINOLOGY.md) before matching a recipe to a paper result.

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](../../docs/SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

This module uses the pinned official author submodule at commit
`204698b45b3712590f06245fbfba32d3be539812`. It does not copy author code or
redownload weights. Initialize submodules before use. The method CPU/CUDA locks
reuse the existing pinned author environment; the downstream CPU lock includes
its `einops` dependency. Container behavior has not been tested.

Set `backbone.kind` to `vjepa2_1`, `arch` to `vit_large` or
`vit_giant_xformers`, `img_size` to 384, `patch_size` to 16, and `encoder` to the
local released checkpoint. The first variant requires `ema_encoder`; the second
requires `target_encoder`. Prefixes are removed and loading is strict. An empty
checkpoint means random smoke weights, never a reproduced result. Optional
`embed_dim`, `depth` and `num_heads` must all be specified for a reduced fixture.
They do not identify a released checkpoint.

For ADE20K, NYUv2, COCO and SSv2 use `profile: capture_basic5_components` with
`adaptation: frozen`, `attentive` or `finetune` in the existing task configuration.
The pretrained encoder geometry remains 384; the **task** defines the input
resolution. Image inputs are padded on the right/bottom to multiples of 16 and
use the real T=1 image branch. There is no implicit resize to 384. Spatial maps
contain all final image patch tokens without special-token removal. Normalization
is performed by the task, exactly once.

SSv2 uses native video tokens, never independent frame encoding. The frozen/FT
linear head receives an L2-normalized mean of the final tokens. AP receives the
complete ordered spatiotemporal tokens. Video requires a positive even frame
count of at least two; the paper protocol uses 16. The linear head starts at
zero. Frozen providers stay in eval mode; the explicit trainable provider
preserves autograd, including the caller's no-grad evaluation context.

## Verification and remaining reproduction work

The captured Basic5 wrapper at snapshot
`806ea3061ea499c75fd15fcaf15dfdeec5c19303` is the behavioral source. The category
VideoSSL wrapper additionally resizes inputs to 384; that **different** profile
is not silently substituted here. ViT-L's key is established by the category
wrapper; the captured unified Basic5 experiment uses ViT-g.

Reduced official encoder tests cover outputs, image/video routing, strict keys,
frozen/FT gradients and updates, and all four actual task CLIs in all three
adaptation modes. A private comparison executes the unchanged captured wrapper
and compares image/video outputs, input/parameter gradients and SGD updates.
Tests using random reduced weights do not reproduce full-checkpoint scores.

This is an encoder/downstream integration, **not a completed method adapter**.
The Step-3/CompEval port remains incomplete relative to the inspected
experimental pipelines; this describes package integration, not absence of
the original implementations. ImageNet FT,
video augmentation, distributed training, seed aggregation and full-data score
comparisons remain separate work. The 2026-09-22 additions below supersede the
earlier missing-components status for ImageNet LP/AP, FT groups and schedules.
Existing component runs remain ineligible for the paper table. No workbook
score is fabricated or used as a substitute for executing its experiment.

The provider also declares verified FT layer groups for the four downstream
component tasks and ImageNet classification readout for online LP/AP. See
[the execution guide](../../docs/DOWNSTREAM.md#extended-training-components-2026-09-22).
FT grouping follows this encoder family's captured mapping; it is not a
universal policy for other providers. ImageNet FT model composition is tested,
but the CLI refuses its unresolved augmentation recipe. These additions do not
constitute a complete method adapter or canonical paper-score reproduction.
