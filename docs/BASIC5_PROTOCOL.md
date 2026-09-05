# BASIC5 evaluation protocol

This document is the on-`main` source of truth for the **BASIC5** evaluation
protocol defined by the implementation team. The definitions originally arrived
as chat messages; a chat message is not in the working tree next session and
cannot be diffed, so the protocol is captured here and its machine-checkable
claims are enforced by `tests/test_basic5_protocol.py` (this repository's rule:
a policy in a document does not hold unless it is also machinery).

BASIC5 evaluates a frozen or tuned backbone on the same five datasets --
ImageNet-1k (classification), COCO (detection/instance), ADE20K (semantic
segmentation), NYUv2 (depth), and SSv2 (video action) -- across three tracks:

| track | id | backbone | head |
|---|---|---|---|
| Linear probe | `BASIC5_FAIR_v1` | frozen | minimal task head only |
| Attentive probe | `BASIC5_ATTENTIVE_v1` | frozen | one shared attention reader + baseline head |
| Finetune | `BASIC5_FINETUNE_v1` | fully trainable | baseline head |

A dataset that a backbone cannot serve with a real spatial-token map (e.g. a
generative or global-only backbone on a dense task) is reported as
`UNSUPPORTED`, not faked.

This repository's present focus is **`BASIC5_FAIR_v1` on ImageNet-1k** -- the
linear probe -- because that is what the paper figure and the feature-extraction
sweep need first. The other tracks and the four dense/video datasets are
captured below for context but are not yet the subject of conformance work.

---

## Two artifacts, one protocol

The word "linear probe" maps onto **two distinct things** in this repository,
and they must not be conflated:

1. **The saved feature dump** -- `bin/extract-features.py` +
   `methods/*/feature_provider.py`. One command runs every method's frozen
   encoder over the ImageNet val split and saves one feature vector per image
   (`<out>/<method>/features.npy`). It computes **no accuracy**; it is the input
   a paper figure and a downstream probe read. The *feature* rules of the
   protocol (below: `b`, `c`, `d`, `e`) govern this artifact.

2. **The trained linear classifier** -- each method's `linear_eval` stage
   (`configs/linear_eval*.yaml` + `evaluate_linear*.py`). It trains a 1000-class
   `nn.Linear` head on the frozen feature and reports Top-1/Top-5. The *probe*
   rules (`opt`, `seed`, `aug`, `metric`) govern this artifact.

---

## `BASIC5_FAIR_v1` -- ImageNet-1k linear-probe rules

The rules below are the checkable core of the FAIR/ImageNet recipe. Each has a
stable id used by the conformance table and by commit messages. Status is one
of: **conformant** (the port matches), **deviation** (the port differs and
should be reconciled), **deliberate** (the port differs on purpose, with a
recorded reason), **partial** (some methods conform, some do not), **pending**
(work in progress).

<!-- BASIC5-FAIR-IMAGENET-RULES -->

| id | scope | requirement | status |
|----|-------|-------------|--------|
| b | feature | Eval preprocessing is Resize (shorter side) 256 + CenterCrop 224 -- one deterministic 224 center crop, no eval-time augmentation | deviation |
| c | feature | The saved final **global** feature is **L2-normalised** to unit length | conformant |
| d | feature | Exactly **one canonical final feature layer** -- do not search layers or concatenate features from multiple layers | deviation |
| e | feature | Use the **published backbone normalisation** (the mean/std the backbone was trained under) | conformant |
| opt | probe | Optimiser SGD, momentum 0.9, weight decay **0**, base LR **0.1** at effective batch **256** with linear LR scaling, **cosine** decay, **100** epochs | partial |
| seed | probe | Run **seeds 0, 1, 2** and report **mean ± std** | deviation |
| aug | probe | Train-time augmentation is **RandomResizedCrop + HorizontalFlip only** | partial |
| metric | probe | Report **Top-1 and Top-5** | conformant |

<!-- default-representation: l2 -->

The saved representation is chosen by `bin/extract-features.py --representation
{l2,raw}`. The **default is `l2`** (rule `c`); `raw` saves the encoder's
pre-normalisation vector for inspection. The value used is recorded in each
`meta.json` and in the run `manifest.json`.

### Additional FAIR/ImageNet rules (not yet machine-checked)

- The backbone is **frozen** and in `eval()` mode: no gradients, and no running
  updates to norm affine parameters, BatchNorm statistics, or positional
  embeddings. (Verified conformant across all 51 providers by audit: every
  provider calls `.eval()` and sets `requires_grad=False`.)
- The head is a single linear layer to the dataset's class count (1000 for
  ImageNet-1k).

---

## Conformance status of the feature dump (ImageNet, per method)

Measured by audit (see the audit summary in the session history). `frozen/eval`
and `representation` are uniform, so they are stated once: **every** provider
freezes the backbone in `eval()` and returns the raw feature (the driver then
applies rule `c`). Columns below record the two rules that vary per method --
single-layer (`d`) and eval size/crop (`b`) -- plus the normalisation (`e`).

Deviations that need reconciliation:

- **Rule `d` (single canonical layer):**
  - `27_ibot` concatenates the `[CLS]` token from the **last four blocks**
    (`n_last_blocks=4, avgpool_patchtokens=0`), giving a 4×-wide vector. This is
    a genuine multi-layer concatenation and the clearest `d` violation. Note
    `23_dino` already made the opposite, **deliberate** choice: its config
    documents that this port uses the single-feature probe instead of DINO's
    last-4 concatenation -- so the port already has a single-feature policy that
    `27_ibot` is inconsistent with.
  - `12_cmc` and `08_split_brain` concatenate the two **colour-channel branch**
    encoders (L and ab). This is the model's single architectural
    representation, not a multi-depth cherry-pick; whether it counts against
    rule `d` is a judgement call recorded here and to be decided before any
    change.

- **Rule `b` (Resize 256 + CenterCrop 224):**
  - Final size not 224: `02_vae` (28), `04_context_encoder` (227),
    `05_jigsaw_puzzle` (255), `09_jigsaw_puzzle_pp` (75), `11_cpc` (256),
    `26_simmim` (192), `36_franca` (518), `cosmos3_super` (448), `sam3` (336),
    `vjepa2`/`vjepa2_ac` (256), `image_gpt` (32). Several of these are the
    backbone's **native input resolution**; forcing 224 could be wrong, so each
    is a per-method judgement, not a blanket edit.
  - Size 224 but wrong crop pipeline (square resize / no center crop): `cae`,
    `data2vec2`, `videomae`, `06_rotation_prediction`, `10_inst_disc`.

- **Rule `e` (published normalisation):** mostly conformant -- each provider
  reproduces its own backbone's published normalisation, which is correct even
  when it is not ImageNet (CLIP mean/std for `38_clip`/`aimv2`/`eva02`; 0.5/0.5
  for `siglip`/`data2vec2`/`32_nepa`/`var`; CIE-Lab for the colour methods).
  One numeric quirk: `17_swav` uses `std[0]=0.228` vs the standard `0.229`.
  Candidates using bare `[0,1]`/no normalisation: `02_vae`, `05`, `06`, `09`,
  `14_simclrv1`, `16_simclrv2`, `image_gpt`.

## Conformance status of the trained probe (`linear_eval`, per method)

Uniformly conformant: frozen backbone + `eval()`, single linear head, SGD +
momentum 0.9 + weight decay 0 + cosine decay, and Top-1 **and** Top-5 reporting
(rule `metric`).

Deviations that need reconciliation:

- **Rule `seed` (seeds 0,1,2 + mean±std):** **every** method runs a single seed
  (42, or 0 for `23_dino`). No seed loop, no mean±std aggregation. This is the
  single most systematic probe deviation.
- **Rule `aug` (RRC + HFlip only):** the feature-cache family (`14_simclrv1`,
  `13_mocov1`, `23_dino`, `25_mae`, `28_dinov2`, `06_rotation`, `02_vae`,
  `sam3`, `data2vec2`) caches features **once** using the deterministic
  Resize+CenterCrop val transform for training too -- so it applies **no**
  RandomResizedCrop and no HorizontalFlip. The end-to-end family (`20_simsiam`,
  `21_barlow_twins`, `27_ibot`) applies RRC+HFlip and conforms.
- **Rule `opt` (LR/epochs/batch/scaling):**
  - LR ≠ 0.1: `21_barlow_twins` (0.3), `27_ibot` (1e-3).
  - Linear LR scaling with batch: only `20_simsiam` scales by `batch/256`; the
    rest use the LR directly.
  - Batch ≠ 256: `28_dinov2`/`sam3`/`data2vec2` = 32 (large-backbone memory).
  - Epochs ≠ 100: `25_mae`/`06_rotation`/`20_simsiam` = 90.
  - The feature-cache family additionally applies an **extra mean-centering**
    step on top of L2 before the head, which the protocol does not specify.

---

## Reconciliation order (fact-based)

Best-practice order for bringing the port into conformance, most-defensible and
lowest-risk first. Each step is TDD (failing test first) and updates the status
column above when it lands.

1. **`c` — L2-normalised saved feature.** DONE (branch
   `feat/extract-features-l2-representation`): the driver applies L2 by default,
   with a `raw` toggle; 13/13 mutants killed.
2. **Externalise the protocol.** DONE (this document + its enforcing test).
3. **`d` — single canonical layer for `27_ibot`.** Bring iBOT's saved feature
   (and its probe) to a single canonical final feature, consistent with the
   `23_dino` single-feature policy already in the port. Decide `12_cmc` /
   `08_split_brain` (two-tower architectural concat) explicitly -- likely
   conformant-by-design, to be recorded not changed.
4. **`seed` — seeds 0,1,2 + mean±std.** The most systematic probe deviation;
   add a seed loop and aggregated reporting to `linear_eval`.
5. **`aug` — RRC + HFlip for the cache family.** Larger change (defeats the
   feature cache); scope carefully.
6. **`b` — eval preprocessing.** Per-method judgement; native-resolution
   backbones may legitimately keep their size. Reconcile the wrong-crop cases
   (square resize / no center crop) first.
7. **`opt` residuals.** Localised LR/epoch/batch/mean-centering differences.

Items 3–7 are not yet started; this section is the plan of record.

---

## The other two tracks (captured for context)

### `BASIC5_ATTENTIVE_v1`
Frozen backbone; a single shared **attention reader/adapter** (one recipe across
methods) plus a baseline head; seeds 0/1/2; scored at the final scheduled epoch;
`UNSUPPORTED` for dense tasks that lack real spatial tokens. Mandatory reporting
fields: checkpoint checksum, parameter counts (backbone vs added), the feature
layer/shape read, and the per-seed and aggregated metrics.

### `BASIC5_FINETUNE_v1`
Full backbone trainable with a baseline head; per-dataset recipes; seeds 0/1/2;
scored at the final scheduled epoch; `UNSUPPORTED` where a dense task cannot be
served. Same mandatory reporting fields as above (checksum, param counts, feature
layer/shape, per-seed + aggregated metrics).

Both tracks share BASIC5's cross-cutting requirements: the same five datasets,
seeds 0/1/2 with mean±std, "final scheduled epoch" scoring, and explicit
`UNSUPPORTED` rather than a faked dense result.
