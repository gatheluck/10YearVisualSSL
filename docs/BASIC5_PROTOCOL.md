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
sweep need first. The other tracks and the four dense/video datasets now have
the opt-in components described below; complete recipe conformance remains
pending.

---

## Two artifacts, one protocol

### Scope of a conformance claim

The canonical supplied v1 protocols use 224-pixel image/video inputs (COCO
uses 800--1333). Native resolutions are a separate optional track. Historical
"conformant-at-native" observations below describe crop form only; they do
not qualify those runs for the canonical table. Native multi-layer readouts
must likewise remain distinguishable from single-layer FAIR features.

Passing a method contract or saving an L2 feature dump does not certify the
complete five-task LP/AP/FT protocol. The existing dense/video runners retain
their historical recipes; the NYUv2 DPT/L1 runner, for example, is not the
canonical 1x1/log-depth recipe. AP/FT conformance remains pending.

### Safe aggregation of existing probe runs

New `adapterlib` manifests record `aggregation_identity`: a canonical hash of
the resolved configuration with only its top-level `seed` removed, the actual
world size, and the recorded upstream identity. Nested sampler seeds, model
configuration, paths, schedules and other settings remain in the hash. This
is captured before the adapter body executes. Per-seed config hashes remain
unchanged and available separately.

`bin/aggregate-seeds.py` refuses different identities or a mixture of new and
legacy manifests. It also refuses empty, nonnumeric, boolean or nonfinite
metrics. Matching results carry `identity_status: matched`; legacy-only
inputs remain readable but carry `identity_status: unverified_legacy`.
Use `--require-identity` to reject such legacy inputs:

```sh
python3 bin/aggregate-seeds.py --run runs/seed0/out --run runs/seed1/out --run runs/seed2/out --require-identity --out runs/aggregate
```

This verifies recorded configuration compatibility, **not full scientific
provenance or protocol compliance**. A file replaced at the same checkpoint
or dataset path is not detected by a config hash. Weight content hashes,
dataset/split identities, software compatibility and the task's canonical
behavior still need independent verification. No old result is relabeled as
canonical, and the existing sample-standard-deviation convention is retained.
The adapter-to-CLI integration, mismatch rejection and mutation tests exercise
the real result-writing path rather than only a hand-authored manifest.

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
| b | feature | Eval preprocessing is Resize (shorter side) 256 + CenterCrop 224 -- one deterministic 224 center crop, no eval-time augmentation | partial |
| c | feature | The saved final **global** feature is **L2-normalised** to unit length | conformant |
| d | feature | Exactly **one canonical final feature layer** -- do not search layers or concatenate features from multiple layers | conformant |
| e | feature | Use the **published backbone normalisation** (the mean/std the backbone was trained under) | conformant |
| opt | probe | Optimiser SGD, momentum 0.9, weight decay **0**, base LR **0.1** at effective batch **256** with linear LR scaling, **cosine** decay, **100** epochs | partial |
| seed | probe | Run **seeds 0, 1, 2** and report **mean ± std** | partial |
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

### The dump covers every linear-probe method (machine-enforced)

The dump's promise is "one sweep over **every** method". That completeness is
enforced by `tests/test_provider_contract.py`: it **discovers** the methods that
must appear (a method that ships a linear-probe evaluator, `evaluate_linear*.py`)
and asserts each ships a `feature_provider.py` whose `extract_val_features` the
driver can call with exactly the keywords it passes -- the required keyword set
is read from the driver's own call site, not restated. The check is pure AST (no
torch, no GPU, no cross-method import), so a newly ported method is covered the
moment it lands, and a method that carried an evaluator but no provider -- which
would silently drop out of the dump -- is a red test. It names no method
(discover, never list): the `_reference` template and the pretrain-only method
ship no evaluator, so neither is ever expected to ship a provider. Non-vacuity is
proven by `mutations/provider-contract.json` (rename a real provider's extractor,
or drop a required parameter -> the guard fails; 2/2 killed).

---

## Conformance status of the feature dump (ImageNet, per method)

Measured by audit (see the audit summary in the session history). `frozen/eval`
and `representation` are uniform, so they are stated once: **every** provider
freezes the backbone in `eval()` and returns the raw feature (the driver then
applies rule `c`). Columns below record the two rules that vary per method --
single-layer (`d`) and eval size/crop (`b`) -- plus the normalisation (`e`).

Deviations that need reconciliation:

- **Rule `d` (single canonical layer):** RECONCILED (step 3, branch
  `feat/extract-features-l2-representation`).
  - `27_ibot` previously concatenated the `[CLS]` token from the **last four
    blocks** (`n_last_blocks=4`), giving a 4×-wide 1536-d vector -- a genuine
    multi-layer concatenation and the clearest `d` violation. It now ships
    `n_last_blocks=1`, so both the saved feature dump and the trained probe read
    the **single final-block `[CLS]` token** (384-d for `vit_small`). This is the
    same **deliberate** single-feature policy `23_dino` already documents (this
    port uses the single-feature probe instead of the author's last-4
    concatenation). The deviation from iBOT's published recipe is documented in
    `methods/27_ibot/configs/linear_eval.yaml`; the assertion that the port does
    not concatenate is pinned by `tests/test_method_27_ibot.py` and proven
    non-vacuous by `mutations/27_ibot-single-layer.json` (flipping the config
    back to `4` makes the feature 1536-d and kills the width assertion).
  - `12_cmc` and `08_split_brain` concatenate the two **colour-channel branch**
    encoders (L and ab). DECISION: **conformant by design.** This is the model's
    single architectural representation -- the two branches jointly *are* the
    encoder's one final feature, not a multi-depth cherry-pick across layers of a
    single tower. Rule `d` forbids searching/concatenating **layers**, which this
    is not. Recorded, not changed.

- **Rule `b` (Resize shorter side + CenterCrop):** the *form* is the checkable
  core -- an aspect-preserving shorter-side `Resize(int)` then a single
  `CenterCrop`, no eval-time augmentation. The 256/224 pair is the size at 224;
  a native-resolution backbone keeps its own size (forcing 224 could be wrong)
  but must still use the same aspect-preserving crop form. The single
  implementation of the rule is `probe_transforms.basic5_eval_transform` (Resize
  by a single int -- the shorter side, aspect-ratio-preserving, default
  `round(image_size * 256 / 224)` = 256 at 224 -- then `CenterCrop(image_size)`,
  then the method's own ToTensor/normalise tail). It is pinned once in
  `tests/test_probe_transforms.py` (`TestBasic5EvalTransform`) and per wired
  method by `TestTheProbeEvalPreprocessing` (the wiring check lives once in
  `tests/_probe_eval.py`, imported), proven non-vacuous by
  `mutations/basic5-eval.json` (3/3: dropped centre crop, square resize, dropped
  normalise) plus one `<method>-probe-eval.json` per method (1/1 each: the val
  transform reverts to the historical square resize).
  - Size 224 but wrong crop pipeline (square resize / no center crop): `cae`,
    `data2vec2`, `videomae`, `06_rotation_prediction`, `10_inst_disc`.
    RECONCILED (step 6, branch `downstream/basic5-b-eval-crop`). Each of the five
    now reads its val split (which its `feature_provider` extracts) through
    `basic5_eval_transform`.
  - Native resolution but wrong crop pipeline (square `Resize((s,s))`, aspect
    distorted): `05_jigsaw_puzzle` (255), `09_jigsaw_puzzle_pp` (75), `vjepa2`
    (256), `vjepa2_ac` (256). RECONCILED (step 6, same branch). Each keeps its
    **native/config eval size** (not forced to 224) and now routes through
    `basic5_eval_transform` at that size -- so only the crop *form* changed (from
    a square resize with no centre crop to shorter-side Resize + CenterCrop),
    preserving each backbone's own interpolation and normalise tail (jigsaw:
    default bilinear, no normalise; vjepa2: bicubic + ImageNet mean/std;
    vjepa2_ac: bilinear + ImageNet mean/std). Pinned by `TestTheProbeEvalPreprocessing`
    in each method's test and proven by `mutations/<method>-probe-eval.json`
    (1/1 killed each).
  - Native resolution, **already conformant** (measured, shorter-side
    `Resize(int)` + CenterCrop -- no change): `04_context_encoder` (227; Resize 256
    + CenterCrop 227), `26_simmim` (192; `round(192*256/224)` + CenterCrop 192),
    `36_franca` (518; `round(518/0.875)` + CenterCrop 518), `cosmos3_super`
    (448), `sam3` (336; val branch), `image_gpt` (32), and `02_vae`'s ImageFolder
    (ImageNet) path (Resize + CenterCrop at 28). These were verified by reading
    the code, not assumed from size.
  - **Deliberate** (recorded, not changed):
    - `11_cpc` (256): the shipped native path resizes to a fixed **square** 300
      (`Resize((300, 300))`) then `CenterCrop(256)`. This is architecture-specific
      -- CPC splits the crop into a fixed grid of overlapping patches, so the
      source is deliberately a fixed square before the grid split, not an
      aspect-preserving shorter-side resize. Recorded as deliberate, like the
      two-tower `d` cases. (Its separate `arch=vit` path already uses shorter-side
      Resize + CenterCrop, but that is not the shipped native path.)
    - `02_vae` (28): the native MNIST path passes 28×28 grayscale through with no
      resize or crop (grayscale→3-channel repeat only). Dataset-specific: a 28×28
      MNIST digit is neither resized nor cropped; only the ImageFolder/ImageNet
      path (above) applies rule `b`.
  - The row stays `partial` (not `conformant`) for the same reason `aug` does:
    every **known** case above is now reconciled, conformant-at-native, or
    deliberate, but "conformant" should rest on a fleet-wide *discover-all* `b`
    audit that enumerates every probe method, not on this hand-verified list.

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

- **Rule `seed` (seeds 0,1,2 + mean±std):** the shared aggregation mechanism is
  now in place (step 4). Each shipped `linear_eval*.yaml` still carries a single
  `seed` (42, or 0 for `23_dino`) so one launch is one reproducible run; the
  three-seed **mean ± std** is produced at eval time by running that same config
  under each seed and aggregating, in exactly one place:

  ```
  for s in 0 1 2; do
    python3 bin/launch.py --config methods/<method>/configs/linear_eval.yaml \
      --method <method> --override seed=$s --runs-dir runs/<method>
  done
  # each launch writes one run dir under runs/<method>/; point --run at the three
  python3 bin/aggregate-seeds.py \
    --run runs/<method>/<seed0-run> --run runs/<method>/<seed1-run> \
    --run runs/<method>/<seed2-run> --out runs/<method>/agg
  ```

  `bin/resolve-config.py --override seed=N` sets the seed per run (no config
  templating needed), and `bin/aggregate-seeds.py` reads the per-seed
  `run_manifest.json` (seed, status, method, stage) and `metrics.json` (the
  contract metrics) each launch already writes, then emits `aggregate.json` with,
  per comparable metric, its mean and its **sample** standard deviation (ddof=1)
  and the per-seed numbers kept verbatim. It refuses a mean over the wrong seed
  set, a failed run, mixed methods/stages, or a metric missing from a seed
  (`tests/test_aggregate_seeds.py`; `mutations/aggregate-seeds.json`, 7/7 killed).
  Status is **partial**, not conformant: the mechanism and recipe exist and are
  the single implementation of the rule, but the shipped default is still one
  seed per config -- a full three-seed sweep is GPU work run at evaluation time,
  not baked into every config.
- **Rule `aug` (RRC + HFlip only):** the feature-cache family (`14_simclrv1`,
  `13_mocov1`, `23_dino`, `25_mae`, `28_dinov2`, `06_rotation`, `02_vae`,
  `sam3`, `data2vec2`) historically cached features **once** using the
  deterministic Resize+CenterCrop val transform for training too -- so it applied
  **no** RandomResizedCrop and no HorizontalFlip. The end-to-end family
  (`20_simsiam`, `21_barlow_twins`, `27_ibot`) applies RRC+HFlip and conforms.
  Step 5 fan-out is **DONE for the whole feature-cache family**: the shared
  implementation of the rule is `probe_transforms.basic5_train_transform`
  (RandomResizedCrop + HFlip only, then the method's own ToTensor/Normalize
  tail), and every listed method now reads its train split through it while the
  val split (which the provider extracts) stays deterministic. Following the
  port's decision (option B, matching `21_barlow_twins`), the augmented pass is
  still extracted **once and cached** -- one RandomResizedCrop + HorizontalFlip
  draw per image -- so the feature cache is preserved and the backbone is not
  re-run per epoch. Each method carries a `TestTheProbeTrainAugmentation` class
  (the wiring check lives once in `tests/_probe_aug.py`, imported) and a
  `<method>-probe-aug.json` mutation spec (2/2 killed each: train falls back to
  the val transform; `run()` builds train with `train=False`). `02_vae` is a
  **recorded exception in shape, not spirit**: it is dataset-agnostic, so rule
  `aug` applies to its ImageFolder (ImageNet) train path only -- the MNIST path
  stays deterministic on both splits, because randomly cropping or flipping a
  digit can change its class. The row stays `partial` pending a fleet-wide
  discover-all `aug` audit (the analogue of the provider audits behind rules
  b–e): the two families named here conform, but "conformant" should rest on a
  mechanism that enumerates *every* probe method, not on this list.
- **Rule `opt` (LR/epochs/batch/scaling):** the shared mechanism now exists
  (`probe_optim`; see reconciliation step 7) and `cae` is wired as the pilot; the
  residual deviations below are the per-method wiring fan-out still to land.
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
3. **`d` — single canonical layer for `27_ibot`.** DONE (branch
   `feat/extract-features-l2-representation`): iBOT now ships `n_last_blocks=1`,
   so its saved feature and its probe read the single final-block `[CLS]` token
   (384-d), consistent with the `23_dino` single-feature policy; 1/1 mutant
   killed. `12_cmc` / `08_split_brain` (two-tower architectural concat) decided
   **conformant by design** and recorded above, not changed.
4. **`seed` — seeds 0,1,2 + mean±std.** MECHANISM DONE (branch
   `feat/extract-features-l2-representation`): `bin/aggregate-seeds.py` is the
   single shared implementation of the rule -- run any `linear_eval` config under
   `--override seed=0/1/2` and it aggregates the per-seed run outputs into
   mean ± std (sample std, ddof=1), refusing a mean over the wrong seed set, a
   failed run, mixed methods/stages, or a metric missing from a seed
   (`tests/test_aggregate_seeds.py`; `mutations/aggregate-seeds.json`, 7/7
   killed). No per-method `evaluate_linear_*.py` was edited -- the rule lives in
   one place, not fifty. The actual three-seed sweep is GPU work run at
   evaluation time; the shipped configs stay single-seed for one reproducible
   run per launch.
5. **`aug` — RRC + HFlip for the cache family.** DONE across the whole
   feature-cache family (branch `downstream/basic5-aug-feature-cache`).
   `probe_transforms.basic5_train_transform` is the single implementation of the
   rule (RandomResizedCrop + HorizontalFlip only), and every feature-cache method
   reads its train split through it while val stays deterministic. Per the port's
   decision (**option B**, matching `21_barlow_twins`), the augmented pass is
   cached once -- the feature cache is kept, the backbone is not re-run per epoch.
   Proven non-vacuous by `mutations/basic5-aug.json` (3/3: vertical-flip swap,
   centre-crop-instead-of-RRC, dropped-normalise) for the shared rule, and by a
   per-method spec for the wiring: `mutations/14_simclrv1-linear-eval.json` (8/8)
   for the pilot, plus one `<method>-probe-aug.json` for each of `13_mocov1`,
   `23_dino`, `25_mae`, `28_dinov2`, `06_rotation_prediction`, `data2vec2`, `sam3`,
   and `02_vae` (2/2 killed each: train falls back to the val transform; `run()`
   builds the train split with `train=False`; for `02_vae`, train/val on the
   ImageFolder path). The wiring check lives once in `tests/_probe_aug.py`,
   imported by each method's `TestTheProbeTrainAugmentation`. `02_vae` is a
   recorded exception in shape: dataset-agnostic, so the rule applies to its
   ImageFolder (ImageNet) train path only -- the MNIST path stays deterministic on
   both splits (cropping/flipping a digit can change its class). Not yet flipped
   to `conformant`: that needs a fleet-wide discover-all `aug` audit enumerating
   every probe method, not this hand-named family list.
6. **`b` — eval preprocessing.** DONE for every known case (branch
   `downstream/basic5-b-eval-crop`), across two waves:
   - Wrong crop at size 224: `cae`, `data2vec2`, `videomae`,
     `06_rotation_prediction`, `10_inst_disc` -- square resize with no centre
     crop -- now route their val split through
     `probe_transforms.basic5_eval_transform` (Resize shorter-side 256 +
     CenterCrop 224), the single implementation of the rule and the deterministic
     sibling of `basic5_train_transform`. 3/3 shared mutants + 1/1 per method.
   - Wrong crop at native resolution: `05_jigsaw_puzzle` (255),
     `09_jigsaw_puzzle_pp` (75), `vjepa2` (256), `vjepa2_ac` (256) -- square
     `Resize((s,s))` -- now route through the same transform **at their native
     size** (kept, not forced to 224), so only the crop form changed; each
     backbone's interpolation/normalise tail is preserved. 1/1 mutant per method.
   - The remaining native-resolution methods were **measured, not assumed**:
     `04_context_encoder`, `26_simmim`, `36_franca`, `cosmos3_super`, `sam3`,
     `image_gpt`, and `02_vae`'s ImageFolder path already use shorter-side
     Resize + CenterCrop (conformant, no change). `11_cpc` (fixed square source
     for its patch grid) and `02_vae`'s MNIST path (28×28 passthrough) are
     **deliberate**, recorded above. See the rule `b` conformance bullet.
   - The row stays `partial` for the same reason as `aug`: all known cases are
     resolved, but flipping to `conformant` should rest on a fleet-wide
     discover-all `b` audit enumerating every probe method, not this list.
7. **`opt` — probe optimiser (LR scaling / SGD recipe / cosine / epochs).**
   MECHANISM + PILOT DONE (branch `downstream/basic5-opt-reconcile`).
   `probe_optim` is the single implementation of the rule: `basic5_scaled_lr`
   (pure arithmetic, `lr = base_lr * batch/256`), `basic5_probe_optimizer`
   (SGD, momentum 0.9, weight decay 0, at the scaled LR), and
   `basic5_cosine_schedule` (cosine decay over the configured epochs). The rule
   is pinned once in `tests/test_probe_optim.py` (the scaling clause, the SGD
   recipe, the schedule, and a negative control that scaling actually happens
   away from batch 256). Historically each evaluator read the config LR directly,
   so only a batch-256 probe was correct and the batch-32 / batch-1024 configs
   trained at the wrong effective LR; this is the universally-missing part of the
   rule and the reason it leads with the shared mechanism. The per-method wiring
   check lives once in `tests/_probe_opt.py` (`run()` must build its
   optimiser/schedule through `probe_optim`), imported by each method's
   `TestTheProbeOptimizer`. Pilot: `cae` (batch 32, so its head trained at 8x the
   intended effective LR) now wires the shared builders;
   `mutations/cae-probe-opt.json` proves it non-vacuous (1/1: revert to the raw
   `torch.optim.SGD` with the config LR). Remaining: fan out the wiring across the
   ~50 template/custom evaluators (localised LR/epoch/batch/mean-centering
   residuals still to reconcile per method).

Items 5, 6, and the item-7 mechanism are done (each proven by mutation); those
rows stay `partial` until a fleet-wide discover-all audit enumerating every probe
method (for `opt`, until the wiring fan-out completes and every deviation is
reconciled or recorded). This section is the plan of record.

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

## AP and FT building blocks

`downstream.attention` provides shared readers from the attentive protocol:

- `QueryReader(C)` accepts caller-selected final tokens `[B, N, C]`, projects
  to width 512, applies one eight-head pre-LN cross-attention/MLP block to 32
  learned queries initialized with normal standard deviation `1/sqrt(512)`,
  and mean-pools them. The task's classifier is external.
- `SpatialAdapter(C)` operates on real `[B, C, H, W]` maps through width 256,
  one eight-head pre-LN self-attention/MLP block and a zero-initialized
  1x1 Conv output projection, added residually. The input projection is also
  a 1x1 Conv. Reuse the same instance at compatible scales.
  Global vectors and incompatible channel counts are rejected.
- Both use MLP ratio four and no dropout. The implementation uses GELU, affine
  LayerNorm, and residual connections around attention and MLP. Query and
  context have separate pre-attention norms in the cross-attention reader.
  These details follow the captured Basic5 reader implementation. They are
  tested explicitly; they do not resolve missing task definitions in the
  supplied protocol. Private reference files and their provenance stay in
  the task's external evidence, not in this repository.

`downstream.spatial_backbones.build_attentive_backbone(spec, device)` composes
the existing discovered frozen spatial provider with the shared adapter.
Calling `train()` on the enclosing model trains the adapter but keeps the
backbone in evaluation mode, with parameters, BatchNorm state and input
gradients frozen. A task head consumes the resulting spatial map as before.

`build_trainable_backbone(spec, device)` exposes a separate differentiable
timm `vit` path. It uses the same token-to-grid readout and checkpoint loading
as the frozen builder. Parent `train()` and `eval()` propagate in this path;
parameters and normalization state can update. Other providers are explicitly
unsupported for FT until their differentiable forward is verified. Toggling
`requires_grad` alone on a frozen provider is insufficient.

The default `build_frozen_backbone` and default task entrypoints remain frozen.
The shared timm checkpoint loader now rejects missing encoder weights and
unknown keys. Only the removed classifier's exact `head.weight` and `head.bias`
keys may be extra. An empty encoder path still means a random smoke model,
never a released-weight evaluation.

**Coverage boundary:** these are executable, tested components, not complete
AP/FT task recipes or canonical result eligibility. The opt-in attentive task
integration below composes these components. The five task recipes,
optimization/scheduling and full-dataset evaluation remain separate work.
Callers must select the final layer, remove excluded special tokens, preserve
real grids, and supply the unchanged task head. No global-to-spatial fallback,
temporal position policy, FPN geometry or depth-loss variant is invented here.
In particular, query attention without temporal position information is
permutation invariant; passing ordered video tokens alone does not establish
order-sensitive video recognition.

Tests in `tests/test_basic5_readers.py` exercise checkpoint round trips,
gradient and BatchNorm behavior, adapter identity at initialization, shared
scales, malformed inputs, and CUDA optimizer/save/reload paths. The mutation
spec `mutations/basic5-readers.json` checks the corresponding safeguards.

## Opt-in captured depth and video components

The existing `downstream.nyuv2` and `downstream.ssv2` entrypoints accept the
optional top-level configuration field `"profile": "capture_basic5_components"`.
Omitting it (or selecting `"legacy"`) retains the historical runner behavior.
An unknown profile is refused. Use the existing runner configuration and CLI;
all existing required fields remain required, including NYUv2's
`probe.head_hidden_dim` (unused by the single-convolution head).

This profile closes specific data/head/metric gaps, **not the complete canonical
training recipe**. It retains the existing frozen backbone and optimizer settings;
it does not establish AP/FT conformance, batch-scaled learning rates, warmup/cosine, or full result
eligibility. New-profile results always have `record_value: false` and
`canonical_eligible: false`, even without subset limits. A successful downstream
contract verifies execution and artifacts, not paper/protocol conformance.

### NYUv2

- Reads `labeled/splits.mat` alongside `labeled/nyu_depth_v2_labeled.mat`.
  Supports ordinary MATLAB files through SciPy and HDF5 MATLAB files through
  h5py, including referenced scalar IDs. The one-based `trainNdxs`/`testNdxs`
  values select actual samples; order is preserved. Both lists must be nonempty
  and form a complete, disjoint, in-range integer partition of the labelled
  file. Missing/malformed files never fall back to a positional split.
  Tiny synthetic partitions are allowed for tests; this does not certify a
  supplied split as the official 795/654 dataset. Results hash the split file.
- Uses the captured square image geometry, joint training horizontal flip,
  nearest-neighbor depth resize and finite inclusive 0.1--10.0 metre mask.
  The retained ImageNet normalization is not a universal model-specific policy.
- Replaces the legacy DPT head with a single 1x1 convolution, initialized with
  normal standard deviation 0.01 and zero bias. It bilinearly resizes logits
  with `align_corners=False`, then applies `softplus + 0.001`.
- Trains with float32 `mean(d^2) - 0.5 * mean(d)^2`, where
  `d = log(max(pred, 1e-6)) - log(max(target, 1e-6))` on valid pixels.
  There is no square root, scaling factor or per-image depth alignment.
  An empty mask returns differentiable zero, matching the captured loss.
- Emits RMSE (metres), AbsRel (ratio), and delta1/2/3 (percent, strict
  ratio thresholds `1.25`, `1.25^2`, `1.25^3`). Like the captured evaluator,
  it averages **per-batch metrics**, including zero metrics for empty masks.
  Results record `metric_aggregation: "batch_mean"`; this is batch-size
  dependent and must not be mixed with legacy `global_valid_pixels` metrics.
  An empty evaluation loader or nonfinite metric is refused.

### SSv2

- Splits decoded frames into rounded equal temporal segments; chooses a random
  frame inside each training segment and its lower center for evaluation.
  Short clips repeat valid frames in order. Empty clips and invalid counts
  are refused. `probe.num_frames` remains explicit (canonical value: 16).
- Training draws one random resized crop and applies it to all selected frames.
  Evaluation resizes the shorter side to 256 and center-crops to
  `probe.image_size` (canonical value: 224). No horizontal flip is introduced.
- By default the classifier averages frozen image-frame features. The explicit
  attentive option below replaces this readout. Native-video providers and
  video finetuning remain unsupported by this runner.

`tests/test_basic5_depth_temporal.py` tests actual split parsing, numerical
loss/metric fixtures, geometry, real video decoding, both CLI result contracts,
and a CUDA depth-head update with an unchanged backbone. Capture reference
comparisons and their private provenance are kept outside this repository.

## Opt-in captured segmentation and detection components

The same `capture_basic5_components` profile is available in `downstream.ade20k`
and `downstream.coco`. The default remains `legacy`. These additions preserve
the frozen encoder and existing optimizer; they do not implement the complete
canonical AP/FT recipe. Results always mark `record_value: false` and
`canonical_eligible: false` in this profile, even on a full dataset.

### ADE20K geometry

Training draws a uniform scale in [0.5, 2.0], rounds the scaled dimensions and
clamps each to at least `probe.image_size`, then applies a shared random crop
and 50% horizontal flip to image and mask. Images use bilinear resize; masks
use nearest neighbor. Evaluation retains deterministic square resize.
ImageNet normalization and the 0/255 ignore-label mapping remain unchanged.
Color jitter is not added; this matches the captured zero-jitter configuration.

### COCO geometry, pyramid and metrics

Training flips images and bounding boxes together with probability 0.5;
evaluation does not flip. A frozen shared `vit` provider with `patch_size: 16`
feeds four trainable 1x1 projections: bilinear upsampling by four and two
(`align_corners=False`), the original map, and 2x2 max pooling. ROI pooling
uses all four maps; padding is to multiples of 32. Other providers and strides
are refused until their geometry and normalization are verified.

Set `detector.anchor_sizes` to four positive sizes, one per map (captured
configuration: `[32, 64, 128, 256]`). The shared timm encoder receives the
existing ImageNet normalization exactly once, through the detector transform.
The captured model-specific normalization wrapper is not ported by this change.

Metrics include `coco_map`, `coco_map_50`, `coco_map_75`, `coco_map_small`,
`coco_map_medium`, and `coco_map_large`. Values retain the existing contract's
ratio units, unlike the captured evaluator's percentages. Undefined area bins
preserve COCOeval's `-1`; empty predictions emit zero metrics. Results explicitly
record the units. Evaluation remains restricted to the evaluated image IDs.

`tests/test_basic5_seg_detection.py` covers paired geometry, numerical pyramid
outputs/gradients, real COCO evaluation, both CLI contracts, incompatible
configurations, and a CUDA detector update that leaves the encoder unchanged.
`mutations/basic5-seg-detection.json` verifies detection of broken behavior.
Private captured-source comparisons and provenance remain outside Git.

## Opt-in attentive task integration

All four downstream entrypoints accept `"adaptation": "attentive"` together
with `"profile": "capture_basic5_components"`. Unknown adaptation values and
attentive/legacy combinations are rejected. Omitted adaptation defaults to
`"frozen"`, preserving the previous heads, initialization and optimizer path.
Add these two fields to an otherwise complete existing downstream configuration.

- ADE20K and NYUv2 apply the residual spatial adapter after the frozen encoder
  and before the unchanged single-convolution head.
- COCO applies one adapter to the stride-16 map before the existing trainable
  four-level pyramid. The same verified shared `vit`/stride-16 restriction holds.
- SSv2 spatially averages each image frame's patch features and passes the
  resulting `[batch, frames, channels]` tokens to the query reader. There is no
  L2 normalization or temporal averaging before attention. The classifier has
  512 input channels and zero-initialized weight and bias, following the captured
  attentive classifier. No temporal positional encoding is invented; the query
  reader remains permutation invariant.

The encoder forward alone runs without gradients. The adapter/reader, task head
and detection pyramid remain trainable, including during the real CLI training
loop. Optimizers include all trainable parameters and exclude the frozen
encoder. Attentive training clips their combined gradient norm to 1.0.
ADE20K/NYUv2/SSv2 retain the runner's AdamW and COCO retains SGD.

**These are partial component runs.** By default, learning rates remain explicit unscaled
runner values; existing weight decay and epoch schedules remain unchanged.
Canonical optimizer recipes, warmup, batch scaling, multi-seed aggregation,
model-specific readouts and full-dataset score reproduction are not certified.
Results record `adaptation` and always retain `record_value: false` and
`canonical_eligible: false` in the component profile. Do not put these scores
in the canonical AP table.

`tests/test_basic5_attentive_tasks.py` exercises all four actual CLI entrypoints,
optimizer membership, gradient clipping, unchanged encoders, two-step attention
updates on CPU/CUDA, frame-token readout, and refusal of legacy/unknown options.
The CI downstream dependency job invokes it explicitly. Private captured-code
comparisons cover initialization, outputs, gradients and optimizer updates;
their source copies and provenance remain outside Git.

## Opt-in full-gradient task components

The four downstream CLIs also accept `"adaptation": "finetune"` with
`"profile": "capture_basic5_components"` and `backbone.kind: vit`. This selects
the existing differentiable timm provider. Unsupported frozen-only providers
and legacy/finetune configurations are refused. The default remains frozen;
the attentive path retains its frozen encoder and trainable reader.

For ADE20K and NYUv2, gradients flow through the unchanged spatial readout and
single-convolution heads to the encoder. COCO trains the encoder, four-level
pyramid and detector together; its verified stride-16 restriction still applies.
No attention reader is added. Training/evaluation mode propagates to the encoder,
and evaluation respects the caller's no-gradient context. Every batch recomputes
features; no feature cache is used. The optimizer includes the encoder and head,
and their combined gradient norm is clipped to 1.0.

For SSv2 image backbones, the captured FT composition uses float32 per-frame
spatial means, L2-normalizes each frame, then averages over time. Its linear head
starts at zero. This differs from the historical unnormalized frozen readout;
the explicit adaptation field keeps the runs distinguishable. The attention
path still receives unnormalized frame tokens. Native-video FT is unsupported.

**This is full-gradient execution, not a complete `BASIC5_FINETUNE_v1` recipe.**
The default unscaled runner LR, optimizer weight decay, schedules and data
augmentation are retained. Layer decay, zero-decay parameter groups, FT color
jitter, strong video augmentation and effective-batch accumulation remain
unimplemented here. The generic timm final-layer grid does not certify a
model-specific multi-layer representation. All component results remain
noncanonical and nonrecordable, including full-data runs; random tiny-model
tests are not released-weight benchmarks or paper-score reproduction.

`tests/test_basic5_finetune_tasks.py` covers CPU/CUDA encoder and head updates,
all four real CLIs, optimizer membership, gradient clipping, video readout,
checkpoint state round trips, evaluation autograd and rejected configurations.
The downstream CI dependency job executes it, and
`mutations/basic5-finetune-tasks.json` tests the guards. Private evidence records
comparisons with captured task compositions using identical differentiable
feature providers; it does not assert equivalence of different pretrained models.

## Opt-in frozen/AP optimizer components (2026-09-19)

All four downstream CLIs accept top-level `optimizer_profile: basic5_frozen_v1`
with `profile: capture_basic5_components` and `adaptation: frozen` or
`attentive`. Set `probe.lr` (COCO: `detector.lr`) to the JSON string
`"protocol"`. Numeric LR overrides are refused, so an old requested LR cannot
silently be ignored. Omitting `optimizer_profile` preserves the old numeric-LR
optimizer and data-loader behavior, including existing FT component runs.

| Task | Adaptation | Optimizer | Base LR / reference batch | Weight decay |
|---|---|---|---|---|
| ADE20K, NYUv2 | frozen | AdamW | 0.001 / 8 | 0.0001 |
| ADE20K, NYUv2 | attentive | AdamW | 0.001 / 8 | 0.05 |
| COCO | frozen or attentive | SGD | 0.02 / 16 | 0.0001 |
| SSv2 | frozen | SGD | 0.1 / 256 | 0.0001 |
| SSv2 | attentive | AdamW | 0.001 / 256 | 0.05 |

SGD uses momentum 0.9; AdamW uses betas (0.9, 0.999). Every trainable
head/reader parameter is included; frozen encoder parameters are excluded.
These values agree with the supplied protocols and the inspected captured
optimizer/configuration. This is optimizer parity, not score reproduction.

Realized LR is `base_lr * batch_size / reference_batch`. This implementation
supports one process and one physical batch per update, with no accumulation.
Both `WORLD_SIZE` and any initialized distributed process group must indicate
one process. Training drops incomplete final batches, matching the captured
loader and keeping the realized batch constant. Validation retains every
sample. A training set smaller than one batch fails rather than producing a
successful result without updates. Batch size must be a positive JSON integer.

`results.json` records an `optimization` object containing the selected profile,
optimizer, base/reference/realized LR, weight decay, momentum or betas, effective
batch, world size, accumulation steps, `drop_last: true`, and `schedule: none`.
The manifest hashes this result along with the original config. The existing
frozen-state behavior and AP gradient clipping remain in force.

**This is not a complete LP/AP recipe.** LR remains constant throughout this
component run. Warmup/decay schedules, accumulation, native-video paths and
complete model-specific feature validation remain separate work. FT is refused
for this optimizer profile until parameter groups are verified. Results remain
`canonical_eligible: false` and `record_value: false`, even on full datasets.
No main-table eligibility flag is relaxed by this addition.

The captured cosine scheduler scales nominal warmup/floor endpoints with batch
scaling, while the supplied protocol describes fixed endpoints. That discrepancy
remains pending; this component does not choose either scheduler interpretation.
`tests/test_basic5_optimization.py` covers all eight task/adaptation CLI paths,
numerical updates, frozen state, explicit refusals, full-batch behavior and
result integrity. CPU tests and separately marked CUDA tests use synthetic
fixtures; they are not released-weight benchmarks.
