# Porting roadmap — the 38 Step 1&2 methods, order, and status

> **Legacy terminology:** numbered Step 1-4 labels in this file are historical
> code/experiment identifiers, not manuscript section numbers. See the
> [ASIS, CTRL and manuscript comparison map](PAPER_TERMINOLOGY.md) before matching a recipe to a paper result.

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

Last updated: 2026-08-11

This is the fact-based plan for porting the **Step 1&2** visual-SSL methods. It is
externalised here so the plan survives across sessions. Generative-AR methods
(`var`, `mar`, `image_gpt`) belong to **other steps**, not this list, and are
already ported as pilots for the submodule / eval-only / download machinery.

## DINOv3 scope update (2026-09-25)

The historical rows below describe the original core-only port. The optional
[Step-4 components](DINOV3_STEP4.md) now integrate four sharing layouts and a
frozen-teacher Gram stage. They do not establish distributed or paper-score
reproduction; see the [current gap ledger](PAPER_REPRODUCTION_GAPS.md).

## The 38 methods (Step 1&2), and how the capture holds them

IDv2 update (2026-09-25): the existing InstDisc ViT adapter now has seven
explicit [Step-4 component profiles](IDV2_COMPONENTS.md), including complete
component checkpoint continuation. Historical core-only descriptions below do
not cover this addition. Native distributed integration and full-score reruns in this port remain
unverified.

The private archive (`snapshots` branch) has a
**self-contained `methods/<n>_<name>/` directory for every one of the 38**, and
~32 of them carry the lab's **own** model code under `models/*.py` (an
independent implementation following the paper, torch-based — the same
license-clean pattern as `mae`/`image_gpt`/the clean six). Those port
**self-contained on the existing `pretrain → encoder.pt → linear_probe` contract**;
no submodule, no download, no noncommercial entanglement. Six have no `models/`
(`8_split_brain`(done, `split_brain` -- a plain AlexNet, self-contained after all),
`msn`, `vjepa`, `franca`(done), `lejepa`, `clip`)
and need a submodule / eval-only treatment (and, for msn/vjepa/lejepa, a
noncommercial-licence decision; `clip` is MIT, so no such decision).

**Verify per method at port time** (do not assume): the model header's licence
(lab-own vs a copy of author code → the latter needs a submodule reference), the
exact dependencies, the backbone, and pretrain reproducibility.

## Dependency tiers (measured from the capture's requirements.txt)

- **A — torch/torchvision/numpy/PyYAML** (reuse `image_gpt`/`mae` locks): `dino`, `pirl`.
- **B — + timm** (reuse a mar-style lock): mocov1/2/3, simclrv1/2, byol, sela, inst_disc, cpc, cmc, rotation, simmim, dinov2, dinov3, ijepa, nepa.
- **B' — + huggingface_hub** (var-style): `aim`.
- **C — heavy special deps**: ~~`3_colorization` (opencv/scikit-image)~~ **ported torch-only as `colorization`: its code imports neither opencv nor scikit-image despite the capture's requirements.txt naming them (measured); the Lab conversion and ab-quantisation are numpy**. `beit` (DALL-E tokeniser). `7_deepcluster` **ported as `deepcluster` with faiss** (the paper-target k-means backend, confirmed required; faiss-gpu is linux-x86_64-only, so the method is GPU/x86_64-only and faiss lives in the CUDA lock via a `# gpu-only` marker). `9_jigsaw_puzzle_pp`'s **knowledge-transfer stage** (faiss-GPU k-means, mandatory — the capture's `cluster_and_pseudolabels.py` explicitly refuses the CPU/sklearn fallback; **measured, contradicts the earlier "tier B / +timm" guess**) is **now ported** as the `knowledge_transfer` stage of `jigsaw_puzzle_pp`, GPU/x86_64-only with faiss in the CUDA lock (`# gpu-only`). The VGG16 **pretext** stage alone is tier-A (torch/torchvision/numpy/Pillow/PyYAML) and CPU-portable.
- **D — no `models/` (submodule / eval-only, often noncommercial)**: `msn`, `vjepa`, `lejepa`. (`8_split_brain` was here on the "no `models/` dir" signal, but measuring showed its flat `model.py` is a plain two-branch AlexNet -- self-contained torch-only, now ported as `split_brain`; the label was not evidence.)

TensorBoard / tqdm / wandb logging is dropped (the port owns a thin single-process
loop), as in every prior port.

## Fact-based order (reuse templates and locks; keep CI free)

Port self-contained + light-dep + shared-template methods first; heavy/ViT next;
special-dep / submodule / eval-only / noncommercial last.

- **Group 1 — ResNet/CNN pretext & contrastive** (reuse the simsiam/swav/barlow ResNet + linear-probe template): jigsaw, rotation, jigsaw++ (VGG16 pretext only), inst_disc, pirl, deepcluster, cpc, cmc, mocov1/2/3, simclrv1/2, sela, byol.
- **Group 2 — ViT** (reuse the ibot/mae ViT template): simmim, dino, dinov2, dinov3, ijepa, nepa, aim.
- **Group 3 — special deps / submodule / eval-only / licence decision**: colorization (ported torch-only, `colorization` -- the opencv/skimage tag was a requirements.txt mislabel), beit (ported torch-only, `beit` -- its DALL-E dVAE tokeniser is a hash-pinned download lazy-imported for a real run; the smoke uses a random tokeniser), deepcluster (ported with faiss as `deepcluster`, GPU/x86_64-only -- the first method whose closure adds a non-torch dep), `9_jigsaw_puzzle_pp`'s faiss-GPU knowledge-transfer stage (ported as the `knowledge_transfer` stage of `jigsaw_puzzle_pp`, GPU/x86_64-only: cluster → pseudo-labels → AlexNet cluster-cls), split_brain (ported torch-only, `split_brain` -- the "no models/" tag was not a blocker), msn, vjepa, lejepa.

## Numbering: capture number vs this list's number

The third column records the original source directory, including its historical number. The final column names the current public port. The list and source numbers follow different orders.

| list # | method | capture dir | number matches? | status |
|---|---|---|---|---|
| 1 | VAE | `2_vae` | no | ported (`vae`) |
| 2 | Context Prediction | `1_context_prediction` | no | ported (`context_prediction`) |
| 3 | Colorization | `3_colorization` | **yes** | ported (`colorization`; torch-only -- opencv/skimage were listed in the capture's requirements.txt but its code imports neither, measured) |
| 4 | Context Encoder | `4_context_encoder` | **yes** | ported (`context_encoder`) |
| 5 | Jigsaw Puzzles | `5_jigsaw_puzzle` | **yes** | ported (`jigsaw_puzzle`) |
| 6 | Rotation Prediction | `6_rotation_prediction` | **yes** | ported (`rotation_prediction`) |
| 7 | DeepCluster | `7_deepcluster` | **yes** | ported (`deepcluster`; faiss clustering, **GPU / x86_64-linux only** -- faiss-gpu has no cross-platform wheel, so it lives in the CUDA lock and the method is exempt from the CPU lock via the `# gpu-only` marker) |
| 8 | SplitBrain | `8_split_brain` | **yes** | ported (`split_brain`; torch-only -- the capture's flat `model.py` (no `models/` dir) is a plain two-branch AlexNet, not a submodule/eval-only method; scipy/skimage were imported but the released target is numpy argmin, measured) |
| 9 | Jigsaw Puzzle++ | `9_jigsaw_puzzle_pp` | **yes** | ported (`jigsaw_puzzle_pp`, VGG16 pretext + faiss-GPU knowledge transfer: cluster → pseudo-labels → AlexNet cluster-cls) |
| 10 | InstDisc | `10_inst_disc` | **yes** | ported (`inst_disc`) |
| 11 | CPC | `11_cpc` | **yes** | ported (`cpc`) |
| 12 | CMC | `12_cmc` | **yes** | ported (`cmc`) |
| 13 | MoCo v1 | `13_mocov1` | **yes** | ported (`mocov1`) |
| 14 | PIRL | `33_pirl` | no | ported (`pirl`; torch-only -- ResNet-50 + jigsaw + memory-bank NCE, the third memory-bank method after inst_disc/cmc; encoder.pt is the trunk, the bank lives in the loss module) |
| 15 | SimCLR v1 | `14_simclrv1` | no | ported (`simclrv1`) |
| 16 | MoCo v2 | `15_mocov2` | no | ported (`mocov2`) |
| 17 | SimCLR v2 | `16_simclrv2` | no | ported (`simclrv2`) |
| 18 | SwAV | `17_swav` | no | ported (`swav`) |
| 19 | SeLa | `18_sela` | no | ported (`sela`) |
| 20 | BYOL | `19_byol` | no | ported (`byol`) |
| 21 | SimSiam | `20_simsiam` | no | ported (`simsiam`) |
| 22 | Barlow Twins | `21_barlow_twins` | no | ported (`barlow_twins`) |
| 23 | MoCo v3 | `22_mocov3` | no | ported (`mocov3`; the **first timm-locked port** -- timm supplies the ViT base class, but the ViT is built from scratch so the run stays hermetic) |
| 24 | DINO | `23_dino` | no | ported (`dino`; torch-only -- DINO ships its own ViT, NOT timm, measured; encoder.pt is the teacher backbone) |
| 25 | BEiT | `24_beit` | no | ported (`beit`; torch-only ViT (BEiT's own LayerScale ViT, NOT timm) + masked image modeling over DALL-E dVAE visual tokens; the tokeniser is a hash-pinned `encoder.pkl` download imported lazily via `dall_e` for a real run, and a random torch-only tokeniser for the hermetic smoke, so CI downloads nothing; encoder.pt is the backbone trunk (mask token + MIM head excluded)) |
| 26 | MAE | `25_mae` | no | ported (`mae`) |
| 27 | SimMIM | `26_simmim` | no | ported (`simmim`; needs timm for Swin -- reuses the `mocov3` lock; the `transformers` dep was docstring-only, measured; encoder.pt is the bare Swin) |
| 28 | iBOT | `27_ibot` | no | ported (`ibot`) |
| 29 | MSN | `34_msn` | no | ported (`msn`; **submodule-import**: pins `facebookresearch/msn` (third_party/msn @4388dc1, CC BY-NC 4.0, research-use) and imports its ViT (src/deit) + MSN loss (src/losses) + optimiser (src/msn_train.init_opt) into a single-process trainer; anchor/target EMA + prototypes + me-max; encoder.pt = anchor ViT trunk (fc excluded). The multi-view aug is reimplemented (upstream trips Pillow 12.x); DDP/submitit/cyanure dropped, shared ARSSL probe; torch-only closure. Nothing under the licence is copied) |
| 30 | DINOv2 | `28_dinov2` | no | ported (`dinov2`; **two comparisons**. Step-1 (as-is): pinned `third_party/dinov2` submodule (xformers disabled -> torch-only), official ViT-g/14 LVD-142M weights hash-pinned via `bin/fetch-weights.py` (the original LVD-142M data is not public, so the as-is row reuses the official backbone). Step-2 (unified): a from-scratch ViT-B/16 (DINO+iBOT+KoLeo) on ImageNet-1k, ported additively) |
| 31 | I-JEPA | `29_ijepa` | no | ported (`ijepa`; torch-only -- I-JEPA ships its own ViT, NOT timm, measured; trains from scratch on ImageNet; encoder.pt is the EMA target encoder) |
| 32 | AIM | `30_aim` | no | ported (`aim`; **two comparisons**. Step-1 (as-is): pinned `third_party/ml-aim` submodule (Apple's `aim`), official AIM-600M ViT-H/14 (DFN-2B+) backbone hash-pinned via `bin/fetch-weights.py`, probed on the last-6-block average patch-mean-pooled (DFN-2B+ is not public, so the as-is row reuses the official backbone). Step-2 (unified): a from-scratch ViT-B/16 (prefix-LM next-patch MSE) on ImageNet-1k, ported additively (the lab's own torch-only AIM re-impl, not apple's code/weights). Licence apple-amlr binds only the Step-1 probe = non-commercial research; nothing under it copied (submodule + download), academic-research use only) |
| 33 | V-JEPA | `35_vjepa` | no | ported (`vjepa`; **submodule-import** of `facebookresearch/jepa` (third_party/jepa @51c59d5, CC BY-NC 4.0, research-use). Ports the capture's step-2 image adaptation (num_frames=1 image ViT-B/16, from scratch on ImageNet) -- a genuine comparable row, NOT the pretrain caveat probe of the released video model. Imports init_video_model + 3D MaskCollator + apply_masks; context/target EMA + latent smooth-L1 prediction; encoder.pt = EMA target encoder. torch-only closure; single-process (DDP/TensorBoard dropped); shared ARSSL probe. Nothing under the licence is copied) |
| 34 | Franca | `36_franca` | no | ported (`franca`) |
| 35 | DINOv3 | `31_dinov3` | no | ported (`dinov3`; torch-only, from-scratch on ImageNet -- the capture's **step 2** unified SSL comparison, DINOv3 **core** objective: own ViT (register tokens + axial RoPE) + DINO (Sinkhorn centring) + iBOT + KoLeo, EMA teacher, multi-crop; encoder.pt = teacher backbone (prefix stripped). The capture's step 1 (HF-**gated** official weights) and the released **Gram anchoring** stage (`gram.mode: core_only`) are excluded; a from-scratch re-implementation, so no Meta code/weights are used) |
| 36 | LeJEPA | `37_lejepa` | no | ported (`lejepa`; torch+timm, from-scratch on ImageNet -- a timm ViT + projector trained by SIGReg (Epps-Pulley quadrature + random slicing, reimplemented locally, no external package) + a cross-view invariance loss; encoder.pt is the bare backbone, prefix stripped) |
| 37 | NEPA | `32_nepa` | no | ported (`nepa`; torch-only -- its own ViT with 2D RoPE / QK-norm / causal autoregressive predictor, from scratch on ImageNet; encoder.pt is the EMA model) |
| 38 | CLIP | `38_clip` | **yes** | ported (`clip`; a 38th method added to the capture after the original decade list. **submodule-import** of `openai/CLIP` (third_party/CLIP @d05afc4, **MIT**, imported not copied). **two comparisons**: Step-1 (as-is) freezes the released official ViT-B/32 image tower (a sha256-pinned download via `bin/fetch-weights.py`) and probes its pooled image embedding (CLIP's 400M image-text data is not public, so the as-is row reuses the released checkpoint); Step-2 trains a CLIP ViT-B/16 from scratch on ImageNet-1k, pairing each labeled image with an official ImageNet class-name prompt (symmetric image-text contrastive loss), probed frozen at milestones 100/200/300. Step-2 is a **supervised label-text adaptation**, stamped `supervised_label_text_adaptation=true` / `main_vssl_comparability=false` -- a CLIP-adaptation reference, not a comparable unlabeled number. Model constructor + BPE tokenizer + ImageNet prompts imported from the submodule; single-process (DDP/all-gather dropped); shared ARSSL probe. MIT, so nothing under a noncommercial licence and nothing copied) |

## Historical source identifiers

The original ports used the Capture directory names. The table retains those source identifiers. Current public directories and commands use [unnumbered names](METHOD_NAMING.md).

**Porting order used:** by capture-directory number, torch-only (tier-A)
self-contained methods first, then the special-dep / submodule / eval-only tier
(`msn`, `vjepa`) last. All are now done.
**Correction (2026-08-10, measured):** an earlier note here claimed "the clean
from-scratch tier is exhausted" and filed `lejepa` under the submodule tier.
Re-reading the capture showed that was wrong -- `lejepa` imports no external
package at runtime (SIGReg is reimplemented locally; only torch/torchvision/timm/
yaml), trains from scratch on ImageNet, and needs no downloaded weights, so it is
a clean torch+timm self-contained port. It is **now ported** (`lejepa`). With
it done, the genuinely-clean from-scratch tier is exhausted. On 2026-08-10 the user
ruled that **non-commercial-licensed code may be used for academic research** (with
careful licence documentation), which unblocks the remaining eval-only /
non-commercial tier. `aim` is **now ported** (an as-is Step-1 download probe plus a from-scratch unified Step-2; see below).
`dinov3` is **now ported** (2026-08-10): its eval-only download path is blocked
(HF login-gated weights), but its capture **step 2** -- the from-scratch unified SSL
comparison -- is self-contained torch-only code (own ViT + DINO/iBOT/KoLeo losses,
measured no timm/transformers), so it ports as a clean from-scratch method (DINOv3
**core**; the released Gram anchoring stage is excluded via the capture's
`gram.mode: core_only`, and no Meta code/weights are used). `msn` is **now
ported** (2026-08-11): rather than run the official DDP `main.py` + cyanure
`linear_eval`, it pins `facebookresearch/msn` as a submodule and IMPORTS its ViT +
MSN loss + optimiser into a single-process trainer (the imported `src` modules are
torch/torchvision/numpy/PIL only -- no apex/opencv/submitit/cyanure), reimplements
the multi-view aug (the upstream one trips Pillow 12.x with a Tensor blur radius),
and uses the shared ARSSL probe; CC-BY-NC documented as research-use, nothing
copied. `vjepa` is **now ported** (2026-08-11), the LAST method: rather than the
pretrain caveat probe of the released video model, it ports the capture's step-2 image
adaptation -- pin `facebookresearch/jepa` as a submodule and import init_video_model
+ the 3D mask collator to train a V-JEPA-objective image ViT-B/16 (num_frames=1) from
scratch on ImageNet, a genuine comparable row; CC-BY-NC documented, nothing copied.
**ALL 38 Step-1&2 methods are now ported** (the original 37-method decade list
plus `clip`, a CLIP method the capture added afterwards; the three
generative-AR pilots `image_gpt`, `mar`, `var` are separate — see the intro —
bringing the repository to 41 ported methods in total). `beit` is ported (torch-only ViT + a hash-pinned DALL-E dVAE
tokeniser download, lazy-imported for a real run via the `third_party/dall_e`
submodule; the smoke uses a random tokeniser). **The eval-only-download phase is
underway**: `dinov2` and `aim` are ported in the Franca-style
frozen-backbone-download shape (a pinned submodule + hash-pinned weights via
`bin/fetch-weights.py`, CONTRACT §7); `aim` adds `huggingface_hub` to the closure
(the ml-aim model module needs it) and pins Apple's `aim` under `third_party/ml-aim`.
(Verify each by **measuring** the
capture before porting, never from the label; ViT/Swin-based ones need care —
**measure whether timm is pretrain-essential**: `mocov3` (subclasses timm's ViT)
and `simmim` (wraps timm's Swin) needed it, but `dino`, `ijepa` and
`nepa` do NOT (they ship their own ViT), so the label "ViT/Swin ⇒ timm" is not
evidence. When timm is needed, reuse the `mocov3` lock; when not, reuse the
torch-only `jigsaw_puzzle` / `byol` closure). **Also measure whether "step 1"
is genuine from-scratch training or a pretrained download**: `dinov2`, `aim`
and `dinov3` all turned out to be eval-only downloads (their pretraining data is
not public, so step 1 loads official checkpoints via torch.hub / HuggingFace /
`ml-aim`) -- ported (`dinov2`) or to be ported in the Franca-style
frozen-backbone-download tier (CONTRACT §7, `bin/fetch-weights.py`).
Ported under this decision so far: `simclrv1` (list #15), `mocov2`
(list #16), `simclrv2` (list #17), `sela` (list #19), `byol` (list #20),
`mocov3` (list #23, the first timm-locked port), `dino` (list #24,
torch-only -- its own ViT), `simmim` (list #27, timm Swin -- reuses the mocov3
lock), `ijepa` (list #31, torch-only -- its own ViT, from-scratch on ImageNet),
`pirl` (list #14, torch-only -- ResNet + jigsaw + memory bank, the third
memory-bank method), `nepa` (list #37, torch-only -- its own 2D-RoPE ViT with a
causal autoregressive predictor; the last clean from-scratch port), `dinov2`
(list #30, eval-only download -- pinned dinov2 submodule + hash-pinned ViT-g/14
weights, the first of the download phase).

**(historical) HOLD rule (user decision pending, 2026-08-06):** only port methods
whose capture directory number equals this list's number — TRUE only for numbers
3–13, which are now all ported (colorization, context_encoder, jigsaw, rotation,
deepcluster, split_brain, jigsaw++ incl. its faiss-GPU knowledge-transfer stage,
inst_disc, cpc, cmc, mocov1). `simclrv1` (list #15) is the first port taken
under the 2026-08-09 keep-capture-numbering decision.

## Per-port procedure (strict TDD, as used for every prior method)

1. Read the capture's `methods/<n>_<name>/` (model, trainer, eval, configs) **and**
   the closest already-ported method as a template.
2. Write the failing tests first; confirm RED.
3. Implement: model(s), thin single-process trainer (drop DDP/TensorBoard,
   resolve the device), evaluator, adapter (pretrain + linear_eval), configs,
   requirements + CPU/cu130 locks, Dockerfile, provenance, README.
4. Mutation spec, all killed (proves the tests bite — required when tests are
   written after any code).
5. Base CPU suite `EXIT=0`, then the **whole suite under a torch venv** (the CI
   `locked`-equivalent — `PYTHONPATH=. .venvs/<m>/bin/python -m unittest discover
   -s tests`; the base run skips deps-gated tests and hides in-process collisions).
6. GPU verify on the available device; `verify-environment` exact.
7. Commit gated on the suite; open a PR; confirm CI green (incl. container).

## Step-4 continuation update (2026-09-25)

H1S200, H1CORE and H1JA now have single-process component paths and versioned
checkpoint continuation. See [the Step-4 guide](DINOV3_STEP4.md) for exact
interfaces and limits. Native distributed checkpoint conversion, full distributed
training/BF16 and score reproduction remain open; IDv2 banks and the remaining
Basic5/Extend work are the next major integration groups.

## Basic5 provider update (2026-09-26)

The preceding IDv2-bank gap was superseded by the [seven component profiles](IDV2_COMPONENTS.md).
[SAM3 trunk and Cosmos3 final-merger components](BASIC5_PATCH_PROVIDERS.md)
now add ImageNet frozen LP and ADE20K/NYUv2/SSv2 frozen/FT paths. These updates
reduce implementation gaps; full-data scores, native distributed execution,
other family/AP/COCO/Extend pipelines remain partly unported or unreconciled
in this package. These are packaging and validation gaps, not a statement
that the corresponding experimental implementations are absent.
