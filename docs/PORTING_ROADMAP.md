# Porting roadmap — the 37 Step 1&2 methods, order, and status

Last updated: 2026-08-06

This is the fact-based plan for porting the **Step 1&2** visual-SSL methods. It is
externalised here so the plan survives across sessions. Generative-AR methods
(`var`, `mar`, `image_gpt`) belong to **other steps**, not this list, and are
already ported as pilots for the submodule / eval-only / download machinery.

## The 37 methods (Step 1&2), and how the capture holds them

The capture (`gatheluck/10YearVisualSSLCapturePrivate`, `snapshots` branch) has a
**self-contained `methods/<n>_<name>/` directory for every one of the 37**, and
~32 of them carry the lab's **own** model code under `models/*.py` (an
independent implementation following the paper, torch-based — the same
license-clean pattern as `25_mae`/`image_gpt`/the clean six). Those port
**self-contained on the existing `step1 → encoder.pt → linear_probe` contract**;
no submodule, no download, no noncommercial entanglement. Five have no `models/`
(`8_split_brain`(done, `08_split_brain` -- a plain AlexNet, self-contained after all),
`34_msn`, `35_vjepa`, `36_franca`(done, eval-only), `37_lejepa`)
and need a submodule / eval-only treatment (and, for msn/vjepa/lejepa, a
noncommercial-licence decision).

**Verify per method at port time** (do not assume): the model header's licence
(lab-own vs a copy of author code → the latter needs a submodule reference), the
exact dependencies, the backbone, and step-1 reproducibility.

## Dependency tiers (measured from the capture's requirements.txt)

- **A — torch/torchvision/numpy/PyYAML** (reuse `image_gpt`/`25_mae` locks): `23_dino`, `33_pirl`.
- **B — + timm** (reuse a mar-style lock): mocov1/2/3, simclrv1/2, byol, sela, inst_disc, cpc, cmc, rotation, simmim, dinov2, dinov3, ijepa, nepa.
- **B' — + huggingface_hub** (var-style): `30_aim`.
- **C — heavy special deps**: ~~`3_colorization` (opencv/scikit-image)~~ **ported torch-only as `03_colorization`: its code imports neither opencv nor scikit-image despite the capture's requirements.txt naming them (measured); the Lab conversion and ab-quantisation are numpy**. `24_beit` (DALL-E tokeniser). `7_deepcluster` **ported as `07_deepcluster` with faiss** (the paper-target k-means backend, confirmed required; faiss-gpu is linux-x86_64-only, so the method is GPU/x86_64-only and faiss lives in the CUDA lock via a `# gpu-only` marker). `9_jigsaw_puzzle_pp`'s **knowledge-transfer stage** (faiss-GPU k-means, mandatory — the capture's `cluster_and_pseudolabels.py` explicitly refuses the CPU/sklearn fallback; **measured, contradicts the earlier "tier B / +timm" guess**) is **now ported** as the `knowledge_transfer` stage of `09_jigsaw_puzzle_pp`, GPU/x86_64-only with faiss in the CUDA lock (`# gpu-only`). The VGG16 **pretext** stage alone is tier-A (torch/torchvision/numpy/Pillow/PyYAML) and CPU-portable.
- **D — no `models/` (submodule / eval-only, often noncommercial)**: `34_msn`, `35_vjepa`, `37_lejepa`. (`8_split_brain` was here on the "no `models/` dir" signal, but measuring showed its flat `model.py` is a plain two-branch AlexNet -- self-contained torch-only, now ported as `08_split_brain`; the label was not evidence.)

TensorBoard / tqdm / wandb logging is dropped (the port owns a thin single-process
loop), as in every prior port.

## Fact-based order (reuse templates and locks; keep CI free)

Port self-contained + light-dep + shared-template methods first; heavy/ViT next;
special-dep / submodule / eval-only / noncommercial last.

- **Group 1 — ResNet/CNN pretext & contrastive** (reuse the simsiam/swav/barlow ResNet + linear-probe template): jigsaw, rotation, jigsaw++ (VGG16 pretext only), inst_disc, pirl, deepcluster, cpc, cmc, mocov1/2/3, simclrv1/2, sela, byol.
- **Group 2 — ViT** (reuse the ibot/mae ViT template): simmim, dino, dinov2, dinov3, ijepa, nepa, aim.
- **Group 3 — special deps / submodule / eval-only / licence decision**: colorization (ported torch-only, `03_colorization` -- the opencv/skimage tag was a requirements.txt mislabel), beit, deepcluster (ported with faiss as `07_deepcluster`, GPU/x86_64-only -- the first method whose closure adds a non-torch dep), `9_jigsaw_puzzle_pp`'s faiss-GPU knowledge-transfer stage (ported as the `knowledge_transfer` stage of `09_jigsaw_puzzle_pp`, GPU/x86_64-only: cluster → pseudo-labels → AlexNet cluster-cls), split_brain (ported torch-only, `08_split_brain` -- the "no models/" tag was not a blocker), msn, vjepa, lejepa.

## Numbering: capture number vs this list's number

The port directories follow the **capture's** numbering (e.g. `17_swav`,
`25_mae`, `36_franca`). That numbering **matches this list only for numbers
3–13**; from #14 on it diverges (the list inserts PIRL at #14, shifting the
rest — e.g. capture `14_simclrv1` ↔ list #14 = PIRL; capture `17_swav` ↔ list
#18; capture `25_mae` ↔ list #26).

| list # | method | capture dir | number matches? | status |
|---|---|---|---|---|
| 1 | VAE | `2_vae` | no | ported (`02_vae`) |
| 2 | Context Prediction | `1_context_prediction` | no | ported (`01_context_prediction`) |
| 3 | Colorization | `3_colorization` | **yes** | ported (`03_colorization`; torch-only -- opencv/skimage were listed in the capture's requirements.txt but its code imports neither, measured) |
| 4 | Context Encoder | `4_context_encoder` | **yes** | ported (`04_context_encoder`) |
| 5 | Jigsaw Puzzles | `5_jigsaw_puzzle` | **yes** | ported (`05_jigsaw_puzzle`) |
| 6 | Rotation Prediction | `6_rotation_prediction` | **yes** | ported (`06_rotation_prediction`) |
| 7 | DeepCluster | `7_deepcluster` | **yes** | ported (`07_deepcluster`; faiss clustering, **GPU / x86_64-linux only** -- faiss-gpu has no cross-platform wheel, so it lives in the CUDA lock and the method is exempt from the CPU lock via the `# gpu-only` marker) |
| 8 | SplitBrain | `8_split_brain` | **yes** | ported (`08_split_brain`; torch-only -- the capture's flat `model.py` (no `models/` dir) is a plain two-branch AlexNet, not a submodule/eval-only method; scipy/skimage were imported but the released target is numpy argmin, measured) |
| 9 | Jigsaw Puzzle++ | `9_jigsaw_puzzle_pp` | **yes** | ported (`09_jigsaw_puzzle_pp`, VGG16 pretext + faiss-GPU knowledge transfer: cluster → pseudo-labels → AlexNet cluster-cls) |
| 10 | InstDisc | `10_inst_disc` | **yes** | ported (`10_inst_disc`) |
| 11 | CPC | `11_cpc` | **yes** | ported (`11_cpc`) |
| 12 | CMC | `12_cmc` | **yes** | ported (`12_cmc`) |
| 13 | MoCo v1 | `13_mocov1` | **yes** | ported (`13_mocov1`) |
| 14 | PIRL | `33_pirl` | no | **HOLD** (numbering differs) |
| 15 | SimCLR v1 | `14_simclrv1` | no | ported (`14_simclrv1`) |
| 16 | MoCo v2 | `15_mocov2` | no | **HOLD** |
| 17 | SimCLR v2 | `16_simclrv2` | no | **HOLD** |
| 18 | SwAV | `17_swav` | no | ported (`17_swav`) |
| 19 | SeLa | `18_sela` | no | **HOLD** |
| 20 | BYOL | `19_byol` | no | **HOLD** |
| 21 | SimSiam | `20_simsiam` | no | ported (`20_simsiam`) |
| 22 | Barlow Twins | `21_barlow_twins` | no | ported (`21_barlow_twins`) |
| 23 | MoCo v3 | `22_mocov3` | no | **HOLD** |
| 24 | DINO | `23_dino` | no | **HOLD** |
| 25 | BEiT | `24_beit` | no | **HOLD** |
| 26 | MAE | `25_mae` | no | ported (`25_mae`) |
| 27 | SimMIM | `26_simmim` | no | **HOLD** |
| 28 | iBOT | `27_ibot` | no | ported (`27_ibot`) |
| 29 | MSN | `34_msn` | no | **HOLD** |
| 30 | DINOv2 | `28_dinov2` | no | **HOLD** |
| 31 | I-JEPA | `29_ijepa` | no | **HOLD** |
| 32 | AIM | `30_aim` | no | **HOLD** |
| 33 | V-JEPA | `35_vjepa` | no | **HOLD** |
| 34 | Franca | `36_franca` | no | ported (`36_franca`) |
| 35 | DINOv3 | `31_dinov3` | no | **HOLD** |
| 36 | LeJEPA | `37_lejepa` | no | **HOLD** |
| 37 | NEPA | `32_nepa` | no | **HOLD** |

## Numbering decision (resolved 2026-08-09): keep the capture numbering

The HOLD rule below was **resolved on 2026-08-09**: the user chose to **keep the
capture's directory numbering** (e.g. SimCLR v1 ports as `14_simclrv1`, not as
list #15). Renaming the eight already-ported mismatched dirs (`01`, `02`, `17`,
`20`, `21`, `25`, `27`, `36`) — and their tests, mutations, venvs and internal
references — is deferred to publication; keeping the capture numbers now avoids
that churn and preserves provenance alignment with the (append-only) Capture
repository. So **HOLD is lifted**: the remaining `**HOLD**` cells in the table
above no longer mean "blocked" — they mean "not yet ported", and each is portable
under its capture number.

**Porting order now:** by capture-directory number among the not-yet-ported,
preferring torch-only (tier-A) self-contained methods and deferring the special-dep
/ submodule / eval-only tier (`24_beit`, `34_msn`, `35_vjepa`, `37_lejepa`). The
next candidates by capture number are `15_mocov2`, `16_simclrv2`, `18_sela`,
`19_byol`, `22_mocov3`, `23_dino`, ... (each verified tier-A by **measuring** the
capture before porting, never from the label).

**(historical) HOLD rule (user decision pending, 2026-08-06):** only port methods
whose capture directory number equals this list's number — TRUE only for numbers
3–13, which are now all ported (colorization, context_encoder, jigsaw, rotation,
deepcluster, split_brain, jigsaw++ incl. its faiss-GPU knowledge-transfer stage,
inst_disc, cpc, cmc, mocov1). `14_simclrv1` (list #15) is the first port taken
under the 2026-08-09 keep-capture-numbering decision.

## Per-port procedure (strict TDD, as used for every prior method)

1. Read the capture's `methods/<n>_<name>/` (model, trainer, eval, configs) **and**
   the closest already-ported method as a template.
2. Write the failing tests first; confirm RED.
3. Implement: model(s), thin single-process trainer (drop DDP/TensorBoard,
   resolve the device), evaluator, adapter (step1 + linear_eval), configs,
   requirements + CPU/cu130 locks, Dockerfile, provenance, README.
4. Mutation spec, all killed (proves the tests bite — required when tests are
   written after any code).
5. Base CPU suite `EXIT=0`, then the **whole suite under a torch venv** (the CI
   `locked`-equivalent — `PYTHONPATH=. .venvs/<m>/bin/python -m unittest discover
   -s tests`; the base run skips deps-gated tests and hides in-process collisions).
6. GPU verify on the available device; `verify-environment` exact.
7. Commit gated on the suite; open a PR; confirm CI green (incl. container).
