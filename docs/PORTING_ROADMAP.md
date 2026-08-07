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
(`8_split_brain`, `34_msn`, `35_vjepa`, `36_franca`(done, eval-only), `37_lejepa`)
and need a submodule / eval-only treatment (and, for msn/vjepa/lejepa, a
noncommercial-licence decision).

**Verify per method at port time** (do not assume): the model header's licence
(lab-own vs a copy of author code → the latter needs a submodule reference), the
exact dependencies, the backbone, and step-1 reproducibility.

## Dependency tiers (measured from the capture's requirements.txt)

- **A — torch/torchvision/numpy/PyYAML** (reuse `image_gpt`/`25_mae` locks): `23_dino`, `33_pirl`.
- **B — + timm** (reuse a mar-style lock): mocov1/2/3, simclrv1/2, byol, sela, inst_disc, cpc, cmc, rotation, simmim, dinov2, dinov3, ijepa, nepa.
- **B' — + huggingface_hub** (var-style): `30_aim`.
- **C — heavy special deps**: `3_colorization` (opencv/scikit-image), `24_beit` (DALL-E tokeniser), `7_deepcluster` (faiss? — verify), `9_jigsaw_puzzle_pp`'s **knowledge-transfer stages** (faiss-GPU k-means, mandatory — the capture's `cluster_and_pseudolabels.py` explicitly refuses the CPU/sklearn fallback; **measured, contradicts the earlier "tier B / +timm" guess**). The VGG16 **pretext** stage alone is tier-A (torch/torchvision/numpy/Pillow/PyYAML) and is what `09_jigsaw_puzzle_pp` ports.
- **D — no `models/` (submodule / eval-only, often noncommercial)**: `8_split_brain`, `34_msn`, `35_vjepa`, `37_lejepa`.

TensorBoard / tqdm / wandb logging is dropped (the port owns a thin single-process
loop), as in every prior port.

## Fact-based order (reuse templates and locks; keep CI free)

Port self-contained + light-dep + shared-template methods first; heavy/ViT next;
special-dep / submodule / eval-only / noncommercial last.

- **Group 1 — ResNet/CNN pretext & contrastive** (reuse the simsiam/swav/barlow ResNet + linear-probe template): jigsaw, rotation, jigsaw++ (VGG16 pretext only), inst_disc, pirl, deepcluster, cpc, cmc, mocov1/2/3, simclrv1/2, sela, byol.
- **Group 2 — ViT** (reuse the ibot/mae ViT template): simmim, dino, dinov2, dinov3, ijepa, nepa, aim.
- **Group 3 — special deps / submodule / eval-only / licence decision**: colorization, beit, deepcluster(if faiss), `9_jigsaw_puzzle_pp`'s faiss-GPU knowledge-transfer stages (cluster → pseudo-labels → AlexNet cluster-cls, alongside deepcluster), split_brain, msn, vjepa, lejepa.

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
| 3 | Colorization | `3_colorization` | **yes** | not ported (Group 3: opencv/skimage) |
| 4 | Context Encoder | `4_context_encoder` | **yes** | ported (`04_context_encoder`) |
| 5 | Jigsaw Puzzles | `5_jigsaw_puzzle` | **yes** | ported (`05_jigsaw_puzzle`) |
| 6 | Rotation Prediction | `6_rotation_prediction` | **yes** | ported (`06_rotation_prediction`) |
| 7 | DeepCluster | `7_deepcluster` | **yes** | portable now (Group 1; verify faiss) |
| 8 | SplitBrain | `8_split_brain` | **yes** | not ported (Group 3: no `models/`) |
| 9 | Jigsaw Puzzle++ | `9_jigsaw_puzzle_pp` | **yes** | ported (`09_jigsaw_puzzle_pp`, VGG16 pretext only; faiss knowledge-transfer deferred to Group 3) |
| 10 | InstDisc | `10_inst_disc` | **yes** | ported (`10_inst_disc`) |
| 11 | CPC | `11_cpc` | **yes** | ported (`11_cpc`) |
| 12 | CMC | `12_cmc` | **yes** | ported (`12_cmc`) |
| 13 | MoCo v1 | `13_mocov1` | **yes** | ported (`13_mocov1`) |
| 14 | PIRL | `33_pirl` | no | **HOLD** (numbering differs) |
| 15 | SimCLR v1 | `14_simclrv1` | no | **HOLD** |
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

## HOLD rule (user decision pending, 2026-08-06)

**Only port methods whose capture directory number equals this list's number.**
Everything numbered ≠ (numbers 1, 2, and 14–37) is **held pending the user's
numbering decision** (keep the capture numbering, or renumber to this list). The
already-ported methods that are mismatched (`01`, `02`, `17`, `20`, `21`, `25`,
`27`, `36`) stay as-is for now; whether to renumber them at publication is part of
that same decision.

**Portable now (numbering matches + not yet ported):** the Group-1 tier
(`5_jigsaw_puzzle`, `6_rotation_prediction`, `9_jigsaw_puzzle_pp`, `10_inst_disc`,
`11_cpc`, `12_cmc`, `13_mocov1`) is now fully ported. What remains needs
dependency / submodule work: `3_colorization`, `7_deepcluster`, `8_split_brain`
(Group 3).
(`5_jigsaw_puzzle`, `6_rotation_prediction`, `10_inst_disc`, `11_cpc`, `12_cmc`,
`13_mocov1` ported.
`9_jigsaw_puzzle_pp` ported as the VGG16 pretext only; its faiss-GPU
knowledge-transfer stages are deferred to Group 3, see below.)

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
