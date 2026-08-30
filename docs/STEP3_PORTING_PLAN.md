# Step 3 porting plan (on `main`, and enforced)

Last updated: 2026-08-30

This is the **sequenced, authoritative plan** for porting the "Step 3" methods
into this repository. It lives on `main`, in the working tree, so it is present
every session -- and it is enforced by `tests/test_step3_plan.py`, so a
deviation goes RED rather than unnoticed.

## Why this document exists (the drift it prevents)

The existence/coverage audit is `docs/STEP3_PORTING_AUDIT.md` on the side branch
`docs/step3-porting-audit` (it records, method by method, that the capture source
has the code and that this port did not yet). That audit carried the ordered
plan -- but it was **off `main`**, and the declared on-`main` source of truth
(`README.md`) carries only a flat methods table, no ordering. No test checked
either. So the plan was invisible where the work happens, and across context
compactions the "next method" was reconstructed from impression:

- the **A1 evaluation harness was skipped**; the Autoreg-SSL ports (EVA-02,
  AIMv2, BEiT v2) were built as single-`linear_eval` probes (the
  `docs/EVAL_DOWNLOAD.md` shape), not the six-task ARSSL harness A1 was meant to
  establish first;
- **SigLIP (Phase B1) was ported before Phase A finished** -- out of order.

Nothing went red because nothing was watching. This document, plus its test, is
the mechanism. It does not undo the four ports already merged out of turn
(EVA-02, AIMv2, BEiT v2, SigLIP): those are recorded as done. Once A1 -- the
item they jumped -- landed, `next` advanced past EVA-02/AIMv2/BEiT v2 (orders
2-4), so three of the four fell back *into* turn; only SigLIP (a Phase-B item,
order 13) remains ahead of `next`, and the guard's frozen ceiling tightened
from 4 to 1 to admit exactly it. What it stops is the **next** silent drift.

## How it is enforced

The machine-readable block at the end is the single source of truth for state.
`tests/test_step3_plan.py`:

1. **Accuracy** -- an item is `done` iff it is on disk (a `method` when its
   adapter exists; a `task` when its named artifact exists). A checkbox cannot
   lie.
2. **The `next` pointer** -- must be the earliest unfinished item in the order.
   "What is next" is a checked fact, read from the tree, not a memory.
3. **No new out-of-order work** -- the count of `done` items sitting *after*
   `next` (ported ahead of their turn) may not exceed `grandfathered_ceiling`,
   whose value is frozen in the test. A new out-of-turn port makes it RED.
4. **Partition** -- every un-numbered directory under `methods/` is either a
   Step-3 port listed here or an explicit `non_step3_unnumbered` exception (the
   off-decade Step-1&2 ports `image_gpt` / `mar` / `var`, and `_reference`). A
   ported Step-3 method cannot go unlisted.

Naming follows the 2026-08-27 decision (see the audit): **un-numbered
snake_case** directory names (`methods/eva02`, ...); the word "step 3" never
appears in a code identifier. The dir names for not-yet-ported methods below are
the planned names; when a method is actually ported, its `status` becomes `done`
and the accuracy test forces the dir to match.

## The plan, phase by phase

The order is by increasing self-containment cost: families that reuse an
existing backbone and this repo's downstream task heads come first; the
3D/4D/generative families that need a new submodule and new eval scaffolding come
later. Every unit is one PR under the strict-TDD contract (write the failing test
first): pinned submodule for author code (never copied), an adapter to the
`launch.py` chain, a smoke spec, machine verification via `contract-test` /
`matrix-run` / `matrix-audit`. **A1 builds no new task heads** -- it is a driver
over the downstream task probes this port already carries.

**What "the downstream task probes" are, measured (2026-08-29).** The capture's
ARSSL harness evaluates six task keys (`in100`, `in1k`, `coco`, `ade20k`,
`nyuv2`, `ssv2`). This port carries **four** of them as cross-method runners
under `downstream/` (`ade20k`, `coco`, `nyuv2`, `ssv2`); ImageNet-1k is the
*per-method* `linear_eval` by deliberate design (docs/DOWNSTREAM.md,
docs/EVALUATION.md -- not re-homed under `downstream/`), and ImageNet-100 is not
yet ported. So A1 is a **driver only**: it drives one frozen backbone through the
downstream probes that exist and aggregates them. The ImageNet columns are wired
in later items (A3 threads the lineage backbones' ImageNet `linear_eval`;
ImageNet-100 is a separate future port), not A1.

### Phase A -- Autoreg SSL (closest to what this port already does)

- **A1 (done).** `downstream/arssl.py`: a thin, pure-stdlib driver that runs one
  **frozen backbone** through the downstream task probes already in this repo
  (discovered, not listed) as subprocesses, checks each with the downstream
  contract, and aggregates into one `arssl_results.json` whose verdict is `ok`
  only when every selected task is. It re-uses the existing task runners (one
  implementation, invoked -- as `bin/matrix-run.py` re-uses `launch.py`), builds
  **no** new task head, and does not re-home the ImageNet columns. **This
  establishes the driver/aggregation pattern the rest of Phase A reuses.**
- **A2** -- new backbones with no lineage here: **BEiT v2, EVA-02, data2vec 2.0,
  AIMv2, CAE** (each a pinned submodule / pinned download + adapter + smoke).
  *EVA-02, AIMv2, BEiT v2 are done (as single-`linear_eval` ports, ahead of A1);
  data2vec 2.0 and CAE remain.*
- **A3** -- wire the lineage backbones (**MAE, I-JEPA, LeJEPA, iGPT, AIM, VAR**),
  already present as Step-1&2 ports, into the A1 harness so their Step-3 numbers
  are reproducible here. *MAE is done:* the method declares a `mae_vit`
  spatial-backbone provider in its **own** directory
  (`methods/25_mae/downstream_backbone.py`, a module-level `KIND` + `build`), and
  the shared layer (`downstream/spatial_backbones.py`) **discovers** it by
  structure -- the shared machinery names no method
  (`tests/test_no_hard_coded_methods.py`). The provider **reuses** MAE's own model
  (MAE's encoder.pt is not timm-loadable and its sincos position embedding is a
  regenerated, unstored buffer, so a timm remap would be a second, drift-prone
  implementation). The MAE module is loaded by file path under a unique name (no
  cross-method `models` collision) and the patch grid is read with a forward hook
  on the encoder norm (MAEEncoder.forward reused verbatim). This establishes the
  discovered lineage-provider pattern the remaining A3 items reuse: I-JEPA,
  LeJEPA, iGPT, AIM and VAR each add their own `downstream_backbone.py`. Those
  remain.

### Phase B -- Multimodal / CompEval (reuse frozen backbones)

- **B1** -- **SigLIP, SAM3, DINOv3-7B, Cosmos3 Super**: frozen-backbone adapters
  (the `38_clip` / `docs/EVAL_DOWNLOAD.md` pattern). *SigLIP is done (ahead of
  Phase A).*
- **B2** -- the CompEval_Extend60 adapter set over backbones ported in other
  phases: **RAE1, RAE2, RAEv2-K7, VDPM, VGGT-Omega, Cosmos 3, V-JEPA 2.1**
  (adapters, not new backbones).

### Phase C -- Video SSL & Gen (video data path; the SSv2 head exists)

- **C1** -- **Shuffle & Learn, Video MoCo, Video MAE** (classical video SSL).
- **C2** -- **V-JEPA 2, V-JEPA 2-AC, V-JEPA 2.1** (extend the `35_vjepa`
  submodule).
- **C3** -- **Cosmos3, WAN2.2** (video-generative backbones as frozen probes).

### Phase D -- GenSSL (generative representation learning)

- **D1** -- **MAGE, DiT, JiT, RAE, RAEv2** (new submodules + adapters;
  linear_eval over the frozen generative encoder, VAE-style probe pattern).

### Phase E -- 3DFM (3D foundation models, all wholly new)

- **E1** -- **CroCo, DUSt3R, MASt3R** (the DUSt3R lineage).
- **E2** -- **DA3, Pi^3, VGGT, VGGT-Omega** (VGGT lineage + variants).

### Phase F -- 4DFM (4D / dynamic scene, all wholly new)

- **F1** -- **MonST3R, D2ST3R, StreamVGGT** (dynamic 3D reconstruction).
- **F2** -- **Deja View (DVLT), V-DPM** (multi-view / video diffusion priors).

## Machine-readable checklist (the source of truth for state)

```json
{
  "next": "A3:ijepa",
  "grandfathered_ceiling": 1,
  "non_step3_unnumbered": ["_reference", "image_gpt", "mar", "var"],
  "items": [
    {"id": "A1", "phase": "A", "subphase": "A1", "order": 1, "kind": "task", "title": "ARSSL eval harness (driver over the downstream task probes)", "artifact": "downstream/arssl.py", "status": "done"},
    {"id": "A2:eva02", "phase": "A", "subphase": "A2", "order": 2, "kind": "method", "dir": "eva02", "title": "EVA-02", "status": "done"},
    {"id": "A2:aimv2", "phase": "A", "subphase": "A2", "order": 3, "kind": "method", "dir": "aimv2", "title": "AIMv2", "status": "done"},
    {"id": "A2:beitv2", "phase": "A", "subphase": "A2", "order": 4, "kind": "method", "dir": "beitv2", "title": "BEiT v2", "status": "done"},
    {"id": "A2:data2vec2", "phase": "A", "subphase": "A2", "order": 5, "kind": "method", "dir": "data2vec2", "title": "data2vec 2.0", "status": "done"},
    {"id": "A2:cae", "phase": "A", "subphase": "A2", "order": 6, "kind": "method", "dir": "cae", "title": "CAE", "status": "done"},
    {"id": "A3:mae", "phase": "A", "subphase": "A3", "order": 7, "kind": "task", "title": "wire MAE into the A1 harness", "artifact": "methods/25_mae/downstream_backbone.py", "status": "done"},
    {"id": "A3:ijepa", "phase": "A", "subphase": "A3", "order": 8, "kind": "task", "title": "wire I-JEPA into the A1 harness", "artifact": null, "status": "todo"},
    {"id": "A3:lejepa", "phase": "A", "subphase": "A3", "order": 9, "kind": "task", "title": "wire LeJEPA into the A1 harness", "artifact": null, "status": "todo"},
    {"id": "A3:igpt", "phase": "A", "subphase": "A3", "order": 10, "kind": "task", "title": "wire iGPT into the A1 harness", "artifact": null, "status": "todo"},
    {"id": "A3:aim", "phase": "A", "subphase": "A3", "order": 11, "kind": "task", "title": "wire AIM into the A1 harness", "artifact": null, "status": "todo"},
    {"id": "A3:var", "phase": "A", "subphase": "A3", "order": 12, "kind": "task", "title": "wire VAR into the A1 harness", "artifact": null, "status": "todo"},
    {"id": "B1:siglip", "phase": "B", "subphase": "B1", "order": 13, "kind": "method", "dir": "siglip", "title": "SigLIP", "status": "done"},
    {"id": "B1:sam3", "phase": "B", "subphase": "B1", "order": 14, "kind": "method", "dir": "sam3", "title": "SAM3", "status": "todo"},
    {"id": "B1:dinov3_7b", "phase": "B", "subphase": "B1", "order": 15, "kind": "method", "dir": "dinov3_7b", "title": "DINOv3-7B", "status": "todo"},
    {"id": "B1:cosmos3_super", "phase": "B", "subphase": "B1", "order": 16, "kind": "method", "dir": "cosmos3_super", "title": "Cosmos3 Super", "status": "todo"},
    {"id": "B2:rae1", "phase": "B", "subphase": "B2", "order": 17, "kind": "task", "title": "CompEval adapter: RAE1", "artifact": null, "status": "todo"},
    {"id": "B2:rae2", "phase": "B", "subphase": "B2", "order": 18, "kind": "task", "title": "CompEval adapter: RAE2", "artifact": null, "status": "todo"},
    {"id": "B2:raev2_k7", "phase": "B", "subphase": "B2", "order": 19, "kind": "task", "title": "CompEval adapter: RAEv2-K7", "artifact": null, "status": "todo"},
    {"id": "B2:vdpm_eval", "phase": "B", "subphase": "B2", "order": 20, "kind": "task", "title": "CompEval adapter: VDPM", "artifact": null, "status": "todo"},
    {"id": "B2:vggt_omega_eval", "phase": "B", "subphase": "B2", "order": 21, "kind": "task", "title": "CompEval adapter: VGGT-Omega", "artifact": null, "status": "todo"},
    {"id": "B2:cosmos3_eval", "phase": "B", "subphase": "B2", "order": 22, "kind": "task", "title": "CompEval adapter: Cosmos 3", "artifact": null, "status": "todo"},
    {"id": "B2:vjepa2_1_eval", "phase": "B", "subphase": "B2", "order": 23, "kind": "task", "title": "CompEval adapter: V-JEPA 2.1", "artifact": null, "status": "todo"},
    {"id": "C1:shufflelearn", "phase": "C", "subphase": "C1", "order": 24, "kind": "method", "dir": "shufflelearn", "title": "Shuffle & Learn", "status": "todo"},
    {"id": "C1:video_moco", "phase": "C", "subphase": "C1", "order": 25, "kind": "method", "dir": "video_moco", "title": "Video MoCo", "status": "todo"},
    {"id": "C1:videomae", "phase": "C", "subphase": "C1", "order": 26, "kind": "method", "dir": "videomae", "title": "Video MAE", "status": "todo"},
    {"id": "C2:vjepa2", "phase": "C", "subphase": "C2", "order": 27, "kind": "method", "dir": "vjepa2", "title": "V-JEPA 2", "status": "todo"},
    {"id": "C2:vjepa2_ac", "phase": "C", "subphase": "C2", "order": 28, "kind": "method", "dir": "vjepa2_ac", "title": "V-JEPA 2-AC", "status": "todo"},
    {"id": "C2:vjepa2_1", "phase": "C", "subphase": "C2", "order": 29, "kind": "method", "dir": "vjepa2_1", "title": "V-JEPA 2.1", "status": "todo"},
    {"id": "C3:cosmos3", "phase": "C", "subphase": "C3", "order": 30, "kind": "method", "dir": "cosmos3", "title": "Cosmos3", "status": "todo"},
    {"id": "C3:wan22", "phase": "C", "subphase": "C3", "order": 31, "kind": "method", "dir": "wan22", "title": "WAN2.2", "status": "todo"},
    {"id": "D1:mage", "phase": "D", "subphase": "D1", "order": 32, "kind": "method", "dir": "mage", "title": "MAGE", "status": "todo"},
    {"id": "D1:dit", "phase": "D", "subphase": "D1", "order": 33, "kind": "method", "dir": "dit", "title": "DiT", "status": "todo"},
    {"id": "D1:jit", "phase": "D", "subphase": "D1", "order": 34, "kind": "method", "dir": "jit", "title": "JiT", "status": "todo"},
    {"id": "D1:rae", "phase": "D", "subphase": "D1", "order": 35, "kind": "method", "dir": "rae", "title": "RAE", "status": "todo"},
    {"id": "D1:raev2", "phase": "D", "subphase": "D1", "order": 36, "kind": "method", "dir": "raev2", "title": "RAEv2", "status": "todo"},
    {"id": "E1:croco", "phase": "E", "subphase": "E1", "order": 37, "kind": "method", "dir": "croco", "title": "CroCo", "status": "todo"},
    {"id": "E1:dust3r", "phase": "E", "subphase": "E1", "order": 38, "kind": "method", "dir": "dust3r", "title": "DUSt3R", "status": "todo"},
    {"id": "E1:mast3r", "phase": "E", "subphase": "E1", "order": 39, "kind": "method", "dir": "mast3r", "title": "MASt3R", "status": "todo"},
    {"id": "E2:da3", "phase": "E", "subphase": "E2", "order": 40, "kind": "method", "dir": "da3", "title": "DA3", "status": "todo"},
    {"id": "E2:pi3", "phase": "E", "subphase": "E2", "order": 41, "kind": "method", "dir": "pi3", "title": "Pi^3", "status": "todo"},
    {"id": "E2:vggt", "phase": "E", "subphase": "E2", "order": 42, "kind": "method", "dir": "vggt", "title": "VGGT", "status": "todo"},
    {"id": "E2:vggt_omega", "phase": "E", "subphase": "E2", "order": 43, "kind": "method", "dir": "vggt_omega", "title": "VGGT-Omega", "status": "todo"},
    {"id": "F1:monst3r", "phase": "F", "subphase": "F1", "order": 44, "kind": "method", "dir": "monst3r", "title": "MonST3R", "status": "todo"},
    {"id": "F1:d2st3r", "phase": "F", "subphase": "F1", "order": 45, "kind": "method", "dir": "d2st3r", "title": "D2ST3R", "status": "todo"},
    {"id": "F1:streamvggt", "phase": "F", "subphase": "F1", "order": 46, "kind": "method", "dir": "streamvggt", "title": "StreamVGGT", "status": "todo"},
    {"id": "F2:dvlt", "phase": "F", "subphase": "F2", "order": 47, "kind": "method", "dir": "dvlt", "title": "Deja View (DVLT)", "status": "todo"},
    {"id": "F2:vdpm", "phase": "F", "subphase": "F2", "order": 48, "kind": "method", "dir": "vdpm", "title": "V-DPM", "status": "todo"}
  ]
}
```
