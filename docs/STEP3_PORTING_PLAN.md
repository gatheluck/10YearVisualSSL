# Step 3 porting plan (on `main`, and enforced)

Last updated: 2026-09-02

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
2-4), so three of the four fell back *into* turn; SigLIP (a Phase-B item,
order 13) remained the one out-of-turn port, and the frozen ceiling tightened
from 4 to 1 to admit exactly it. Now that all of Phase A has landed (A3:var,
order 12, was the last A3 item) and all three ungated B1 backbones have landed in
turn (SigLIP order 13, SAM3 order 14, Cosmos3 Super order 16), no out-of-order
port stands and the ceiling stays 0. **DINOv3-7B (order 15) is `deferred`, not
skipped:** its weights are Hugging Face gated and, unlike SAM3, the capture recorded
no full sha256, so a real `backbone_artifact` hash cannot be pinned honestly from
this machine (see the item's `deferred_reason`). A deferral is a recorded departure
from the queue -- the test requires a non-empty reason -- not a silent one; it moves
the item off the critical path without marking it done, so `next` stepped over it to
Cosmos3 Super (order 16), which has now landed. With every ungated B1 backbone done
and no B2 CompEval adapter yet buildable (see the correction below), `next` is
**C1:shufflelearn (order 17)**, the earliest `todo`. When the DINOv3-7B weights are
obtained through authorized access, its status returns to `todo`.

**A second latent drift, found and fixed 2026-09-02:** the B2 CompEval adapters
are frozen-backbone probes "over backbones ported in other phases", yet the plan
had listed all of them at orders 17-23 -- *ahead* of the C/D/E/F phases that
produce those backbones. An adapter cannot run before its input exists. The set
was re-ordered so each eval adapter now sits immediately after the backbone it
probes, and each adapter carries a `depends_on` naming that backbone. This is
enforced, not merely written: `TestAnAdapterFollowsItsBackbone` fails if any
`depends_on` points at a later-ordered item.

**A correction to that fix, same day (2026-09-02):** the re-order first placed
`B2:cosmos3_eval` at order 17 with `depends_on: B1:cosmos3_super`, on the belief
that the CompEval adapter titled "Cosmos 3" probed the already-ported Cosmos3
Super. Measurement of the capture record showed that belief was wrong. The
capture holds *two* distinct Cosmos CompEval adapters -- `cosmos3_adapter.py`
("Cosmos 3", loading `checkpoints/cosmos3/Cosmos3-Nano`, a 16B `nvidia/Cosmos3-Nano`
checkpoint, feature dim ~4096) and `cosmos3_super_adapter.py` ("Cosmos 3 Super",
the 64B `nvidia/Cosmos3-Super`, feature dim 1152, which is what `methods/cosmos3_super`
already ported). By its title and by that split, `B2:cosmos3_eval` is the **Nano**
adapter, whose backbone is **C3:cosmos3** (order 24, `todo`), *not* the ported
Super. Its `depends_on` is corrected to `C3:cosmos3` and it is re-ordered to 25
(immediately after C3:cosmos3). The order guard passed throughout because
Cosmos3 Super (order 16) also precedes order 17 -- the guard checks *ordering*, not
*which model* an adapter probes; that model-identity fact lives in the capture repo,
which is not present here or in CI, so it cannot be mechanised from this repo and is
recorded in prose instead. The consequence: **no B2 CompEval adapter's backbone is
on disk yet** (each depends on a C/D/E/F backbone not yet ported), so `next` is not
a B2 item at all but **C1:shufflelearn** (order 17), the earliest `todo`; the B2
adapters follow their backbones in C2/C3/D1/E2/F2. What this document stops is the
**next** silent drift.

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
  discovered lineage-provider pattern the remaining A3 items reuse. *I-JEPA is
  done:* `methods/29_ijepa/downstream_backbone.py` adds an `ijepa_vit` provider
  the shared layer discovers with no edit -- simpler than MAE because I-JEPA's own
  `VisionTransformer.forward` returns raster-order patch tokens with no
  masking/shuffling and no CLS token, its pos-embed ships in `encoder.pt`, and the
  bare-key checkpoint loads with an exact match (unexpected/missing both refused).
  *LeJEPA is done, and it is the first A3 item wired by **config, not a provider**:
  measurement (not the file name) shows LeJEPA trains a standard timm ViT-B/16 and
  its `encoder.pt` is the bare backbone (the `backbone.` prefix stripped at save),
  so it loads straight into the shared `vit` spatial-backbone kind whose default
  arch is that very `vit_base_patch16_224`. A dedicated provider would be an empty
  duplicate of the `vit` kind (never implement the same rule twice; a guard with no
  killed mutant is not a guard), so the wiring is
  `methods/37_lejepa/configs/downstream_arssl.json` -- discovered by structure
  (`methods/*/configs/downstream_arssl.json`, naming no method) and run through the
  dense-task probes by the JSON-native ARSSL driver. *iGPT is done, and it is the
  first A3 provider whose input is **not an image tensor**:*
  `methods/image_gpt/downstream_backbone.py` adds an `igpt` provider (hand-written,
  non-timm, so a provider, not a config). It does what the method's own probe does
  before reading features -- resize to the token grid, quantise pixels to colour
  tokens, read a **middle** transformer layer -- and reshapes the per-position
  tokens to a `[B, C, h, w]` map (`IGPT.extract_token_features`, which the probe's
  `extract_features` mean-pools -- one representation, two readers). The chosen
  shape has the provider **absorb** what the shared ViT schema has no slot for, so
  the four task runners stay unchanged: the colour vocabulary is inferred from
  `encoder.pt` (its token-embedding rows) and the colour clusters are read from
  `clusters.npy` beside it (a missing/mismatched set is refused). *AIM is done,
  the third provider* (`methods/30_aim/downstream_backbone.py`, `aim_vit`):
  measurement shows AIM is a hand-written, non-timm ViT (a `nn.Linear` patch
  embedding, bias-free, no CLS token, a sincos position buffer), so its
  `encoder.pt` cannot load into the shared `vit` kind -- it needs a provider, not a
  config. Its `forward` returns a training tuple under a prefix-LM mask, so features
  come only from `AIMViT.forward_features(x, layer_ids)` -- the last
  `num_feature_layers` blocks averaged, run bidirectionally -- which the method's
  own probe patch-mean-pools; the provider reproduces that read and reshapes the
  tokens to `[B, C, h, w]` (pooling the map equals AIM's probe feature). Following
  iGPT, the provider **absorbs** what the schema has no slot for so the four runners
  stay unchanged: `num_feature_layers` is fixed to AIM's protocol value 6 (clamped
  to depth), and the prediction head (`predictor.*`, excluded from `encoder.pt`) is
  built minimally and its absence tolerated, while an alien key or a missing trunk
  weight is refused. *VAR is done, the fourth provider*
  (`methods/var/downstream_backbone.py`, `var_vqvae`) -- and it completes Phase A.
  Measurement (not the name) shows VAR's probed representation is the **VQVAE
  tokeniser's encoder** output, global-average-pooled (`evaluate_linear_var.encode`)
  -- not the VAR transformer step 1 trains, and not `encoder.pt`. `vae.encoder(x)`
  already returns a `[B, Cvae, H/16, W/16]` map (the VQGAN encoder is fully
  convolutional, stride 16), so unlike the ViT providers there is no token grid to
  reshape: the provider returns that map directly, and pooling it reproduces VAR's
  own probe feature (one representation, two readers). So the backbone spec's
  `encoder` is the **VQVAE tokeniser checkpoint** (the pinned download
  `vae_ch160v4096z32.pth`); following iGPT, the VQVAE architecture the shared ViT
  schema has no slot for (`Cvae`, the vocabulary `V`, the base width `ch`) is
  **inferred from the checkpoint**, so a config cannot disagree with the trained
  tokeniser. The tokeniser is built through the method's own
  `train_pretrain_var.build_vqvae` (one place knows how it is built, loaded by file
  path under a unique name -- no cross-method `models` collision), and a checkpoint
  that is not this VQVAE -- an inference weight, a tokeniser weight, or an alien key
  -- is refused rather than half-loaded.

### Phase B -- Multimodal / CompEval (reuse frozen backbones)

- **B1** -- **SigLIP, SAM3, DINOv3-7B, Cosmos3 Super**: frozen-backbone adapters
  (the `38_clip` / `docs/EVAL_DOWNLOAD.md` pattern). *SigLIP is done (it was
  ported ahead of Phase A; now that Phase A has landed it is back in turn, the
  first Phase-B item). SAM3 is done: a pure eval-only `linear_eval` probe on the
  frozen SAM 3 vision encoder (mean-pooled patch tokens), the transformers-sourced
  sibling of data2vec2, with a trunk converter (`methods/sam3/sam3_trunk.py`) that
  maps the official ViTDet-style `sam3.pt` onto `transformers`' `Sam3ViTModel`
  (unit-tested on synthetic tensors, so the gated-weight path is covered
  hermetically). DINOv3-7B is `deferred`: its ViT-7B/16 weights
  (`facebook/dinov3-vit7b16-pretrain-lvd1689m`) are HF-gated and the capture
  recorded no full sha256, so -- unlike SAM3 -- a real `backbone_artifact` hash
  cannot be pinned from here; it returns to `todo` once the weights are fetched
  through authorized access. Cosmos3 Super is done: a pure eval-only `linear_eval`
  probe on the frozen Qwen3-VL vision encoder (`nvidia/Cosmos3-Super`, OpenMDW-1.1,
  public), loaded directly by `Qwen3VLVisionModel.from_pretrained` on the
  `vision_encoder/` directory -- no trunk converter, because the released layout
  matches the HF class and a `save_pretrained` -> `from_pretrained` round-trip is
  exact, so the loading path is unit-tested without the ~1.1GB weights; its
  `vision_encoder/model.safetensors` is pinned by a sha256 that the capture and the
  HF LFS metadata agree on. With every ungated B1 backbone done and no B2 CompEval
  adapter's backbone yet on disk (the CompEval "Cosmos 3" adapter probes Cosmos3
  Nano = `C3:cosmos3`, not this Super backbone -- see the correction above), `next`
  is the first Phase-C method, `C1:shufflelearn`.*
- **B2** -- the CompEval_Extend60 adapter set over backbones ported in other
  phases: **Cosmos 3, V-JEPA 2.1, RAE1, RAE2, RAEv2-K7, VGGT-Omega, VDPM**
  (adapters, not new backbones). These are frozen-backbone probes, so each one is
  scheduled **after** the backbone it evaluates rather than as a contiguous
  early block: `B2:cosmos3_eval` follows `C3:cosmos3` (the Cosmos3-Nano backbone
  it probes -- *not* the ported `B1:cosmos3_super`, see the correction above);
  `B2:vjepa2_1_eval` follows `C2:vjepa2_1`; `B2:rae1`/`B2:rae2` follow `D1:rae`;
  `B2:raev2_k7` follows `D1:raev2`; `B2:vggt_omega_eval` follows `E2:vggt_omega`;
  `B2:vdpm_eval` follows `F2:vdpm`. Each adapter item carries a `depends_on`
  naming its backbone, and `TestAnAdapterFollowsItsBackbone` enforces that the
  backbone is ordered first. (RAE1/RAE2/RAEv2-K7 additionally have no recorded
  weight provenance today -- the same class of blocker as DINOv3-7B; that is a
  separate concern from ordering and is handled when their backbones are ported.)

### Phase C -- Video SSL & Gen (video data path; the SSv2 head exists)

- **C1** -- **Shuffle & Learn, Video MoCo, Video MAE** (classical video SSL).
  *Shuffle & Learn and Video MoCo are `deferred` (measured 2026-09-02):* the
  capture holds them as first-party PyTorch re-implementations
  (`methods_step3/VideoSSL/01_shufflelearn`, `02_videomoco`) with **no released
  checkpoint** and, critically, with the pretext-training code their own VideoSSL
  README documents (`pretrain/{dataset,model,train}.py`) **absent from the capture
  snapshot** -- only the backbone wrapper, the eval stages, and the qsub launchers
  are captured. With neither weights nor the capture's own training code they cannot
  be faithfully ported; each carries a `deferred_reason` and `next` steps over them
  to **Video MAE** (order 19), which uses official public HuggingFace weights
  (`MCG-NJU/videomae-base`, CC-BY-NC-4.0) and is portable as an eval-only
  frozen-backbone probe.
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
  "next": "C2:vjepa2",
  "grandfathered_ceiling": 0,
  "non_step3_unnumbered": ["_reference", "image_gpt", "mar", "var"],
  "items": [
    {"id": "A1", "phase": "A", "subphase": "A1", "order": 1, "kind": "task", "title": "ARSSL eval harness (driver over the downstream task probes)", "artifact": "downstream/arssl.py", "status": "done"},
    {"id": "A2:eva02", "phase": "A", "subphase": "A2", "order": 2, "kind": "method", "dir": "eva02", "title": "EVA-02", "status": "done"},
    {"id": "A2:aimv2", "phase": "A", "subphase": "A2", "order": 3, "kind": "method", "dir": "aimv2", "title": "AIMv2", "status": "done"},
    {"id": "A2:beitv2", "phase": "A", "subphase": "A2", "order": 4, "kind": "method", "dir": "beitv2", "title": "BEiT v2", "status": "done"},
    {"id": "A2:data2vec2", "phase": "A", "subphase": "A2", "order": 5, "kind": "method", "dir": "data2vec2", "title": "data2vec 2.0", "status": "done"},
    {"id": "A2:cae", "phase": "A", "subphase": "A2", "order": 6, "kind": "method", "dir": "cae", "title": "CAE", "status": "done"},
    {"id": "A3:mae", "phase": "A", "subphase": "A3", "order": 7, "kind": "task", "title": "wire MAE into the A1 harness", "artifact": "methods/25_mae/downstream_backbone.py", "status": "done"},
    {"id": "A3:ijepa", "phase": "A", "subphase": "A3", "order": 8, "kind": "task", "title": "wire I-JEPA into the A1 harness", "artifact": "methods/29_ijepa/downstream_backbone.py", "status": "done"},
    {"id": "A3:lejepa", "phase": "A", "subphase": "A3", "order": 9, "kind": "task", "title": "wire LeJEPA into the A1 harness", "artifact": "methods/37_lejepa/configs/downstream_arssl.json", "status": "done"},
    {"id": "A3:igpt", "phase": "A", "subphase": "A3", "order": 10, "kind": "task", "title": "wire iGPT into the A1 harness", "artifact": "methods/image_gpt/downstream_backbone.py", "status": "done"},
    {"id": "A3:aim", "phase": "A", "subphase": "A3", "order": 11, "kind": "task", "title": "wire AIM into the A1 harness", "artifact": "methods/30_aim/downstream_backbone.py", "status": "done"},
    {"id": "A3:var", "phase": "A", "subphase": "A3", "order": 12, "kind": "task", "title": "wire VAR into the A1 harness", "artifact": "methods/var/downstream_backbone.py", "status": "done"},
    {"id": "B1:siglip", "phase": "B", "subphase": "B1", "order": 13, "kind": "method", "dir": "siglip", "title": "SigLIP", "status": "done"},
    {"id": "B1:sam3", "phase": "B", "subphase": "B1", "order": 14, "kind": "method", "dir": "sam3", "title": "SAM3", "status": "done"},
    {"id": "B1:dinov3_7b", "phase": "B", "subphase": "B1", "order": 15, "kind": "method", "dir": "dinov3_7b", "title": "DINOv3-7B", "status": "deferred", "deferred_reason": "The DINOv3 ViT-7B/16 weights (facebook/dinov3-vit7b16-pretrain-lvd1689m) are Hugging Face gated (Meta DINOv3 License) and the capture's SOURCE_SNAPSHOT.json records no full sha256 (only the .pth 8-char suffix a955f4ea and weight_bytes); with no HF token or local snapshot on this machine a real backbone_artifact sha256 cannot be obtained honestly. Deferred (2026-09-02) until the weights are fetched via authorized Hugging Face access, so the backbone can be pinned by a real, verified sha256 like every other eval-only port."},
    {"id": "B1:cosmos3_super", "phase": "B", "subphase": "B1", "order": 16, "kind": "method", "dir": "cosmos3_super", "title": "Cosmos3 Super", "status": "done"},
    {"id": "C1:shufflelearn", "phase": "C", "subphase": "C1", "order": 17, "kind": "method", "dir": "shufflelearn", "title": "Shuffle & Learn", "status": "deferred", "deferred_reason": "Shuffle & Learn is a first-party PyTorch re-implementation in the capture (methods_step3/VideoSSL/01_shufflelearn), not an author-code port, and has no released checkpoint: the VideoSSL README's checkpoint column reads 'Requires video data (UCF-101 / Kinetics-400)'. The pretext-training code the same README documents (pretrain/{dataset,model,train}.py, the temporal-order-verification task) is ABSENT from the capture snapshot -- only the ResNet backbone wrapper, the eval stages, and the qsub launchers are captured, and scripts/qsub_pretrain_multinode.sh invokes a pretrain/train.py that does not exist under origin/snapshots. With neither released weights nor the capture's own pretext-training code, the method cannot be faithfully ported. Deferred (2026-09-02) until the capture snapshot includes the pretrain/ code (or a deliberate decision is taken to re-implement the pretext from arXiv:1603.08561, which would be a third implementation and is out of scope for a faithful port)."},
    {"id": "C1:video_moco", "phase": "C", "subphase": "C1", "order": 18, "kind": "method", "dir": "video_moco", "title": "Video MoCo", "status": "deferred", "deferred_reason": "Video MoCo is a first-party PyTorch re-implementation in the capture (methods_step3/VideoSSL/02_videomoco), not an author-code port, with no released checkpoint (VideoSSL README checkpoint column: 'Requires video data (Kinetics-400)'). As with C1:shufflelearn, the pretext-training code the README documents (pretrain/{dataset,model,train}.py, MoCo-v2 video contrastive training) is ABSENT from the capture snapshot -- only the backbone wrapper, eval stages, and qsub launchers are captured. Deferred (2026-09-02) until the capture snapshot includes the pretrain/ code (or a decision is taken to re-implement from arXiv:2103.05346). C1:videomae (order 19) uses official public HuggingFace weights (MCG-NJU/videomae-base, CC-BY-NC-4.0) and is portable, so next advances to it."},
    {"id": "C1:videomae", "phase": "C", "subphase": "C1", "order": 19, "kind": "method", "dir": "videomae", "title": "Video MAE", "status": "done"},
    {"id": "C2:vjepa2", "phase": "C", "subphase": "C2", "order": 20, "kind": "method", "dir": "vjepa2", "title": "V-JEPA 2", "status": "todo"},
    {"id": "C2:vjepa2_ac", "phase": "C", "subphase": "C2", "order": 21, "kind": "method", "dir": "vjepa2_ac", "title": "V-JEPA 2-AC", "status": "todo"},
    {"id": "C2:vjepa2_1", "phase": "C", "subphase": "C2", "order": 22, "kind": "method", "dir": "vjepa2_1", "title": "V-JEPA 2.1", "status": "todo"},
    {"id": "B2:vjepa2_1_eval", "phase": "B", "subphase": "B2", "order": 23, "kind": "task", "title": "CompEval adapter: V-JEPA 2.1", "artifact": null, "depends_on": "C2:vjepa2_1", "status": "todo"},
    {"id": "C3:cosmos3", "phase": "C", "subphase": "C3", "order": 24, "kind": "method", "dir": "cosmos3", "title": "Cosmos3", "status": "todo"},
    {"id": "B2:cosmos3_eval", "phase": "B", "subphase": "B2", "order": 25, "kind": "task", "title": "CompEval adapter: Cosmos 3", "artifact": null, "depends_on": "C3:cosmos3", "status": "todo"},
    {"id": "C3:wan22", "phase": "C", "subphase": "C3", "order": 26, "kind": "method", "dir": "wan22", "title": "WAN2.2", "status": "todo"},
    {"id": "D1:mage", "phase": "D", "subphase": "D1", "order": 27, "kind": "method", "dir": "mage", "title": "MAGE", "status": "todo"},
    {"id": "D1:dit", "phase": "D", "subphase": "D1", "order": 28, "kind": "method", "dir": "dit", "title": "DiT", "status": "todo"},
    {"id": "D1:jit", "phase": "D", "subphase": "D1", "order": 29, "kind": "method", "dir": "jit", "title": "JiT", "status": "todo"},
    {"id": "D1:rae", "phase": "D", "subphase": "D1", "order": 30, "kind": "method", "dir": "rae", "title": "RAE", "status": "todo"},
    {"id": "B2:rae1", "phase": "B", "subphase": "B2", "order": 31, "kind": "task", "title": "CompEval adapter: RAE1", "artifact": null, "depends_on": "D1:rae", "status": "todo"},
    {"id": "B2:rae2", "phase": "B", "subphase": "B2", "order": 32, "kind": "task", "title": "CompEval adapter: RAE2", "artifact": null, "depends_on": "D1:rae", "status": "todo"},
    {"id": "D1:raev2", "phase": "D", "subphase": "D1", "order": 33, "kind": "method", "dir": "raev2", "title": "RAEv2", "status": "todo"},
    {"id": "B2:raev2_k7", "phase": "B", "subphase": "B2", "order": 34, "kind": "task", "title": "CompEval adapter: RAEv2-K7", "artifact": null, "depends_on": "D1:raev2", "status": "todo"},
    {"id": "E1:croco", "phase": "E", "subphase": "E1", "order": 35, "kind": "method", "dir": "croco", "title": "CroCo", "status": "todo"},
    {"id": "E1:dust3r", "phase": "E", "subphase": "E1", "order": 36, "kind": "method", "dir": "dust3r", "title": "DUSt3R", "status": "todo"},
    {"id": "E1:mast3r", "phase": "E", "subphase": "E1", "order": 37, "kind": "method", "dir": "mast3r", "title": "MASt3R", "status": "todo"},
    {"id": "E2:da3", "phase": "E", "subphase": "E2", "order": 38, "kind": "method", "dir": "da3", "title": "DA3", "status": "todo"},
    {"id": "E2:pi3", "phase": "E", "subphase": "E2", "order": 39, "kind": "method", "dir": "pi3", "title": "Pi^3", "status": "todo"},
    {"id": "E2:vggt", "phase": "E", "subphase": "E2", "order": 40, "kind": "method", "dir": "vggt", "title": "VGGT", "status": "todo"},
    {"id": "E2:vggt_omega", "phase": "E", "subphase": "E2", "order": 41, "kind": "method", "dir": "vggt_omega", "title": "VGGT-Omega", "status": "todo"},
    {"id": "B2:vggt_omega_eval", "phase": "B", "subphase": "B2", "order": 42, "kind": "task", "title": "CompEval adapter: VGGT-Omega", "artifact": null, "depends_on": "E2:vggt_omega", "status": "todo"},
    {"id": "F1:monst3r", "phase": "F", "subphase": "F1", "order": 43, "kind": "method", "dir": "monst3r", "title": "MonST3R", "status": "todo"},
    {"id": "F1:d2st3r", "phase": "F", "subphase": "F1", "order": 44, "kind": "method", "dir": "d2st3r", "title": "D2ST3R", "status": "todo"},
    {"id": "F1:streamvggt", "phase": "F", "subphase": "F1", "order": 45, "kind": "method", "dir": "streamvggt", "title": "StreamVGGT", "status": "todo"},
    {"id": "F2:dvlt", "phase": "F", "subphase": "F2", "order": 46, "kind": "method", "dir": "dvlt", "title": "Deja View (DVLT)", "status": "todo"},
    {"id": "F2:vdpm", "phase": "F", "subphase": "F2", "order": 47, "kind": "method", "dir": "vdpm", "title": "V-DPM", "status": "todo"},
    {"id": "B2:vdpm_eval", "phase": "B", "subphase": "B2", "order": 48, "kind": "task", "title": "CompEval adapter: VDPM", "artifact": null, "depends_on": "F2:vdpm", "status": "todo"}
  ]
}
```
