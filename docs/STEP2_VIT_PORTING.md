# Porting the Step-2 unified ViT-B/16 pretraining (playbook + progress)

This is the **source of truth** for adding the capture's **Step 2** to each method.
It is written to survive context loss: it records *why*, the *reusable pattern*, the
*per-method facts from the capture*, the *batch plan*, *progress*, and the
*gotchas*. Keep it updated as batches land.

## 1. What Step 2 is (fact, verified)

The capture's paper axis has two comparisons:
- **Step 1** — as-is frozen native/official backbone, evaluated on 5 downstream tasks.
- **Step 2** — from-scratch **unified ViT-B/16** (timm `vit_base_patch16_224`,
  `pretrained=False`, `num_classes=0` → 768-d CLS), 300 epochs, checkpointed at
  **100/200/300**, evaluated frozen. Every method plugs *its own SSL objective*
  into the **same** backbone; only the objective differs.

Verified from the capture (`origin/snapshots`): all `methods/*/configs/step2_vit*.yaml`
build `vit_base_patch16*`; none is a ResNet; registry `family: step2_vit`; unified
recipe **AdamW lr≈6e-4, wd 0.05, betas [0.9,0.999], 10-epoch warmup → cosine to
min_lr 1e-6, grad-clip 1.0, AMP, global batch 1024**. Consistent with the results
CSV. See docs/EVALUATION.md.

This repo previously ported only **native-arch, single-epoch** pretraining (neither
CSV Step 1 nor Step 2). The fan-out adds the unified ViT-B/16 Step-2 path per method,
**evaluated by the existing ImageNet linear probe** (the other 4 downstream tasks
stay out of scope).

## 2. The reusable pattern (established by the pilot, PR #84 = 06_rotation_prediction)

**Additive and non-destructive.** Selected by a new optional `arch: vit` key in the
config; absent / `arch: alexnet` (or the method's native arch name) is the native
path, byte-for-byte unchanged. No contract or `adapterlib.METRIC_VOCABULARY` change
(ImageNet eval only → same `pretrain`/`linear_eval` stages, same top-1/5 metrics).

Per method, add:
1. **`models/vit_<name>.py`** — timm `VisionTransformer(num_classes=0)` under
   `self.encoder` (so weights are `encoder.*`, matching the existing
   `ENCODER_PREFIXES=("encoder.",)` extract/load), plus the method's pretext/SSL
   **head** as a separate module. Expose `get_encoder()` returning a module whose
   `forward(x)` is the frozen representation the probe reads (usually the CLS token),
   so the existing `evaluate_linear_<name>.py` probes it **unchanged**. **Import timm
   lazily** (module top-level of this file only) so the native path never needs timm.
2. **`train_pretrain_vit_<name>.py`** — faithful to the capture's `train_step2_vit.py`:
   the unified AdamW + warmup→cosine + grad-clip + AMP(CUDA-only) recipe, the method's
   objective/loss (reuse the already-ported loss where the objective matches Step 1),
   from scratch (load only on `--resume`). Reuse `resolve_device` + `make_deterministic`
   from the native `train_pretrain_<name>.py`. Save `checkpoint_epoch_{N}.pth` at each
   `save_at_epochs` milestone + `checkpoint_latest.pth`. Return `final_loss` (+acc).
3. **`configs/pretrain_vit.yaml`** — `stage: pretrain`, `arch: vit`, the ViT knobs +
   `save_at_epochs: [100, 200, 300]`. **Must carry the DATA_ROOT convention line**
   (`test_data_root_convention` scans `pretrain*.yaml`):
   `#   --set DATA_ROOT=<path>   the dataset root; pretraining reads its train/ subdirectory`
4. **adapter (`adapter/__init__.py`)** — backward-compatible branches:
   - key sets: keep native `PRETRAIN_TRAIN_KEYS`; add `VIT_MODEL_KEYS`,
     `PRETRAIN_VIT_KEYS`, `EVAL_VIT_KEYS`. Read `arch = train.get("arch","<native>")`;
     validate the remaining keys against the arch-specific set (the native and ViT key
     sets are **disjoint** so knobs can't leak; test both directions).
   - `run_training` routes to the ViT trainer when `arch=="vit"`.
   - `load_encoder` builds the ViT for `arch=="vit"`.
   - `body` (pretrain, arch=vit): after training, write `encoder.pt` (final) **and**
     `encoder_epoch{N}.pt` for each milestone (extract `encoder.*` from
     `work/checkpoint_epoch_{N}.pth`).
   - `linear_eval` unchanged; the 100/200/300 sweep = run `linear_eval` once per
     milestone `encoder_epoch{N}.pt` (orchestration; no contract stage added).
5. **deps** — add `timm>=0.9.0` to `requirements.txt`; regenerate **both** locks.
   Fleet timm closure = **`timm==1.0.28`** (+ huggingface-hub / safetensors / hf-xet /
   anyio / httpx / tqdm / …). **LOW-RISK lock build** (do NOT naively `uv pip compile`
   the CPU lock — it pulls nvidia/triton via the wrong torch): reuse a timm-precedent
   method's proven closure. CPU: `grep '^[A-Za-z]' methods/22_mocov3/requirements.lock.txt
   | sed 's/ .*//' > /tmp/resolved.txt`, then
   `python3 bin/build-lock.py --resolved /tmp/resolved.txt --header <method-cpu-header> -o
   methods/<m>/requirements.lock.txt`. cu130: splice the method's existing cu130 header
   onto `22_mocov3`'s cu130 entries (identical closure). Verify with
   `python3 -m unittest tests.test_method_requirements`.

## 3. TDD bar (every method, every batch)

RED first, then implement, then GREEN; cover success / failure / edge:
- **config**: ViT config accepted; native config still accepted; missing ViT key
  refused by name; unknown key refused; native↔ViT key-set leakage refused both ways;
  bad `arch` refused by name.
- **model**: `get_encoder()(x)` is the expected feature per image; `forward` is the
  objective's output; `extract_encoder` keeps only `encoder.*` (head excluded);
  `load_encoder(arch=vit)` round-trips weights.
- **milestones**: a tiny ViT pretrain with `save_at_epochs=[1,2]` writes `encoder.pt`
  + `encoder_epoch1.pt` + `encoder_epoch2.pt`.
- **hermetic smoke** (tiny ViT dims for CPU speed): pretrain(vit) → milestone encoder →
  `linear_eval` → `contract-test` passes, comparable `*_linear_probe_top1/5_accuracy`
  written, eval writes no `encoder.pt`.
- **regression**: the method's existing native smoke stays green.
- **mutation**: kill one new guard (e.g. milestone extraction, or the arch key-set
  split) and record it.
- Gate a `needs_timm` skip on the ViT torch tests (timm may be absent in base env).

Verification per PR: guards (`test_stage_vocabulary`, `test_referenced_paths_exist`,
`test_encoder_convention`, `test_method_requirements`), `./tests/run-tests.sh` EXIT=0,
and a **torch+timm full discover** (`PYTHONPATH=. .venvs/<m>/bin/python -m unittest
discover -s tests`; rebuild a venv with the updated lock, or `pip install timm==1.0.28`
into it for local verification) before pushing. Docs updated (README, provenance,
docs/EVALUATION.md) for consistency. No AI attribution in commits/PRs.

## 4. Per-method capture facts (CNN-pretrain methods) — from `origin/snapshots`

Backbone is timm `vit_base_patch16_224(num_classes=0)`, CLS token, unless noted.
"Reuse Step-1 loss" = the objective equals the method's already-ported Step-1 loss.

| # | Method | Family | Head on ViT | Loss | Notes / data |
|---|--------|--------|-------------|------|--------------|
| 1 | context_prediction | pretext+CE | `Linear(768*2, hid)→Linear(hid,8)` on **two** patch CLS concatenated | CE over 8 relative positions | patch-pair dataset |
| 5 | jigsaw_puzzle | pretext+CE | `Linear(768,hid)→Linear(hid,100)` on single CLS | CE over 100 perms | **tiles reassembled into one image**, not 9 separate; shares `step2_runtime.py` |
| 9 | jigsaw_puzzle_pp | pretext+CE | `Linear(768,hid)→Linear(hid,701)` | CE over 701 | same shape as 5 |
| 10 | inst_disc | contrastive (NCE bank) | `Linear(768,feat)` | NCELoss, momentum memory bank m=4096, τ=0.07 | single-view + index; `update_memory` each step |
| 12 | cmc | contrastive (NCE bank) | two `ProjectionHead(768,hid,feat)` (L, ab) | NCESoftmaxLoss ×2 over NCEAverage bank K=65536 | Lab dataset, ImageFolderInstance |
| 13 | mocov1 | contrastive (InfoNCE) | `Linear(768,feat)` proj + momentum enc + queue 65536 | InfoNCE (in-model) | two-view; **EMA encoder + FIFO queue** |
| 14 | simclrv1 | contrastive (NT-Xent) | `Linear(768,2048,bias=F)→Linear(2048,out)` | NTXentLoss(τ) | two-view (stateless) |
| 15 | mocov2 | contrastive (InfoNCE) | MLP proj `Linear(768,768)→ReLU→Linear(768,feat)` + momentum enc + queue | InfoNCE | two-view + blur; **EMA + queue** |
| 16 | simclrv2 | contrastive (NT-Xent) | 3-layer proj | NTXentLoss(τ) | two-view (stateless) |
| 17 | swav | clustering | proj MLP + `prototypes=Linear(out,3000,bias=F)` | swapped-assignment + **Sinkhorn-Knopp** (3 it, ε=0.05), τ=0.1 | **multi-crop 2×224 + 6×96** |
| 18 | sela | clustering | `top_layer=Linear(768,k)` prototypes | soft/hard CE vs **Sinkhorn** | two loaders (assign + train) |
| 19 | byol | siamese | proj + `predictor`; **EMA target enc/proj (deepcopy, frozen)** | `-0.5(⟨p1,z2⟩+⟨p2,z1⟩)` L2-normed, target detached | two-view; EMA teacher + SyncBN; `test_step2_vit_target_bn_fix.py` |
| 20 | simsiam | siamese | proj(3×Linear+BN) + predictor | neg-cosine + **stop-grad** | two-view |
| 21 | barlow_twins | redundancy | `_build_projector(768,…)` + BN | cross-corr `(C-1)²+λ·offdiag²`, C=BN(z1)ᵀBN(z2) | two-view; all_reduce over C |
| 33 | pirl | contrastive (NCE bank) | `Linear(768,feat)`; forward_original / forward_jigsaw | PIRLMemoryBankNCE (image + jigsaw_weight·jigsaw) | original + 9 jigsaw patches; the capture reuses its PIRL pretext trainer script as the Step-2 entry |
| 3 | colorization | dense/gen | **hand-rolled ViT (not timm)** + CNN decoder → `[B,313,H,W]` | weighted CE over 313 ab-bins | reads **patch tokens**; bespoke |
| 4 | context_encoder | dense/gen | timm enc `global_pool=''` + `TransformerDecoder` + `Linear(512,16²·3)` | recon L2 on hole + **adversarial** | patch tokens + mask; discriminator; via `train.py --model_type vit` |
| 8 | split_brain | dense/gen | **two half-ViTs** `in_chans=1/2` + ConvTranspose decoders | CE(l)+CE(ab) | patch tokens; via `train.py --backbone vit` |

## 5. Batch plan and progress

One family = one PR. Order by reuse (high → complex):

- [x] **Pilot** — `06_rotation_prediction` (PR #84, merged). `Linear(768,4)` + CE.
- [x] **Batch 1 — pretext+CE:** `05_jigsaw_puzzle`, `09_jigsaw_puzzle_pp`,
  `01_context_prediction` (PR #85). 05/09 reassemble tiles into one image; 01 shares
  the ViT over two patches + concat head. 01's eval (`evaluate_linear_official`)
  gained an optional pre-built-encoder path + dynamic feature dim.
- [x] **Batch 2 — contrastive:** SimCLR `14_simclrv1`, `16_simclrv2` (stateless NT-Xent;
  PR #86) and MoCo `13_mocov1`, `15_mocov2` (PR #87; EMA momentum encoder + 65536 FIFO
  queue — both registered buffers/submodules, so they ride in `state_dict()` and resume
  restores them; InfoNCE in-model returning `(loss, logits, labels)`; AdamW betas
  (0.9, 0.95), no AMP/clip per the capture's ViT MoCo loop; mocov2 adds the 2-layer MLP
  head + the two-view dataset's Gaussian blur). The queue size must divide the batch.
- [x] **Batch 3 — siamese + redundancy** (PR #88): `20_simsiam` (stop-grad on z, not p;
  3-layer projector + 2-layer predictor; predictor gets warmup+cosine too), `19_byol`
  (EMA target encoder+projector, symmetric neg-cosine; **AMP + grad-clip**, unlike the
  MoCo/SimSiam ViT trainers; target-BN semantics reduce to `model.train()` + frozen
  target params single-process), `21_barlow_twins` (cross-correlation loss in-model,
  reuses `off_diagonal`/`_build_projector`). SimSiam/Barlow eval hardcoded `in_dim=2048`
  → made arch-aware (`embed_dim`); BYOL eval was already dynamic. Reused-native-module
  imports (`vit_byol`, `vit_barlow`) use a package/standalone import fallback for the
  test's `load_from`.
- [x] **Batch 4 — NCE memory-bank** (PR #89): `10_inst_disc` (single ViT + fc; bank in the
  checkpoint), `12_cmc` (two ViT branches, in_chans 1/2, Lab split, two banks; probe
  concatenates both CLS), `33_pirl` (jigsaw view = nine patches reassembled into one image,
  single ViT; memory bank + jigsaw NCE, grad-clip). All reuse each method's index-carrying
  dataset + NCE module; the bank rides in the checkpoint (10/12) or is re-seeded from the
  model (33, initialize_from_model). Eval `in_dim` is dynamic for all three (no eval change);
  10/12/33 native trainers/evals import no tensorboard, so the Batch-3 timm-only-venv hazard
  does not recur here. Gotcha: 33's load_encoder must default feature_dim/num_patches
  (they shape only the excluded projector), since the linear_eval config omits them.
- [x] **Batch 5 — clustering (PR #91):** `18_sela` (single `top_layer=Linear(768,k)`
  prototype head, per-epoch Sinkhorn-Knopp OT, AdamW+AMP; reuses
  `compute_hard_sinkhorn_assignments`+`create_indexed_train_loader`), `17_swav`
  (proj MLP + `prototypes`, multi-crop list-forward mirroring `ResNetSwAV` so the
  native `train_epoch`/`swav_loss`/distributed-Sinkhorn are reused; AdamW, milestone
  ckpts), `07_deepcluster` (from-scratch ViT + reset-each-epoch head; reuses the port's
  **faiss** `extract_features_for_clustering`+`run_kmeans` and `DeepClusterDataset`,
  AdamW+cosine+AMP). All three: `arch` absent == native (17/07 native carry no `arch`
  key; 18 native `arch: resnetv2`), disjoint ViT/native key sets, arch-aware eval
  `in_dim` (embed_dim). 17 has a `tensorboard` floor → lazy-import fix + `@needs_torch`
  smoke gate (Batch-3 hazard). **07 is faiss GPU/x86_64-only**, so its ViT smoke is
  gated `@needs_faiss`+`@needs_timm` and its `encoder.pt` union-prefixes
  `features.`/`classifier.` (AlexNet) with `backbone.` (ViT). 07's locks were
  regenerated with uv (constrained to the existing pins) rather than spliced: 07's
  CUDA closure already pinned `packaging==26.3` vs the fleet's `26.2`, which a splice
  cannot reconcile.
- [x] **Batch 6 — dense/generative (PR #92):** `03_colorization` (a **self-contained,
  hand-written** ViT-B/16 -- no timm -- reading the L channel + a CNN decoder for the same
  313-bin ab classification; trunk under `self.encoder`, `get_encoder()`->CLS, so the eval is
  unchanged and the closure stays torch-only), `08_split_brain` (dual half-width ViT-B/16
  branches, embed_dim 384/6 heads, per-branch in_chans 1/2; `net1.encoder.`/`net2.encoder.`
  prefixes unchanged; ab-CE + L-CE reaches every parameter; needs timm), `04_context_encoder`
  (the one GAN: ViT-B/16 encoder + a transformer decoder predicting the hole patches, always
  adversarial with two AdamW optimisers, reusing the shared `Discriminator`; `encoder.pt` is
  `encoder.*` only -- the ViT has no `fc.`, so the native `("encoder.","fc.")` set still keeps
  it; arch-aware eval reads `get_features` mean patch-tokens; needs timm). Pattern notes: 03's
  ViT trunk pos-embed is sized to the **crop**, not the resize `img_size`; 04 builds the encoder
  as `VisionTransformer` directly (not `create_model('vit_base_patch16_224')`) so a tiny CPU
  smoke runs; 03/08/04 reuse each method's native dataset + eval unchanged (eval already
  arch-agnostic or `get_encoder()`-based). Locks: 03 no timm (self-contained); 08 timm delta via
  uv-constrained regen (packaging==26.2 fleet); 04 already had packaging==26.2 so the timm delta
  was appended to its shared closure (blocks with `# wheel.whl` comments the coverage test reads).

- [ ] **Batch 7 — ViT-native (config-alignment; pilot landed, fan-out pending):**
  `22_mocov3` (already `vit_base`), `23_dino`, `24_beit`, `25_mae`, `26_simmim`,
  `27_ibot`, `29_ijepa`, `31_dinov3`, `32_nepa`, `34_msn`, `35_vjepa`, `37_lejepa`.
  Their pretrain is already a ViT but their paper's size (23/27 vit_small, 25 vit_large;
  22 already vit_base) and recipe, not the unified ViT-B/16 + Step-2 recipe. **Different
  from the fan-out:** same objective/head/dataloader, but `arch: vit_base` + the unified
  recipe (fixed wd, milestones). **Selector is a new explicit `recipe: unified` key**
  (absent = native) — because these methods already use `arch` for model size, so the
  fan-out's `arch: vit` selector can't apply. The capture ships a per-method
  `configs/step2_vit_b.yaml` + `train_step2_vit_b.py` (drop its DDP/`step2_protocol`
  resume machinery). **Pilot: `23_dino` (PR #TBD)** — additive `configs/pretrain_vit.yaml`
  (`recipe: unified`, `arch: vit_base`) + `train_pretrain_vit_dino.py` (reuses
  `build_dino`/`get_dino_dataloader`/in-model loss; fixed wd; milestone ckpts) + adapter
  `recipe`-branch (disjoint key sets: native `weight_decay_start/end` vs unified
  `weight_decay`+`save_at_epochs`) + milestone `encoder_epoch{N}.pt`; `load_encoder`/eval
  unchanged (read `arch`/`img_size`). DINO's own ViT supports `vit_base`, so no timm/lock
  change. Per-method gotchas to assess before fan-out: whether each method's ViT/build
  accepts `vit_base` (MAE=vit_large, iBOT/others differ), whether the native trainer needs
  a separate `train_pretrain_vit_<name>.py` or can be extended, and 22 (already vit_base →
  recipe/milestone alignment only).

  **Sub-PR progress (split into reviewable slices, ~2-4 methods each):**
  - **7a (PR #93):** `23_dino` (pilot), `37_lejepa`. Establishes the `recipe: unified`
    pattern. `load_encoder`/eval unchanged; DINO no timm, LeJEPA timm already a dep.
  - **7b (pushed; PR after #93 merges):** `24_beit`, `22_mocov3`, `25_mae`. All
    `recipe: unified`, native paths byte-for-byte unchanged, no lock changes. 24 = same
    MIM/tokenizer, keep betas 0.9/0.999. 22 = fixed EMA momentum + direct `lr` (two-way
    disjoint keys: unified `{lr,clip_grad,save_at_epochs}` vs native `{learning_rate,
    momentum_cosine}`). 25 = vit_large→vit_base + **adds** cosine+warmup (native fixed-LR
    trainer lacks it); ships `configs/linear_eval_vit.yaml`.
  - **7c (in progress):** `27_ibot` **DONE** (committed on `port/vit-step2-batch7c-vit-native`).
    The nested config took `recipe` as a **top-level** key (stripped before the TOP_KEYS
    check); a fresh single-process `train_pretrain_vit_ibot.py` (the native loop bakes in
    `lr × batch/256`, so it could not be reused) reusing the native pure helpers
    (schedule/setter/clip/meter/device/seed — DRY); fixed wd, `mask_ratio_min/max` (the
    loader's `step="step2"` path, **already carried** in `data/multicrop.py`), `grad_clip
    0.3`, `freeze_last_layer 3`, dropped health keys; two-way-disjoint key sets
    (unified `{weight_decay,save_at_epochs,mask_ratio_min/max}` vs native
    `{weight_decay_start/end,checkpoint_health,fail_fast_after_epoch,save_freq,pred_ratio,
    pred_ratio_var,pred_start_epoch}`); widened `ARCHS`(native vit_small only)/`UNIFIED_ARCHS`
    (vit_base)/`EVAL_ARCHS`(both)/`ARCH_EMBED_DIM`/`load_encoder`. The models (`vit_base`),
    the loader (step2) and the eval (`_EMBED_DIMS`/`_VIT_BUILDERS` already list vit_base)
    were all already carried, so **no model/loader/eval edits and no lock change** were
    needed. Ships `configs/pretrain_vit.yaml` + `configs/linear_eval_vit.yaml`. Full 27
    module 75 tests OK (255s < 300s CI bound); base gate EXIT=0; key-set-split guard
    mutation-killed by the both-ways leakage tests.
    `29_ijepa` **DONE** (recipe: unified in `train`, flat-config like 22/25). Notable:
    the native `train_pretrain_ijepa.py` **already implements the recipe** — it uses
    `lr` directly (no ×bs/256), builds the config-named arch (`vit_base` is a builder),
    reads `augmentation: step2` from the config, and its cosine wd is constant when
    `weight_decay==final_wd`. So **no separate `train_pretrain_vit_ijepa.py`**: the native
    trainer was extended with a **guarded** milestone save (`checkpoint_epoch_{N}.pth` at
    `save_at_epochs`, empty for the native config → byte-for-byte unchanged), the DRY
    choice over duplicating an identical loop (the playbook's "or can be extended" case).
    Two-way disjoint keys: native-only `use_horizontal_flip` (step2 aug ignores it) vs
    unified-only `save_at_epochs`. Smoke keeps `vit_tiny` for CPU speed (arch is a shared
    key, not the selector); `vit_base` covered by config translation + a `load_encoder`
    round trip. `captured_sha256` empty (nothing pinned), models/loader already carry
    `vit_base`/`step2`, so no model/loader/eval/lock change. Full 29 module 45 tests OK
    (62s); base gate EXIT=0; key-set-split mutation-killed.
    `34_msn` **DONE** (recipe: unified in `train`, deit_base via the pinned
    `third_party/msn` submodule — no upstream edit). Same shape as 29: the native
    `train_pretrain_msn.py` already implements the recipe (`src.msn_train.init_opt`
    takes `lr` directly; arch built from config dims → `deit_base` = embed_dim 768/
    heads 12; cosine wd constant when `weight_decay==final_weight_decay`), so **no
    separate trainer** — extended with a guarded milestone save. `save_at_epochs` is
    the sole unified-only key (one-way leakage guard, refused on native), since the
    capture's step2 changes only values, not the key set. Smoke keeps tiny dims for
    CPU speed; deit_base covered by config translation. Full 34 module 41 tests OK
    (65s); base gate EXIT=0; recipe-split mutation-killed.
    `31_dinov3` **DONE** (milestone-only, **no `recipe` key** — this config is already
    the unified ViT-B/16 Step 2). Added a required `save_at_epochs` key (declared in the
    sole pretrain key set), a guarded trainer milestone save (`checkpoint_epoch_{N}.pth`;
    empty list → only `checkpoint_latest.pth`, prior behaviour), and body extraction of
    `encoder_epoch{N}.pt` (teacher backbone, `backbone.` from `teacher_state_dict`). Full
    31 module 33 tests OK (56s); base gate EXIT=0; milestone-save mutation-killed by the
    milestone test.
    `35_vjepa` **DONE** (milestone-only, **no `recipe` key** — already unified ViT-B/16
    Step 2; pinned `third_party/jepa` submodule, `ENCODER_PREFIX=""` from
    `target_encoder_state_dict`). Same shape as 31: declared `save_at_epochs`, guarded
    trainer milestone save, body extraction of `encoder_epoch{N}.pt`. Full 35 module 34
    tests OK (73s); base gate EXIT=0; milestone-save mutation-killed.

  **7c COMPLETE and MERGED** (PR #95: 27_ibot, 29_ijepa, 34_msn, 31_dinov3, 35_vjepa).
  - **Separate PR — `26_simmim` DONE** (branch `port/vit-step2-simmim`). SimMIM's
    native backbone is a **Swin**, so the unified Step 2 is a genuinely different
    backbone (a timm **ViT-B/16**), not a re-tune — the only **non-additive-to-eval**
    port. New `models/simmim_vit.py` (`build_simmim_vit` + `build_vit_encoder`, ViT
    dims threaded through `timm.create_model` so a tiny CPU smoke runs), new
    `train_pretrain_vit_simmim.py` (single-process, direct lr, warmup→cosine, pixel
    mask via `return_pixel_mask=True`, milestone saves). Adapter `recipe: unified`
    branch (two-way disjoint keys: native `window_size`/`depths`/multistep-decay ↔
    unified `depth`/`mlp_ratio`/`min_lr`/`save_at_epochs`), ViT `load_encoder` branch,
    and `evaluate_linear_simmim.py` gains a `pool` arg — **CLS for the ViT, mean for
    the Swin** — fixed from the recipe by `adapter.eval_pool`. Full 26 module 47 tests
    OK (90s); base gate EXIT=0; two guards mutation-killed (pool decision; recipe
    key-set split). timm already a dep → no lock change.

  **ViT Step-2 fan-out COMPLETE**: all Step-1&2 methods that admit a unified ViT-B/16
  Step 2 are ported (Batches 1–6 + 7a/7b/7c + 26_simmim). Remaining = the deferred
  generative / eval-only-download efforts below (different nature).

  Consistency (verified across 23/37/24/22/25): uniform `RECIPES`/recipe-strip/routing/
  milestone `encoder_epoch{N}.pt`; each `pretrain_vit.yaml` carries `recipe: unified` +
  `save_at_epochs` + the DATA_ROOT line; no native config gained a `recipe` key.

**Eval-only-download methods — adding their from-scratch unified Step 2:**
These three (`28_dinov2`, `30_aim`, `36_franca`) were first ported **eval-only**
(the CSV's Step-1 "as-is" cell: download official weights → probe), because their
*original* pretraining data is non-public. But the capture ships a **self-contained
from-scratch ViT-B/16 Step 2** for each (`train_step2_vit.py` + own models), and the
unified Step 2 trains on ImageNet-1k for everyone — so it removes the data confound
and puts them on the same axis. Ported as full method-ports (add a `pretrain` stage
+ a Step-2 `linear_eval`, keeping the eval-only Step-1):
- **`28_dinov2` DONE** (branch `port/vit-step2-dinov2`). DINO+iBOT+KoLeo ViT-B/16
  from scratch; `models/`+`data/` ported from the capture (timm ViT); single-process
  `train_pretrain_vit_dinov2.py`; adapter grew a `pretrain` stage + a `recipe: unified`
  Step-2 eval (probes the trained `teacher_bb` encoder.pt at its CLS token) alongside
  the eval-only Step-1 download probe; **timm added to the lock at fleet `1.0.28`**.
  Full Step-2 paths verified (model round-trip, pretrain→contract+milestones,
  Step-2 eval→contract); base gate EXIT=0; eval recipe-split mutation-killed.
- **`36_franca` DONE** (branch `port/vit-step2-franca`, stacked on 28). Reuses the
  DINOv2 ViT backbone + KoLeo (vendored into `models/`, since methods are isolated) and
  adds Franca's own contribution: nested **MatryoshkaHead** + **Sinkhorn-Knopp**
  DINO/iBOT losses (no centering). The capture's `franca_data` cross-imported 28's
  augmentation — vendored here to avoid a `data` package collision. Single-process
  `train_pretrain_vit_franca.py`; adapter grew a `pretrain` stage + a `recipe: unified`
  Step-2 eval (probes the trained `teacher_bb` encoder.pt at CLS) alongside the eval-only
  Step-1 download probe; **timm added to the lock at fleet `1.0.28`** (same closure as 28).
  Step-2 paths verified (model round-trip, pretrain→contract+milestones, Step-2 eval→
  contract+comparable); base gate EXIT=0; eval recipe-split mutation-killed.
- **`30_aim` DONE** (branch `port/vit-step2-aim`, stacked on 36). Autoregressive
  (prefix-LM next-patch pixel MSE) — a different family from DINOv2/Franca, so an
  independent port: `models/aim_vit.py` (the lab's own torch-only AIM re-impl) +
  `data/aim_dataset.py` copied verbatim, single-process `train_pretrain_vit_aim.py`,
  adapter `pretrain` + `recipe: unified` Step-2 eval (probes the trained trunk;
  `encoder.pt` excludes `predictor.`; the eval averages the last N blocks + mean-pool).
  **No lock change** (AIM is torch/numpy only — no timm). **Licence marked**: the
  Step-2 uses the lab's own from-scratch code (not apple's), so apple-amlr binds only
  Step-1. TDD caught a `grad_clip`/`clip_grad` key mismatch (fixed). Step-2 paths
  verified; base gate EXIT=0; eval recipe-split mutation-killed.

  **Cross-cutting fix in this PR:** converting all three eval-only ports to two-stage
  leaves the repo with **zero** wholly-eval-only ports, so
  `test_encoder_convention.test_both_shapes_are_present_so_the_split_is_exercised`
  (which required ≥1 eval-only port) was updated: the no-encoder shape is now carried
  by the `linear_eval` stage of the two-stage methods, checked via any port that
  declares `_absent_reason` (non-vacuous; fails if no port declares the no-encoder path).

  **Eval-only-download trio COMPLETE** (28_dinov2, 36_franca, 30_aim) — each now has
  its from-scratch unified Step 2 alongside the unchanged eval-only Step-1 probe.
- **Generative** (`image_gpt`, `mar`, `var`): the from-scratch Step-2 is generative;
  the probe-target question (docs/EVAL_DOWNLOAD.md) is separate. Scope later.

**Gap methods missed by the fan-out** (native Step-1 landed after the batches;
the capture ships a `step2_vit.yaml` + `models/*_vit.py` + `train_step2_vit.py`):
- **`11_cpc`** — ported in its own PR (CPC on the ViT patch grid; additive
  `arch: vit`; timm added at fleet `1.0.28`).
- **`32_nepa` DONE** (branch `port/vit-step2-nepa`). Milestone-only / config-align
  (like 31/35): the native port already ports `NEPAModel`, so Step 2 is the **same
  model + trainer** at the unified setting — **no new model, no arch key, no timm**.
  `configs/pretrain_vit.yaml` (patch_size 16 vs step-1 14, `augmentation: step2`,
  SwiGLU, unified AdamW, `save_at_epochs: [100,200,300]`) + `configs/linear_eval_vit.yaml`
  (`pool: embed`, the capture's step2 raw-patch-embedding probe). Additive: the
  trainer honours `save_at_epochs` (absent = native, unchanged); the dataset gained
  a `step2` augmentation; the eval gained an optional `pool` (default `avg`); the
  adapter treats `save_at_epochs`/`pool` as **optional** keys (native and Step-2
  share one key set) and extracts milestone `encoder_epoch{N}.pt`. Two guards
  mutation-killed (milestone save; optional-key allowance); full 32_nepa module OK
  under torch; base gate EXIT=0. **No lock change** (torch-only).

## 6. Complexity flags (schedule carefully)
EMA + queue: 13, 15. Memory bank + resume state: 10, 12, 33. EMA teacher + target BN:
19. faiss k-means: 7. Multi-crop + distributed Sinkhorn: 17. Sinkhorn + second loader:
18. Adversarial + transformer decoder: 4. Dual half-ViTs: 8. Custom non-timm ViT: 3.

## 6a. Batch 1 detailed implementation notes (pretext+CE: 05, 09, 01)

Exact capture specs (from `origin/snapshots`). All backbones: timm
`vit_base_patch16_224(num_classes=0)` from scratch; AdamW lr 6e-4, wd 0.05,
betas [0.9,0.999], 10-epoch warmup→cosine to min_lr 1e-6, grad-clip 1.0, AMP,
`save_at_epochs 100/200/300`, seed 42, batch 1024.

- **05_jigsaw_puzzle (100)** — the 9 permuted tiles are **reassembled into ONE
  `[3,224,224]` image** in the dataset, then fed as a single image. Head:
  `LayerNorm(768)→Linear(768,2048)→GELU→Dropout(drop_rate)→Linear(2048,100)`,
  Linears `trunc_normal_(std=0.02)`, bias 0. Loss `CrossEntropy`. Config data:
  `tile_size 72, tile_gap 4, puzzle_size 224, image_size 256` (3·72+2·4=224),
  drop_rate 0.1, hidden_dim 2048, augmentation type1 (ColorJitter 0.4 + grayscale
  0.2 per tile before assembly). **Local `data/jigsaw_dataset.py` returns tiles
  `[9,..]` and dropped the assembly path** → add a new ViT dataset that emits the
  assembled `[3,224,224]` + permutation label (native dataset untouched; not
  pinned). Reuse the permutation generation (derangements, min Hamming 3, seed 42).
- **09_jigsaw_puzzle_pp (701)** — same shape/head (→701). Data: `tile_size 74,
  tile_gap 1, puzzle_size 224, image_size 256`, `grayscale_prob 0.7,
  max_occlusions 2`, drop_rate 0.0, **per-tile independent normalization** (own
  mean/std, NOT ImageNet), 701 perms cached. Local `data/jigsaw_pp_dataset.py`
  explicitly dropped the `puzzle_size` assembly → add a ViT dataset that assembles
  (70% grayscale grid, 0–2 occlusion tiles from another image, per-tile norm).
- **01_context_prediction (8)** — **two patches share one ViT encoder**; each →
  CLS `[B,768]`; concat `[B,1536]`; head
  `LayerNorm(1536)→Linear(1536,2048)→GELU→Drop→Linear(2048,2048)→GELU→Drop→Linear(2048,8)`.
  Forward takes `(center, context)`; loss CE over 8 positions. Config data:
  `patch_size 224, patch_gap 48, image_size 255`, drop_rate/attn_drop 0.1,
  `lr_scale_by_batch true` (1.5e-4·1024/256=6e-4). **The pinned
  `data/context_dataset_official.py` already yields two patches + an 8-way label**
  — reuse it; confirm/parameterize the patch size to 224 for the ViT (native uses
  the AlexNet size). It is byte-pinned (captured_sha256), so do NOT edit it; if a
  224 patch size is not reachable via its params, add a thin ViT dataset wrapper.

Trainer note: 05/09 are single-input (image,label); **01 is two-input**
(center,context,label) — the ViT trainer's step passes both to `model(center,
context)`. All three: milestone save + `checkpoint_latest.pth`, return final_loss.

## 7. Reference implementation
`methods/06_rotation_prediction/` — `models/vit_rotation.py`,
`train_pretrain_vit_rotation.py`, `configs/pretrain_vit.yaml`, and the adapter's
`arch`-branching + milestone extraction. Copy this shape; swap the head + loss + data
per the table above.
