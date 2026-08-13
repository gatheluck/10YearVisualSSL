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
  PR #86) and MoCo `13_mocov1`, `15_mocov2` (EMA momentum encoder + 65536 FIFO queue —
  both registered buffers/submodules, so they ride in `state_dict()` and resume restores
  them; InfoNCE in-model returning `(loss, logits, labels)`; AdamW betas (0.9, 0.95),
  no AMP/clip per the capture's ViT MoCo loop; mocov2 adds the 2-layer MLP head + the
  two-view dataset's Gaussian blur). The queue size must divide the batch.
- [ ] **Batch 3 — siamese + redundancy:** `20_simsiam`, `19_byol` (EMA teacher + target BN),
  `21_barlow_twins`.
- [ ] **Batch 4 — NCE memory-bank:** `10_inst_disc`, `12_cmc`, `33_pirl` (persist/restore the
  bank on resume; index-carrying dataset).
- [ ] **Batch 5 — clustering (individually):** `07_deepcluster` (faiss k-means),
  `17_swav` (multi-crop + distributed Sinkhorn), `18_sela` (Sinkhorn + second loader).
- [ ] **Batch 6 — dense/generative (bespoke, last):** `03_colorization` (custom non-timm ViT),
  `04_context_encoder` (adversarial + transformer decoder), `08_split_brain` (dual half-ViTs).

**Deferred, separate efforts (different nature, not this fan-out):**
- **ViT-native methods** (`22_mocov3`, `23_dino`, `24_beit`, `25_mae`, `26_simmim`,
  `27_ibot`, `29_ijepa`, `31_dinov3`, `32_nepa`, `34_msn`, `35_vjepa`, `37_lejepa`):
  their pretrain is already a ViT, but not necessarily the unified **ViT-B/16** with
  the Step-2 recipe (e.g. 23/27 = vit_small, 25 = vit_large). The task there is a
  Step-2 **config alignment** to ViT-B/16 + the unified recipe, not a new trainer —
  assess per method.
- **Generative** (`image_gpt`, `mar`, `var`) and **eval-only download** (`28_dinov2`,
  `30_aim`, `36_franca`): the capture also ran a from-scratch ViT Step-2 for these;
  scope later.

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
