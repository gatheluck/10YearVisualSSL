# Real-run verification: what the tests guarantee today, and the short-epoch matrix to build

Last updated: 2026-08-27

This document records, **fact-based and measured**, the state of the test suite as
of 2026-08-27 and the design for the next phase: a **short-epoch real-run harness**
that drives every method across every downstream task on the actual `launch.py`
chain and checks by machine that every weight and evaluation artifact lands where
it should. It exists so the analysis and the plan survive across sessions.

It is the port-side companion to `docs/EVALUATION.md` (which records the capture's
full evaluation design and the current gap) and `docs/DOWNSTREAM.md` (the
cross-method downstream subsystem). Read those two first.

---

## 1. Measured state of the test suite (2026-08-23)

All numbers below are from a run on this machine, not an estimate.

| Measurement | Environment | Result |
|---|---|---|
| Base suite (`./tests/run-tests.sh`) | no dependencies | **EXIT=0**, 2546 tests, **947 skipped**, ~104 s |
| Downstream 4 tasks | `.venvs/06_rotation_prediction` (torch+timm+pycocotools/h5py/av) | **60/60 OK, 0 skipped**, ~71 s |
| Full suite in-process | a torch venv (`.venvs/06_…`) | **EXIT=0**, 2546 tests, **149 skipped**, ~3712 s (~62 min) |

The full in-process run (`.venvs/06_rotation_prediction`, which carries
torch+timm+downstream deps, and -- measured 2026-08-24 -- `huggingface_hub` too)
leaves only 149 skips — the CUDA/GPU path, a few other methods' specific deps that
06's venv does not carry, and checkout-gated tests: `var`/`aim`, for instance, skip
not for a missing dep (06 carries `huggingface_hub`) but because those tests need a
pinned external checkpoint via `needs_checkout`, which is absent here.
No failures and no in-process `models/` namespace collisions.
This is one cell of CI's `locked` matrix, reproduced locally; CI runs it per
method.

The 947 base-env skips are **deps-gated by design** (CLAUDE.md's documented trap:
"the base run skips deps-gated tests"). The largest buckets, measured:

- 112 — the ViT Step-2 path (`arch: vit`, needs timm)
- ~600 — per-method tests needing torch/torchvision (+ tensorboard/Pillow/numpy)
- 57 — PyYAML absent
- 38 — no CUDA device (the GPU path)

None of these hides a failure, because **CI runs them for real** in four layers
(`.github/workflows/tests.yml`, measured 2026-08-23):

1. `base` — no deps, the base suite (deps-gated tests skip).
2. `discover` + `locked` — per method (discovered, not listed), install that
   method's `requirements.lock.txt` hash-checked and run the **whole suite
   in-process** (`unittest discover`). This is where in-process `models/`
   namespace collisions and deps-gated method tests actually execute.
3. `downstream` — install `downstream/requirements.lock.txt` hash-checked and run
   the four downstream smokes for real (they only ever skip in the `locked`
   matrix, because no method venv carries pycocotools/h5py/av).
4. `container` — per method, docker build + `verify-environment` +
   resolve/adapt/contract-test.

**So the suite is healthy and the CI structure is sound.** PR #112 merged green,
which means the `locked` matrix (per-method full in-process suite) passed for
every method.

---

## 2. What the tests guarantee — and what they do not (strict-TDD lens)

The single most important fact for the next phase: **every existing test is
hermetic** — synthetic data, tiny models, and mostly in-process. That is correct
for CI (it must download nothing and stay fast), but it means a whole class of
property is currently unverified.

### Guaranteed today (strong)

- **The contract chain's shape** (`tests/test_end_to_end.py`): resolve → adapter →
  contract-test, judged by exit status, through subprocesses. Catches artifact
  tampering, stray files, config edits after the run, and honest failure
  reporting. **But the adapter under test is `methods/_reference`, which trains
  nothing** — it is a known-good contract stub, not a real method.
- **Per-method smokes** (`tests/test_method_*.py`): 1–2 epochs on **random numpy
  data** with a **tiny model**, checking model shape, `encoder.pt` extraction,
  collate, config acceptance. Mostly in-process (the module is imported, not
  launched as a subprocess).
- **Downstream 4 tasks** (`tests/test_downstream_*.py`): a **random tiny ViT +
  synthetic data**, end to end through each task entrypoint and
  `downstream.contract.verify`. Every run stamps `subset_or_smoke: true` /
  `record_value: false`, so a smoke number can never be mistaken for a real one.
- **Reproducibility** (`tests/test_end_to_end.py::TestReproducibility`): same
  config → identical artifact digests; different config → different digests.

### Not guaranteed today (the gap this phase fills)

1. **No real-run verification.** Nothing drives the `launch.py` core path
   (resolve → submit → verify → record) for a *real* method on real (or
   real-shaped) data through a platform backend. `_reference` is a stub;
   per-method smokes are in-process on synthetic data.
2. **No method × task matrix driver.** The Step-1/Step-2 × 100/200/300 sweep
   driver named in `docs/EVALUATION.md §5.4` does not exist. There is no glue
   that feeds a ported method's `encoder.pt` / `encoder_epoch{100,200,300}.pt`
   into the downstream runners as a backbone.
3. **No cross-method output-layout check.** Each method emits its own
   `run_manifest.json` / `metrics.json` / `encoder.pt`, but nothing runs *all*
   methods × *all* tasks at a short epoch budget and decides by machine that
   every weight and evaluation artifact landed in the expected place.
4. **Downstream is still decoupled from real method backbones.** The downstream
   runners consume a generic backbone spec; no test feeds a ported method's real
   `encoder.pt` into a downstream task (the smokes use a random ViT only).

---

## 3. The plan (agreed 2026-08-23), strict TDD

Build a **short-epoch real-run harness** incrementally, RED test first at each
step. The point is to verify — before an ABCI run and, where hermetically
possible, in CI — that "run every task for every method for 1–2 epochs and every
weight/eval lands in the right place" holds by machine.

1. **Confirm the local full in-process suite completes** — **done (2026-08-23):**
   `.venvs/06_rotation_prediction` full `unittest discover`, **EXIT=0**, 2546
   tests, 149 skipped, ~62 min; no failures, no in-process collisions. CI's
   `locked` matrix runs the equivalent per method and is green.
2. **Short-epoch real-run smoke — done (2026-08-23):** `tests/test_real_run_smoke.py`
   drives `launch.py` for a real method end to end (resolve → local backend →
   contract-test → record) and decides by machine (contract exit 0 **and**
   `status: ok`, `encoder.pt` present, the linear-probe metric present) that every
   artifact landed at its expected path. It is the **first** test that runs
   `python -m adapter` for a real method through `launch.py` — every other
   end-to-end test drives the `_reference` stub, which trains nothing.
   - **First cell = a method's own `pretrain` → `linear_eval`** (the ImageNet
     axis): `pretrain` → `encoder.pt` → `linear_eval` consuming it. Stays inside
     the existing per-method adapter machinery — no cross-method downstream glue
     yet — the smallest foundation. The four downstream tasks are added in step 3.
   - **`local` backend, hermetic:** tiny real-shaped ImageFolder data, 1 epoch,
     so the real-run shape is fixed locally and in CI before spending ABCI time.
     `abci` is the same driver with different parameters (`--platform abci`, real
     `DATA_ROOT`/`*_ROOT`, a GPU), gated behind those exactly as
     `linear_eval`/downstream already are.
   - **Discover, never list (the guard `tests/test_no_hard_coded_methods.py`
     enforces).** A method declares its own short real-run in
     `methods/<m>/real_run_smoke.json` — its own directory, so no shared file
     names a method. The shared test discovers every such spec and drives it;
     adding a method with a spec extends coverage with no edit to the test.
     **This per-method spec is the registry step 3 reads**, so the foundation
     seeds the matrix driver. The spec lists ordered `stages`
     (`config`/`sets`/`overrides`/`produces`/`produces_metric`), a `data_shape`,
     and the `needs` imports that gate it. `06_rotation_prediction` ships the
     first spec.
   - **On CPU vs GPU:** on a GPU host the run resolves `device: auto` to CUDA and
     exercises the GPU path too; on a CPU host it runs on CPU. Both were measured
     green (2026-08-23).
   - **Not RED-then-implement:** the `launch.py` chain already worked for a real
     method (this cell needed no new production code), so the test was proven
     **non-vacuous by mutation** — skipping the contract verify in `launch.py`,
     and writing `encoder.pt` under a wrong name in the adapter, each make the
     test fail (both mutations killed; tree restored from a copy, never
     `git checkout --`).
3. **Method × task matrix driver — done (2026-08-23):** `bin/matrix-run.py`
   drives the discovered method × stage grid, reusing `launch.py` for each cell
   (one implementation of resolve → submit → verify → record, invoked, not
   copied), and emits **one machine verdict** — `matrix.json`, `status: ok` only
   when **every** cell is ok. It **discovers, never lists** (globs
   `methods/*/real_run_smoke.json`, names no method), threads a method's
   `encoder.pt` from the stage that produces it to the `@encoder` stage that
   needs it, resolves `@data` from a `--data SHAPE=PATH` mapping (a shape with no
   mapping is a reported failure, never a skipped cell), and a failed cell
   carries its output tail + job-log path (no silent failure).
   - **The same driver is the ABCI run.** Hermetic check = synthetic
     real-shaped data + default backend; the real cluster verification = real
     `--data` roots + `--platform <scheduler>` + a GPU. Only `--data` and
     `--platform` change; the short-epoch knobs come from each method's spec.
     This is exactly "run every task for every method for 1–2 epochs and check
     every weight/eval lands", as one command with one verdict.
   - **Tests (`tests/test_matrix_run.py`), two controls:** a *positive* (good
     data → every discovered cell present and ok, exit 0, `status: ok`, artifacts
     on disk) and a *negative* (empty data root → a cell fails → the whole grid
     fails, exit ≠ 0, `status: failed`). The negative makes the verdict
     falsifiable. **Non-vacuity proven by mutation** (`mutations/matrix-run.json`,
     2/2 killed): forcing `status` to `ok` regardless of cells, and forcing the
     exit code to 0, are each caught by the negative control.
   - **One implementation, shared.** `discover_specs`/`run_stage` live in
     `bin/matrix-run.py`; both the grid test and the step-2 smoke
     (`tests/test_real_run_smoke.py`, refactored 2026-08-23) import them via
     `tests/_real_run.py`, so the smoke and the grid cannot drift on "which
     methods declare a spec" or "how one stage is run".
   - **Grid today = discovered methods × their declared stages** (one method ×
     `pretrain`, `linear_eval`). It fans out automatically as methods add specs.
     **Still to fan out:** the four downstream tasks (a downstream stage /
     `data_shape` fed a real method `encoder.pt`) and the 100/200/300 Step-2
     milestones.
4. **Cross-method output-layout contract — done (2026-08-23):**
   `bin/matrix-audit.py` is the independent judge of a produced grid. It answers
   "did every declared weight and evaluation land?" from two sources only — the
   outputs on disk and each method's own `real_run_smoke.json` — and **never**
   from the matrix's own claim of success. Per cell (expectations re-derived from
   the spec, not the cell): the run directory and `out/run_manifest.json`
   (`status: ok`) exist, every declared `produces` file is present, the
   `produces_metric` is in `metrics.json`, and `launch.json` records `outcome:
   ok` + `contract_ok`. Across a method's cells: the stages present must be an
   in-order prefix of the declared ones, and a method whose last cell is ok must
   have run all its stages (no silently dropped final stage). It also fails if
   the matrix's own status disagrees with what the disk shows.
   - **It is a genuinely separate check, not a mirror of the driver.**
     `matrix-run.py` trusts `launch.py`/`contract-test` per run and records the
     cell outcome; the auditor re-reads the tree and enforces the *spec's*
     `produces`/`produces_metric` on disk, so a driver that self-reports `ok`
     with a missing `encoder.pt` does not survive (`contract-test` checks the
     generic manifest, not each method's declared artifacts).
   - **Tests (`tests/test_matrix_audit.py`):** mostly hermetic — a fabricated
     runs tree + `matrix.json` exercise the auditor in the base environment, with
     a positive (complete tree passes) and two negatives (delete a produced file;
     corrupt a manifest while the cell claims ok). One **integration** test runs
     the real `matrix-run → matrix-audit` chain under a method venv so the
     fabricated fixture cannot drift from the real layout. **Non-vacuity proven
     by mutation** (`mutations/matrix-audit.json`, 2/2 killed): removing the
     on-disk produced-file check, and accepting a failed manifest, are each
     caught by a negative control.
   - **One implementation, shared:** the auditor loads `discover_specs`/
     `SPEC_NAME` from `bin/matrix-run.py` rather than keep a second copy of
     "which methods declare a spec".

All four steps are done and gated green (base suite EXIT=0, 2026-08-23). Two
fan-outs remain and proceed with no change to the driver, the auditor, or the
tests -- coverage widens purely by methods declaring a spec: (a) more methods each
shipping a `real_run_smoke.json`, and (b) the four downstream tasks + 100/200/300
milestones wired into the grid.

### Fan-out (a): methods declaring a real-run smoke (in progress)

Each `real_run_smoke.json` is **verified by a real 1-epoch run under the method's
own venv**, not written from a template: the override keys are read from that
method's config schema (e.g. `train.img_size` for 10/12/13/14/15/16,
`train.image_size` for 18/19/33, `train.crop_size` for 08 where that is the only
spatial size and alongside `train.img_size` for 03, `train.warmup_epochs=0` where
the schedule warms up over many epochs, `train.queue_size`/`train.num_negatives`
shrunk for the momentum-queue and NCE-bank methods -- including `train.num_negatives`
for PIRL's memory bank (33) -- `train.k` shrunk for SeLa's self-labelling clusters,
and `train.rebalance_sample_size` shrunk for colorization's class-rebalancing prior
(03)). The matrix now spans both architecture families: eight ViT methods declare a
spec. Some own their ViT (`23_dino`, DINO's ViT-S/16, and `31_dinov3`, ViT-B/16 with
registers + RoPE -- `train.img_size`/`train.global_size`/`train.local_size` stay
divisible by the patch size 16, `train.n_local_crops` shrunk, temperature/LR warmups
zeroed); some are timm-backed (`22_mocov3`, ViT-Base, and `37_lejepa`,
`vit_base_patch16_224` -- timm interpolates its position embedding to `img_size=64`),
whose specs therefore list `timm` in `needs`. `25_mae` shows the masked-autoencoder
shape: its model is parameterisable (`models/mae_vit.py` says so explicitly), so the
default ViT-Large is shrunk to a tiny encoder/decoder via the dim keys -- applied
identically in both stages so `linear_eval` rebuilds the encoder `encoder.pt` holds.
The patch-14 JEPA family shrinks the same way: `29_ijepa` builds the tiny `vit_tiny`
variant its port added to `models/vision_transformer.py` (`train.name=vit_tiny`), and
`32_nepa` is built from explicit dims (`train.embed_dim`/`depth`/`num_heads`), both
kept identical across the two stages; each runs at `train.img_size=70`, divisible by
the patch size 14 (a 5x5 patch grid), with `train.warmup_epochs=0`. `11_cpc` (visual
CPC 2018) is the patch-grid predictive-coding shape: its dataset relaxes the paper's
7x7 grid to any `>=2x2` grid for a hermetic smoke, so `train.img_size`/`patch_size`/
`stride` shrink to a 2x2 grid (at which the InfoNCE loss still predicts one future
row, `train.pred_steps=1`) and the ResNet-v2-101 patch encoder is narrowed with
`train.z_dim`/`train.encoder_width_mult` -- every model and patch-grid key set
identically in both stages so `linear_eval` rebuilds the encoder `encoder.pt` holds.
Two jigsaw methods keep a large input on purpose: `05_jigsaw_puzzle` and
`09_jigsaw_puzzle_pp` cannot shrink `train.image_size` below `3*tile_size` (their 3x3
grid of 75px tiles), so both run at the default 255 (the AlexNet/CFN encoder is light
enough at that resolution). `04_context_encoder` is the one GAN (AlexNet inpainting):
its geometry is fixed by the architecture rather than shrunk -- the decoder
reconstructs a 128x128 centre hole and the discriminator sees that same 128x128, so
`mask_size` stays 128 and `img_size` stays 227, while the encoder's terminal
`AdaptiveAvgPool2d((7,7))` accepts any input so `linear_eval` need not match the
pretrain size; only the run-length knobs are overridden and the adversarial path runs.
`26_simmim` is the Swin masked-image-modeling shape: the default Swin-B is shrunk to a
tiny four-stage tower (`train.embed_dim=32`, `train.depths=[2,2,2,2]`,
`train.num_heads=[2,4,8,16]`) whose geometry still satisfies every divisibility the
port asserts -- `encoder_stride = train.patch_size * 2^(len(depths)-1) = 32`, and
`train.img_size=32` is divisible by both that stride and `train.mask_patch_size=8`
(itself a multiple of `train.patch_size=4`), so the mask grid matches the token grid
and the PixelShuffle decoder reconstructs to 32x32; every model key is set identically
in both stages so `linear_eval` rebuilds the same Swin. `var` (VAR next-scale
autoregressive generative pretraining) is the `EVAL_DOWNLOAD` generative shape
(`docs/EVAL_DOWNLOAD.md`): its probe reads the VQVAE tokeniser's features, not the VAR
transformer that `pretrain` trains, so `linear_eval` consumes no `encoder.pt` and the two
stages are independent. The full VAR-d16 recipe is shrunk to the tiny real architecture
the method's own test exercises on CPU (`train.patch_nums=[1,2,3]`, `train.vocab_size=16`,
`train.ch=32`, `train.num_classes=4`, `train.depth=2`); the pretrained VQVAE tokeniser is
a download, so the hermetic smoke leaves `VQVAE_CKPT` empty and a random VQVAE is built
instead (its accuracy is meaningless by design -- only the pipeline is exercised), with
`linear_eval` at `train.img_size=32` (a 2x2 map over the tokeniser's 16x downsample). The run is driven through `matrix-run -> matrix-audit` to confirm
`encoder.pt` and the linear-probe metric land. `image_gpt` (iGPT, generative
pretraining from pixels) is the first spec to thread a second artifact between its
own stages: unlike the `EVAL_DOWNLOAD` shape, its probe reads the model `pretrain`
trains (`IGPT.extract_features`), so `linear_eval` consumes both the `encoder.pt`
(via `@encoder`) *and* the `clusters.npy` colour clusters fit during pretraining
(via `@produces:clusters.npy`) -- the probe must quantise images with the same
colour space the model was trained on. `@produces:<file>` is the general form the
driver grew for exactly this (`@encoder` stays a backward-compatible alias for
`@produces:encoder.pt`); a request for a file no earlier stage produced is a hard,
named failure, never a silently empty `--set`. The iGPT-S recipe (`vocab_size=512`,
`img_size=32`, `n_layer=24`, `n_head=8`, `n_embd=512`) is shrunk to the tiny
architecture the method's own test exercises on CPU (`vocab_size=8`, `img_size=8`,
`n_layer=2`, `n_head=2`, `n_embd=32`), every model key set identically in both
stages so `load_encoder` rebuilds the same model. `24_beit` (BEiT masked image
modeling with dVAE visual tokens) follows the standard shape -- its probe reads the
model `pretrain` trains (the frozen BEiT backbone's mean-pooled patch tokens), so
`linear_eval` consumes `encoder.pt` via `@encoder`. Its MIM targets come from a dVAE
tokenizer that is the frozen OpenAI DALL-E encoder for a real run (a hash-pinned
download named in `provenance.json`), but -- exactly like `var`'s VQVAE -- the
hermetic smoke leaves `TOKENIZER_CKPT` empty so the adapter builds a *random*
tokenizer instead (its accuracy is meaningless by design; only the pipeline is
exercised), so nothing is downloaded and no submodule is needed. The paper's
ViT-Base/16 (`img_size=224`, `embed_dim=768`, `depth=12`, 8192 tokens,
`token_size=112`, `num_masking_patches=75`) is shrunk to the tiny architecture the
method's own test exercises on CPU (`img_size=32`/`patch_size=16` -> a 2x2 patch
grid, `token_size=16` -> a matching 2x2 token grid, `embed_dim=32`, `depth=2`,
`num_heads=4`, `vocab_size=16`, `num_masking_patches=2` of 4); every model key is set
identically in both stages so `load_encoder` rebuilds the same trunk, while the
tokenizer and masking keys are pretrain-only (the `linear_eval` config rejects them).
`28_dinov2` (DINOv2 self-distillation) is the first `EVAL_DOWNLOAD` eval-only shape in
this list (`docs/EVAL_DOWNLOAD.md`): the capture's "Step 1" is a linear probe on the
official pretrained DINOv2 ViT backbone -- a genuine SSL representation, so the number
is comparable -- and the from-scratch pretraining on the unavailable LVD-142M is the
excluded step, so the port has no `pretrain` stage at all and its smoke is a single
`linear_eval` that reads a frozen backbone and produces no `encoder.pt`. The official
ViT-g/14 checkpoint is a pinned sha256 download (`bin/fetch-weights.py`); like `var`
and `24_beit`, the hermetic smoke leaves `DINOV2_CKPT` empty so the backbone is built
`pretrained=False` (random), downloading nothing (its accuracy is meaningless by
design). The default `dinov2_vitg14` builder is swapped for the smallest,
`dinov2_vits14`, at `train.resolution=28` (a 2x2 patch14 grid) -- the tiny backbone the
method's own test exercises on CPU; the ViT's `embed_dim`/`depth`/`patch_size` are
fixed by the named builder and are not config keys, so only the builder name,
resolution and run-length knobs are overridden.
`30_aim` (AIM autoregressive image modeling) is the same eval-only shape, and reuses the
official AIM-600M backbone because AIM's from-scratch pretraining is on the non-public
DFN-2B+ -- so, like `28_dinov2`, it has no `pretrain` stage and its smoke is a single
`linear_eval` reading a frozen backbone, producing no `encoder.pt`, with `CKPT` left
empty so the backbone is random and nothing is downloaded. Unlike `28_dinov2` (whose ViT
is fixed by a named builder), AIM's dims are explicit config keys, so the AIM-600M recipe
(ViT-H/14, `embed_dim=1536`, 24 blocks, 12 heads) is shrunk directly to the tiny
architecture the method's own test exercises on CPU (`img_size=32`, `patch_size=16`,
`embed_dim=32`, `num_blocks=4`, `num_heads=4`, `num_feature_layers=2`), keeping
`num_feature_layers <= num_blocks`.
`36_franca` (Franca self-supervised ViT) is the same eval-only shape as `28_dinov2`: its
capture "Step 1" is a linear probe on the official pretrained Franca ViT-B/14 In21K
backbone -- a genuine SSL representation, so the number is comparable -- and its
from-scratch SSL pretraining is the excluded Step 2, so the port has no `pretrain` stage
and its smoke is a single `linear_eval` reading a frozen backbone, producing no
`encoder.pt`, with `FRANCA_CKPT` left empty so the backbone is random and nothing is
downloaded. Like `28_dinov2`, the architecture is fixed by the named builder
(`franca_vitb14`, the smallest, `embed_dim=768`), not by config dims, so only
`train.resolution` (518 -> 28, a 2x2 patch14 grid) and the run-length knobs are
overridden.
`38_clip` (CLIP contrastive image-text, Radford et al. 2021) is the same eval-only shape:
its capture "Step 1" reuses the released OpenAI ViT-B/32 image tower, freezes it and probes
its pooled `encode_image` embedding -- a genuine learned representation, so the number is
comparable, like `36_franca` -- because CLIP's from-scratch training is on the non-public
400M image-text WIT dataset, so the port has no `pretrain` stage and its smoke is a single
`linear_eval` reading a frozen backbone, producing no `encoder.pt`, with `CLIP_VITB32_CKPT`
left empty so the tower is built random (no `clip.load`, no download). Unlike `28_dinov2`
(named builder), CLIP's image-tower dims are explicit config keys, so the ViT-B/32 recipe
(`resolution=224`, `patch=32`, `width=768`, 12 layers/heads, `output_dim=512`) is shrunk
directly to the tiny tower the method's own test exercises on CPU (`resolution=32`,
`patch_size=16`, `width=64` -- kept a multiple of 64 because CLIP derives its head count as
`width // 64` -- `layers=2`, `heads=1`, `output_dim=16`). The pinned OpenAI `clip` package
instantiates its BPE tokenizer at import time, so `ftfy` and `regex` are needed even on this
eval-only path, which is why its spec lists them in `needs`.
Declaring a spec today (measured
2026-08-27 on `imagefolder_2class`; most are `pretrain -> linear_eval`, the eval-only
`28_dinov2`, `30_aim`, `36_franca` and `38_clip` are a single `linear_eval` stage each,
and `01_context_prediction` is a single `pretrain` stage -- its distributed init lives
only in pretrain and its native AlexNet `linear_eval` is not yet tiny-tested):

- `01_context_prediction`, `03_colorization`, `04_context_encoder`, `05_jigsaw_puzzle`,
  `06_rotation_prediction` (the pilot), `08_split_brain`, `09_jigsaw_puzzle_pp`,
  `10_inst_disc`, `11_cpc`, `12_cmc`, `13_mocov1`, `14_simclrv1`, `15_mocov2`,
  `16_simclrv2`, `17_swav`, `18_sela`, `19_byol`, `20_simsiam`, `21_barlow_twins`,
  `22_mocov3`, `23_dino`, `25_mae`, `26_simmim`,
  `24_beit`, `27_ibot`, `28_dinov2`, `29_ijepa`, `30_aim`, `31_dinov3`, `32_nepa`, `33_pirl`,
  `36_franca`, `37_lejepa`, `38_clip`, `var`, `image_gpt` -- thirty-six methods, each
  verified green.

**Every discovered spec runs under every method's `locked` venv, gated on `needs`.**
The smoke test runs a spec only when its `needs` import in the current venv:
twenty-six of the thirty-six need the same four (`torch`/`torchvision`/`numpy`/`PIL`)
and run everywhere, while `22_mocov3`, `26_simmim` and `37_lejepa` also need `timm`,
`var` and `30_aim` also need `huggingface_hub`, `38_clip` also needs `ftfy` and
`regex` (the pinned OpenAI `clip` package builds its BPE tokenizer at import), and
`17_swav`, `20_simsiam`, `21_barlow_twins` and `27_ibot` also need `tensorboard`
(their trainers write TensorBoard summaries via `torch.utils.tensorboard`), so those
ten run only under a venv carrying the extra dep and are skipped -- by the gate, not
silently -- elsewhere. Thirty-one of the thirty-six were measured green together under
the `26_simmim` venv (which carries `timm` and `huggingface_hub` but not `tensorboard`
or `ftfy`/`regex`), landing in ~543 s (measured 2026-08-27); the five gate-skips there
-- `38_clip` (needs `ftfy`/`regex`) and `17_swav`/`20_simsiam`/`21_barlow_twins`/`27_ibot`
(need `tensorboard`) -- were each verified green under their own `.venvs/<method>` via
`matrix-run -> matrix-audit`. No single venv carries every extra, so the whole
thirty-six cannot land in one run; together the venvs cover all of them -- a ported
method runs under another method's pinned deps, across both architecture families. The
cost grows with the spec count; if it becomes a burden the gate can be narrowed to the
owning method, but the cross-method run is itself a compatibility signal and is left on
for now.

**The single-process trap, and what is left.** `launch.py` always sets
`LOCAL_RANK=0`/`RANK=0`/`WORLD_SIZE=1` for a single-process local run
(`bin/launch.py`) but no `MASTER_ADDR`, and five methods' trainers --
`01_context_prediction`, `17_swav`, `20_simsiam`, `21_barlow_twins`, `27_ibot` --
used to call `dist.init_process_group(backend="nccl")` as soon as `LOCAL_RANK` was
present, which failed on a CPU host with `MASTER_ADDR expected, but not set`. The
harness surfaced this as a failed cell (never a silent pass). The fix keys "go
distributed?" off `WORLD_SIZE > 1` instead of `LOCAL_RANK`'s mere presence, so a
single-process run skips the process group entirely and resolves to CPU (the device
invariant, `docs/GPU.md`); every downstream DDP/`SyncBatchNorm`/`DistributedSampler`
path was already guarded behind `if distributed:`. Each guard is measured with a
mutation that must kill it (the two predicate shapes -- the shared
`if WORLD_SIZE<=1: return` and `01`'s `or WORLD_SIZE<=1` clause -- were each mutated
and confirmed to fail the smoke). `20_simsiam` also stood up a process group in its
`linear_eval` (`evaluate_linear_official.py`); it is fixed in both sites. The other
four's `linear_eval` never touched `dist`, so `01`'s smoke is a single pretrain stage
(the only stage the fix touches).

The one method still outside the hermetic local backend is `02_vae`: its adapter
already forces `distributed=False` (`methods/02_vae/adapter/__init__.py`), so it was
never nccl-blocked, but its pretrain reads `torchvision.datasets.MNIST` (a raw MNIST
directory), and `tests/_real_run.py::build_data` only fabricates the
`imagefolder_2class` shape. Giving `02_vae` a real-run means adding an `mnist`
`data_shape` to the harness; that is tracked as future work rather than worked around
with a broken spec.

Each step follows the repository discipline (CLAUDE.md): RED test first, judge by
exit status, a measured mutation spec for every new guard, discover-not-list, and
docs kept consistent. Where a real run needs real data and a GPU it cannot be
hermetic; that part is documented and gated behind a `*_ROOT` env var and a
platform backend, exactly as `linear_eval` and the downstream tasks already are.

---

## 4. Sources (so the reasoning can be re-checked)

- `docs/EVALUATION.md` — capture evaluation design vs. what the port implements;
  §5 lists the remaining sweep-driver + real-numbers work.
- `docs/DOWNSTREAM.md` — the cross-method `downstream/` subsystem (4 tasks) and
  its CI-only shared environment.
- `.github/workflows/tests.yml` — the four CI layers (`base`, `locked`,
  `downstream`, `container`).
- `bin/launch.py` — the resolve → submit → verify → record core path.
- `platforms/` — the platform layer (`local` default, `abci` backend);
  `docs/PLATFORMS.md` for the interface.
- `tests/test_end_to_end.py` — the contract chain on the `_reference` stub.
