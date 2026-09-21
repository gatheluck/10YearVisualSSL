# ImageNet-val feature sweep — reproducibility record

This document records how the ImageNet-1k **validation** feature dumps under
`/data/visual_ssl/features/imagenet-val/` were produced, so that a colleague can
consume them (or reproduce them) without guessing. Every number below was
measured on the machine that produced the dumps (Tesla T4, 15 GB); where a value
is an estimate rather than a measurement it says so.

The driver is `bin/extract-features.py` (the feature-extraction subsystem). This
document does not restate the driver's contract — see the code and its tests —
it records the **exact invocation, environment, and inputs** used for this run
and the **honest scope** of what was and was not extracted.

## 1. What you get

For every method that ran successfully, `<out>/<method>/` contains:

| file | shape / type | meaning |
|---|---|---|
| `features.npy` | `(N, D)` `float32` | one row per val image, in ImageFolder order |
| `labels.npy` | `(N,)` `int64` | class index per row (see §5 for the ordering) |
| `meta.json` | — | method, representation, `feat_dim`, `count`, arch, `image_size`, the exact `preprocessing` string, `data_root`, `split`, and `encoder_sha256` |
| `result.json` | — | `{status, feat_dim, count}` |

and `<out>/manifest.json` records one entry per **discovered** method with its
status (`ok` / `skipped` / `error`). `N = 50000` for this run (1000 classes ×
50 images).

**Representation.** This sweep used `--representation l2`: every feature row is
L2-normalised to unit length (measured mean norm 1.0). If your experiment needs
raw (un-normalised) features, re-run with `--representation raw` — the dumps here
are unit-norm and you cannot recover raw magnitudes from them.

## 2. How to load (standard library + numpy only)

```python
import numpy as np, json
d = "/data/visual_ssl/features/imagenet-val/data2vec2"
X = np.load(f"{d}/features.npy")   # (50000, 768) float32, unit-norm
y = np.load(f"{d}/labels.npy")     # (50000,)     int64, 0..999
meta = json.load(open(f"{d}/meta.json"))
```

Row `i` of `X` is the feature for the image whose label is `y[i]`. Rows are in
ImageFolder traversal order, so labels are non-decreasing (all class-0 images,
then class-1, …). Do **not** assume alignment across methods by row index alone —
they are all in the same ImageFolder order for the same `data_root`, so they do
align, but confirm `count` and the class histogram (each class appears exactly
50 times) before joining two methods' matrices.

## 3. Scope — what ran, what did not, and why

This run covers exactly the methods whose frozen encoder is a **pinned external
download** (an off-the-shelf backbone), because this environment does not train
Step-1/Step-2 encoders. It does **not** cover methods that probe a locally
*trained* `encoder.pt`.

- **Runnable now (pinned download backbone)** — extracted in this sweep. See the
  measured table in §7.
- **Blocked (needs a trained `encoder.pt` this environment does not have):**
  `24_beit`, `30_aim`, `38_clip`. These providers load a trained Step-2 trunk via
  the adapter, *not* the download artifact. Note the trap: `30_aim`/`38_clip` do
  carry a `backbone_artifact` in provenance, but their providers ignore it and
  require the trained checkpoint — so a present download does not make them
  runnable.
- **Gated:** `sam3` needs a user-supplied, entitlement-gated checkpoint;
  `bin/fetch-weights.py` cannot download it. Left unextracted.
- **Excluded by design:** `mar` is pretrain-only and ships no provider; all other
  numbered/unnumbered methods with no download backbone are `skipped` in the
  manifest with reason "no encoder.pt found".

**`var` caveat for the consumer.** `var`'s feature comes from the VAR **VQVAE
tokeniser** and has `feat_dim = 32`. Provenance itself flags that this is a
tokeniser code, not a learned linear-probe representation — treat it differently
from the ViT features when comparing.

## 4. Environment (GPU, cu130)

Execution was on GPU (`--device cuda`, Tesla T4, sm_75). Each method runs in its
own per-method venv under `.venvs-cu/<method>` (segregated from the CPU
`.venvs/`). The venvs were built from each method's `requirements.lock.cu130.txt`
plus the shared `requirements-tools.lock.txt`.

**Recipe actually used (differs from `docs/GPU.md` on one point):**

```bash
export UV_INDEX_STRATEGY=unsafe-best-match
uv venv .venvs-cu/<method> --python 3.12 --clear
VIRTUAL_ENV=.venvs-cu/<method> uv pip install --no-deps \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple \
  -r methods/<method>/requirements.lock.cu130.txt \
  -r requirements-tools.lock.txt
```

`--require-hashes` is **not** usable here: the cu130 index serves the `+cu130`
torch/torchvision wheels, whose hashes differ from the (cpu-index) hashes pinned
in the lock, so hash-checking cannot pass (measured error: "no hashes were
provided for torchvision==0.28.0+cu130"). Versions are still pinned by the `==`
lines, so `--no-deps` installs the exact pinned closure. This matches how the
pre-existing `.venvs-cu/data2vec2` venv was built. `docs/GPU.md`'s
`--require-hashes` recipe is stale on this point (flagged for a docs fix).

## 5. Data

`data_root = /data/visual_ssl/datasets/imagenet`, `split = val`. Layout is a
standard ImageFolder: `val/<wnid>/*.JPEG`, 1000 wnid directories × 50 images =
50000. The class index for a row is the position of its wnid in **sorted-wnid
order**, which is the canonical ImageNet-1k class index. See
`docs/IMAGENET_VAL_SETUP.md` and `bin/prepare-imagenet-val.py` for how the val
tree was materialised.

## 6. Reproducing the sweep

Three one-shot scripts drive it (kept in `/tmp` during the run; their exact
contents are inlined here so the steps survive). All paths are absolute and all
weights live under `/data/visual_ssl/weights/<method>/`.

### 6a. Fetch the backbones (multi-connection, sha256-verified)

`bin/fetch-weights.py` enforces the recorded sha256 but is single-stream and slow
(~10 min per ~350 MB file). The fetch below uses `aria2c -x16` and then verifies
the **same** provenance-recorded sha256 (a mismatch deletes the file); it is
idempotent. Every URL is the exact pinned source from the method's
`provenance.json`.

```bash
# method|filename|sha256|url   — verify each against methods/<m>/provenance.json
eva02|eva02_base_patch14_224.mim_in22k.pytorch_model.bin|65068e45…|https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k/resolve/<rev>/pytorch_model.bin
aimv2|aimv2_large_patch14_224.apple_pt.pytorch_model.bin|37861b19…|https://huggingface.co/timm/aimv2_large_patch14_224.apple_pt/resolve/<rev>/pytorch_model.bin
siglip|vit_base_patch16_siglip_224.webli.pytorch_model.bin|cdcf73d9…|https://huggingface.co/timm/vit_base_patch16_siglip_224.webli/resolve/<rev>/pytorch_model.bin
beitv2|beitv2_base_patch16_224_pt1k.pth|6d32a5f7…|https://github.com/addf400/files/releases/download/BEiT-v2/beitv2_base_patch16_224_pt1k.pth
cae|…_20221230-808170f3.pth|808170f3…|https://download.openmmlab.com/mmselfsup/1.x/cae/…
videomae|model.safetensors|bc053ca2…|https://huggingface.co/MCG-NJU/videomae-base/resolve/main/model.safetensors
vjepa2|model.safetensors|25466aef…|https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/resolve/main/model.safetensors
36_franca|franca_vitb14_In21K.pth|2a0888b5…|https://github.com/valeoai/Franca/releases/download/v1.0.0/franca_vitb14_In21K.pth
28_dinov2|dinov2_vitg14_pretrain.pth|baf8467e…|https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
var|vae_ch160v4096z32.pth|7c3ec27a…|https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
vjepa2_ac|vjepa2-ac-vitg.pt|0b5e3c4b…|https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
# data2vec2 was fetched during the pilot; cosmos3_super needs a vision_encoder/
# directory (config.json + model.safetensors) — pass the model.safetensors path,
# the provider takes its parent directory for from_pretrained.
```

The authoritative sha256 and revision for each is `methods/<m>/provenance.json`
(`backbone_artifact`, except `var` uses `tokenizer_artifact`). Always verify
against provenance, not against the truncated hashes above.

### 6b. Build the GPU venvs

See §4. Loop the recipe over: `eva02 aimv2 siglip beitv2 cae videomae vjepa2
36_franca 28_dinov2 var vjepa2_ac` (`data2vec2` and `cosmos3_super` venvs already
existed).

### 6c. Run the sweep (one invocation)

One invocation is used deliberately: the driver runs each method in its own
subprocess (a failure becomes `status=error` and the others continue) and writes
one consolidated `manifest.json`.

```bash
W=/data/visual_ssl/weights
PYTHONPATH=. python3 bin/extract-features.py \
  --data-root /data/visual_ssl/datasets/imagenet --split val \
  --out /data/visual_ssl/features/imagenet-val \
  --venvs-root .venvs-cu --device cuda \
  --batch-size 64 --num-workers 8 --representation l2 --allow-missing \
  --encoder "data2vec2=$W/data2vec2/data2vec-vision-base.pytorch_model.bin" \
  --encoder "eva02=$W/eva02/eva02_base_patch14_224.mim_in22k.pytorch_model.bin" \
  --encoder "aimv2=$W/aimv2/aimv2_large_patch14_224.apple_pt.pytorch_model.bin" \
  --encoder "siglip=$W/siglip/vit_base_patch16_siglip_224.webli.pytorch_model.bin" \
  --encoder "beitv2=$W/beitv2/beitv2_base_patch16_224_pt1k.pth" \
  --encoder "cae=$W/cae/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221230-808170f3.pth" \
  --encoder "videomae=$W/videomae/model.safetensors" \
  --encoder "vjepa2=$W/vjepa2/model.safetensors" \
  --encoder "36_franca=$W/36_franca/franca_vitb14_In21K.pth" \
  --encoder "28_dinov2=$W/28_dinov2/dinov2_vitg14_pretrain.pth" \
  --encoder "var=$W/var/vae_ch160v4096z32.pth" \
  --encoder "vjepa2_ac=$W/vjepa2_ac/vjepa2-ac-vitg.pt" \
  --encoder "cosmos3_super=$W/cosmos3_super/vision_encoder/model.safetensors"
```

`--batch-size 64` was chosen so the ViT-g giants (`28_dinov2`, `vjepa2_ac`) fit
in 15 GB; features are identical at any batch size. `--allow-missing` lets the
methods with no `--encoder` map be `skipped` rather than aborting the run.

## 7. Measured results

All **13** runnable methods extracted successfully (`N = 50000` each). Per-method
numbers below are from this run's `manifest.json` and each `meta.json`. Every row
was load-verified in a separate pass: `features.npy` is `(50000, D) float32`, no
NaN/Inf, unit L2 norm (mean 1.0000), 1000 classes × exactly 50, labels
non-decreasing.

| method | `feat_dim` | arch | L2 mean | `encoder_sha256` (first 12) | batch |
|---|---|---|---|---|---|
| `28_dinov2` | 1536 | dinov2_vitg14 | 1.0000 | `baf8467e50af` | 64 |
| `36_franca` | 768 | franca_vitb14 | 1.0000 | `2a0888b5bd5b` | 64 |
| `aimv2` | 1024 | aimv2_large_patch14_224.apple_pt | 1.0000 | `37861b19be35` | 64 |
| `beitv2` | 768 | beitv2_base_patch16_224 | 1.0000 | `6d32a5f76264` | 64 |
| `cae` | 768 | cae_vit-base-p16 (OpenMMLab mmselfsup reproduction) | 1.0000 | `808170f33cb1` | 64 |
| `cosmos3_super` | 1152 | nvidia/Cosmos3-Super | 1.0000 | `3bdba32cec6f` | 64 |
| `data2vec2` | 768 | facebook/data2vec-vision-base | 1.0000 | `e1942d45a500` | 64 |
| `eva02` | 768 | eva02_base_patch14_224 | 1.0000 | `65068e4594e9` | 64 |
| `siglip` | 768 | vit_base_patch16_siglip_224.webli | 1.0000 | `cdcf73d937a9` | 64 |
| `var` | 32 | var_vqvae | 1.0000 | `7c3ec27ae28a` | 64 |
| `videomae` | 768 | MCG-NJU/videomae-base | 1.0000 | `bc053ca2840a` | 8 |
| `vjepa2` | 1024 | facebook/vjepa2-vitl-fpc64-256 | 1.0000 | `25466aef8572` | 8 |
| `vjepa2_ac` | 1408 | vit_giant_xformers | 1.0000 | `0b5e3c4bf77a` | 64 |

**Batch size.** `videomae` and `vjepa2` OOM'd at `--batch-size 64` on the 15 GB
T4 (they expand a still image into temporal frames, so activation memory is much
larger than a plain ViT — note `vjepa2_ac`, a ViT-g *giant*, succeeded at 64).
They were re-run at `--batch-size 8` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and then verified identically.
Features are independent of batch size; the column is recorded only so the run is
exactly reproducible.

`encoder_sha256` for each method is recorded in its `meta.json` and matches the
provenance-pinned artifact hash.

## Audited reference extraction (2026-09-21)

The Figure 2 expansion is a separate collection of profiles. The historical
47-method accounting above does not establish that these additional profiles
have been extracted or delivered. A workbook row, a checkpoint candidate, a
successful worker and a verified shared-directory delivery are distinct states.
Current profile mappings, job IDs, source/result evidence and delivery hashes
are held in private execution state outside Git.

`bin/extract-reference.py` supports an audited train-then-validation evaluator
and the captured shared linear-probe driver. It reuses the reference's model
construction, checkpoint loader, validation transforms and feature function.
The wrapper bypasses training features and terminates before probe training;
a skipped training array is only a shape placeholder for reference logging.
This is not a new implementation of each model or a reproduction of probe scores.
Only evaluators whose control flow has been inspected are supported.

A trusted private JSON profile supplies `id`, `source`, `source_sha256`,
`checkpoint`, `data_root`, `count` and reference `argv`. The optional
`driver: "shared_probe"` uses the reference's configured AMP precision and
relocates its ImageNet-1k paths without changing original files. Directly imported
and module-qualified probe calls are supported, including positional arguments
bound against the actual reference signature. Device selection follows the
shared probe, including backbones that lazily initialize their parameters.
For that driver,
`distributed: true` permits a launcher-provided rank/world configuration:
reference extraction gathers features, exact sampler indices restore dataset
order, and rank zero alone writes. Duplicate or incomplete gathered indices
are errors; sorting labels is not an acceptable substitute.

Run each profile in its own interpreter and read-only sandbox, with a new output
directory. The source hash is checked before import. Reference dependencies,
checkpoint identity, loading diagnostics and scientific agreement still require
separate inspection; the entry-point hash alone does not certify them.
For example, using an already prepared private profile:

```sh
python bin/extract-reference.py --profile "$PRIVATE_PROFILE" --out "$NEW_OUTPUT"
```

The worker validates count, dimensions, dataset label order, finite values and
nonzero vectors, then reuses the established L2 normalization and file writer.
It records checkpoint/source hashes, preprocessing and sample-order provenance.
A separate delivery check must verify 50,000 float32 unit vectors, 1,000 sorted
classes with 50 int64 labels each, metadata consistency and copied file hashes.
Existing profiles must not be overwritten. Reference/profile details are private
and must not be committed with the public extraction tool.

The user authorized up to eight reservation nodes in total for this expansion.
Count existing jobs before submitting replacements; preserve unrelated work.
Reservation-only execution, original read-only inputs, strict testing and PR
review remain required. Ambiguous checkpoint/readout mappings and references
that substitute another backbone or incompletely load weights remain pending.
