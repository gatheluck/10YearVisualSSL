# vjepa2_ac — as-is linear eval on the frozen pretrained backbone (eval-only)

V-JEPA 2-AC (Assran et al., Meta FAIR, *V-JEPA 2: Self-Supervised Video Models
Enable Understanding, Prediction and Planning*, 2025;
[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)) — the **action-conditioned**
world model: V-JEPA 2's latent-prediction video encoder, post-trained on
robot-manipulation video conditioned on actions. This port probes that
pretrained encoder.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. The "as-is" comparison freezes the pretrained video encoder
(a ViT-g/16 with a Conv3d tubelet patch embed) and fits a linear probe on its
**mean-pooled patch tokens**, because the latent-prediction pretraining
(internet-scale video, then action-conditioned post-training, many GPU-days) is
the excluded step. The probed representation is a genuine SSL feature, so the
number **is** comparable — the "pretrained-backbone reuse" row, analogous to
`eva02` / `aimv2` / `data2vec2` / `cae` / `videomae` / `vjepa2`.

## The one thing that differs from the sibling `vjepa2`: the real rotary forward

There are two V-JEPA 2 ports in this repository, and the difference is
deliberate:

- **`vjepa2`** (the action-*free* V-JEPA 2 ViT-L) reimplements a **small
  self-contained** Conv3d-patch-embed ViT and runs it with **plain (non-rotary)
  attention** — an approximation, stated plainly in that method's README.
- **`vjepa2_ac`** (this port) builds the ViT from the **pinned
  facebookresearch/vjepa2 submodule** (`third_party/vjepa2`, imported not
  copied) and runs V-JEPA 2's **real rotary-position attention**
  (`use_rope=True`), reproducing the capture's number rather than approximating
  it.

Because the ViT is an author submodule, this adapter records an `UPSTREAM` block
and `provenance.json` records an `upstream` (they must agree —
`tests/test_port_completeness.py` checks the both-places rule).

## The representation, on a still image

V-JEPA 2 consumes a video clip `(B, 3, T, H, W)`. The capture's ImageNet linear
eval feeds a **still image replicated `tubelet_size` times** along the temporal
axis (never PyAV / a video dataset) so the Conv3d tubelet embed yields a single
temporal token, runs the ViT, and mean-pools the tokens. **This port keeps
exactly that:** `extract_feature` expands each image to a clip, runs the encoder
with the real rotary forward, and returns one mean-pooled vector per image. The
`num_frames` / `tubelet_size` the clip is built with are read from the config, so
the hermetic smoke can shrink the model.

## The checkpoint, stated plainly

The weights are the authors' **official public** `vjepa2-ac-vitg.pt` checkpoint
(a training-state dict), pinned by sha256 in `provenance.json` and fetched by
`bin/fetch-weights.py`. They are the paper authors' released weights — **not** a
reproduction — distributed under **MIT** (measured from the pinned submodule's
`LICENSE`). The capture's own backbone wrapper carries a stale `CC BY-NC 4.0`
comment inherited from V-JEPA 1; `provenance.json` records the **measured** MIT
licence and the correction plainly. Nothing under it is copied into this
repository.

The `.pt` is a training-state dict; its `encoder.*` sub-dict (each key prefixed
`module.`) loads into the submodule's `vit_giant_xformers` after stripping
`module.`/`backbone.`, and the `predictor` (the action-conditioned JEPA
predictor), `opt`, `scaler`, `target_encoder` and scalar entries are dropped. The
load is **measured exact** — 484 encoder tensors, set-equal to the model's keys,
zero missing, zero unexpected — so this port tightens the capture's tolerant
`load_state_dict(strict=False)` (which merely printed the counts) to a **hard
error** on any missing or unexpected key.

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds the V-JEPA 2 ViT from the config's `arch` factory, loads
the downloaded checkpoint's `encoder.*` tensors into it frozen, and fits a single
linear layer on the mean-pooled patch feature. The manifest therefore carries
`encoder_absent_reason` rather than an encoder, and the backbone it read is named
in the config (`train.ckpt`).

Changed during the port (see `provenance.json`): the ViT is the pinned
facebookresearch/vjepa2 submodule (imported via `src.models.vision_transformer`,
never copied), with the `src`/`app` imports lazy and the path prepared (`src*` /
`app*` purged from `sys.modules`, other `third_party` roots stripped,
`third_party/vjepa2` inserted first) so the two submodule ports (this one and
`35_vjepa`) do not collide in the in-process test suite; the forward runs the
**real rotary attention** (`use_rope=True`); the checkpoint is read with
`torch.load` (a `.pt` training-state dict), the `encoder` sub-dict loaded and the
predictor/opt/scaler/target_encoder dropped, any missing/unexpected key a hard
error; the device is **resolved** (`resolve_device`) rather than assumed CUDA;
features are extracted in **fp32** (no autocast); the input normalisation is
ImageNet mean/std with a bilinear square resize; the probe follows this port's
shared frozen-backbone protocol (mean-centre + L2-normalise, one linear layer
with SGD + cosine).

## The representation, and the caveat

The probe reads the V-JEPA 2 ViT's mean-pooled patch tokens, frozen. A real
number therefore measures the **pretrained** backbone, not something this port
trained, and it is an **image** proxy of a **video** model (a still image
replicated across the temporal axis). Unlike the sibling `vjepa2`, it is run with
the **real rotary attention**, so it reproduces the capture's number rather than
approximating it. The checkpoint is a **download pinned by sha256** in
`provenance.json`, fetched and hash-verified by `bin/fetch-weights.py`. The
hermetic smoke leaves `train.ckpt` empty and builds a **random tiny** V-JEPA 2
ViT (`vit_tiny`, still `use_rope=True`) at a small resolution, so nothing is
downloaded and its accuracy is meaningless — only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit. A
  well-formed `encoder.*` checkpoint (carrying a decoy `predictor.*` key that must
  be dropped) is loaded and probed, and a checkpoint missing an encoder weight is
  refused (both without the 11 GB download). The forward is asserted to use rope.
- **Not a full run:** `configs/linear_eval.yaml` pins the official
  `vjepa2-ac-vitg.pt` ViT-g recipe, not a completed run.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and
  runs where a device is visible.

## Environment

The eval stack is torch / torchvision / numpy / PyYAML for this port's own files;
the submodule additionally loads **einops** and **timm** at import, so both are
pinned in the lock (lock-only — this port's files do not import them). The V-JEPA
2 ViT is an author **submodule** that must be checked out:

    git submodule update --init third_party/vjepa2

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/vjepa2_ac/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official V-JEPA 2-AC backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/vjepa2_ac/provenance.json \
        --out .weights/vjepa2_ac --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/vjepa2_ac/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set VJEPA2_AC_CKPT=.weights/vjepa2_ac/vjepa2-ac-vitg.pt \
        --out resolved.json
    cd methods/vjepa2_ac && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
