# aimv2 — as-is linear eval on the frozen pretrained backbone (eval-only)

AIMv2 (Fini, Béthune, Yang, Zhai, Susskind, El-Nouby et al., *Multimodal
Autoregressive Pre-training of Large Vision Encoders*, 2024;
[arXiv:2411.14402](https://arxiv.org/abs/2411.14402)), a vision encoder pretrained
by a multimodal autoregressive objective — an image encoder feeding a decoder that
predicts image patches and text tokens together — built as a `VisionTransformer`
with RMSNorm, a SwiGLU MLP and average pooling over patch tokens (no class token).

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. AIMv2's "as-is" comparison freezes the official pretrained
backbone (`aimv2_large_patch14_224.apple_pt`) and fits a linear probe on its pooled
features, because the from-scratch pretraining (billions of image-text pairs, many
GPU-days) is the excluded step. The probed representation is a genuine learned
feature, so the number **is** comparable — the "pretrained-backbone reuse" row,
analogous to `eva02` / `36_franca` / `30_aim`.

## Why this method, and what is new here

**aimv2 is the second pure eval-only port** (after `eva02`) — its only stage is
`linear_eval`, with no `pretrain` at all. It is a new **unnumbered** port under the
naming decision that new methods take a bare name (`methods/aimv2`, like `eva02` /
`image_gpt` / `mar` / `var`) rather than a capture number.

**There is no author submodule.** The AIMv2 model class is timm's — a pinned pip
dependency (`timm==1.0.28`), imported and never copied — and the pretrained weights
are a **sha256-pinned download** recorded as `backbone_artifact` in
`provenance.json`. So this port records **no** upstream block and defines **no**
`UPSTREAM` in the adapter (they must agree, and here both are absent). The lock
closes over timm's own dependency tree (huggingface-hub, safetensors, tqdm, …).

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds the named timm architecture, loads the downloaded
checkpoint into it frozen, and fits a single linear layer on the pooled feature.
The manifest therefore carries `encoder_absent_reason` rather than an encoder, and
the backbone it read is named in the config (`train.ckpt`).

Changed during the port (see `provenance.json`): the device is **resolved**
(`resolve_device`) rather than assumed CUDA; features are extracted in **fp32** (no
autocast), so the frozen-feature probe runs identically on a CPU or a pre-Ampere
GPU; the input normalisation is taken from the backbone's **own timm data config**
(AIMv2 uses a CLIP-style mean/std, not ImageNet's); the probe follows this port's
shared frozen-backbone protocol (mean-centre + L2-normalise, one linear layer with
SGD + cosine).

## The representation, and the caveat

The probe reads AIMv2's pooled pre-classifier feature (`num_classes=0`), frozen. A
real number therefore measures AIMv2's **pretrained** backbone, not something this
port trained. The official checkpoint is a **download pinned by sha256** in
`provenance.json`, fetched and hash-verified by `bin/fetch-weights.py`. The
hermetic smoke leaves `train.ckpt` empty and builds a **random tiny** AIMv2-style
`VisionTransformer` (RMSNorm, SwiGLU MLP, no class token) at a small resolution, so
nothing is downloaded and its accuracy is meaningless — only the pipeline is
exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit.
- **Not a full run:** `configs/linear_eval.yaml` pins the official AIMv2-Large/14
  apple_pt recipe, not a completed run.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and
  runs where a device is visible.

## Environment

The eval stack is torch / torchvision / timm / numpy / PyYAML; timm supplies the
model class and its data config, and its own dependency closure (huggingface-hub,
safetensors, tqdm, …) is pinned in the lock. There is **no submodule** to check
out.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/aimv2/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/aimv2/provenance.json \
        --out .weights/aimv2 --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/aimv2/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set AIMV2_CKPT=.weights/aimv2/aimv2_large_patch14_224.apple_pt.pytorch_model.bin \
        --out resolved.json
    cd methods/aimv2 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
