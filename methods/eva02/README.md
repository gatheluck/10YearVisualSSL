# eva02 — as-is linear eval on the frozen pretrained backbone (eval-only)

EVA-02 (Fang, Sun, Wang, Huang, Wang & Cao, *EVA-02: A Visual Representation for
Neon Genesis*, 2023; [arXiv:2303.11331](https://arxiv.org/abs/2303.11331)), a
masked-image-modelling ViT that regresses the features of a strong (CLIP) teacher
and rebuilds the transformer with SwiGLU, sub-LN and 2D RoPE.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. EVA-02's "as-is" comparison freezes the official
masked-image-modelling pretrained backbone (`eva02_base_patch14_224.mim_in22k`)
and fits a linear probe on its pooled features, because the from-scratch MIM
pretraining (IN-22k, many GPU-days) is the excluded step. The probed
representation is a genuine SSL feature, so the number **is** comparable — the
"pretrained-backbone reuse" row, analogous to `36_franca` / `30_aim`.

## Why this method, and what is new here

**eva02 is the first pure eval-only port** — its only stage is `linear_eval`,
with no `pretrain` at all (`mar` is pretrain-only; the download-and-probe siblings
`36_franca` / `38_clip` carry a Step-2 pretrain alongside the probe). It is also
the **first unnumbered new port** under the naming decision that new methods take
a bare name (`methods/eva02`, like `image_gpt` / `mar` / `var`) rather than a
capture number.

**There is no author submodule.** The EVA-02 model class is timm's — a pinned pip
dependency (`timm==1.0.28`), imported and never copied — and the pretrained
weights are a **sha256-pinned download** recorded as `backbone_artifact` in
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
(EVA-02 uses a CLIP-style mean/std, not ImageNet's); the probe follows this port's
shared frozen-backbone protocol (mean-centre + L2-normalise, one linear layer with
SGD + cosine).

## The representation, and the caveat

The probe reads EVA-02's pooled pre-classifier feature (`num_classes=0`), frozen.
A real number therefore measures EVA-02's **pretrained** backbone, not something
this port trained. The official checkpoint is a **download pinned by sha256** in
`provenance.json`, fetched and hash-verified by `bin/fetch-weights.py`. The
hermetic smoke leaves `train.ckpt` empty and builds a **random tiny** EVA-02
(`timm.models.eva.Eva`) at a small resolution, so nothing is downloaded and its
accuracy is meaningless — only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit.
- **Not a full run:** `configs/linear_eval.yaml` pins the official EVA-02-B/14
  MIM-IN22k recipe, not a completed run.
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
        -r methods/eva02/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/eva02/provenance.json \
        --out .weights/eva02 --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/eva02/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set EVA02_CKPT=.weights/eva02/eva02_base_patch14_224.mim_in22k.pytorch_model.bin \
        --out resolved.json
    cd methods/eva02 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
