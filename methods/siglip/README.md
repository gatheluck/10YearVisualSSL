# siglip — as-is linear eval on the frozen pretrained image tower (eval-only)

SigLIP (Zhai, Mustafa, Kolesnikov & Beyer, *Sigmoid Loss for Language Image
Pre-Training*, 2023; [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)), an
image-text model pretrained with a pairwise **sigmoid** loss (rather than the
softmax contrastive loss of CLIP), whose image tower is a `VisionTransformer` with
no class token and a MAP attention-pooling head.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. SigLIP's "as-is" comparison freezes the official pretrained
image tower (`vit_base_patch16_siglip_224.webli`) and fits a linear probe on its
pooled image embedding, because the from-scratch pretraining (the WebLI image-text
corpus, many TPU-days) is the excluded step. The probed representation is a genuine
learned feature, so the number **is** comparable — the multimodal
"pretrained-backbone reuse" row, the SigLIP sibling of `38_clip`.

## Why this method, and what is new here

**siglip is the third pure eval-only Step-3 port** (after `eva02` and `aimv2`), and
the first from the **CompEval / multimodal** family: it probes an image-text
backbone rather than a self-supervised one, the same as-is shape `38_clip` uses for
its Step-1 row (`docs/EVAL_DOWNLOAD.md`). It is a new **unnumbered** port under the
naming decision that new methods take a bare name (`methods/siglip`, like `eva02` /
`aimv2` / `image_gpt` / `mar` / `var`) rather than a capture number.

**There is no author submodule.** The SigLIP model class is timm's — a pinned pip
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
(SigLIP uses a symmetric 0.5 mean/std, not ImageNet's); the probe follows this
port's shared frozen-backbone protocol (mean-centre + L2-normalise, one linear
layer with SGD + cosine).

## The representation, and the caveat

The probe reads SigLIP's pooled pre-classifier image embedding (`num_classes=0`),
frozen. A real number therefore measures SigLIP's **pretrained** image tower, not
something this port trained. The official checkpoint is a **download pinned by
sha256** in `provenance.json`, fetched and hash-verified by `bin/fetch-weights.py`.
The hermetic smoke leaves `train.ckpt` empty and builds a **random tiny**
SigLIP-style `VisionTransformer` (no class token, a MAP attention-pool head) at a
small resolution, so nothing is downloaded and its accuracy is meaningless — only
the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit.
- **Not a full run:** `configs/linear_eval.yaml` pins the official ViT-B/16 SigLIP
  WebLI recipe, not a completed run.
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
        -r methods/siglip/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/siglip/provenance.json \
        --out .weights/siglip --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/siglip/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set SIGLIP_CKPT=.weights/siglip/vit_base_patch16_siglip_224.webli.pytorch_model.bin \
        --out resolved.json
    cd methods/siglip && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
