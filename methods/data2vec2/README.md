# data2vec2 — as-is linear eval on the frozen pretrained backbone (eval-only)

data2vec 2.0 (Baevski, Babu, Hsu & Auli, *Efficient Self-supervised Learning with
Contextualized Target Representations for Vision, Speech and Language*, 2023;
[arXiv:2212.07525](https://arxiv.org/abs/2212.07525)), a self-distillation method
that predicts the top-K averaged layer representations of an EMA teacher (not
pixels) under block-wise masking, across vision, speech and language.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. data2vec's "as-is" comparison freezes the official
pretrained vision backbone (`facebook/data2vec-vision-base`) and fits a linear
probe on its CLS token, because the from-scratch self-distillation pretraining
(IN-1k, many GPU-days) is the excluded step. The probed representation is a genuine
SSL feature, so the number **is** comparable — the "pretrained-backbone reuse" row,
analogous to `eva02` / `36_franca`.

## The checkpoint, stated plainly

The capture's data2vec 2.0 comparison (`docs/data2vec.md`) names the HuggingFace
checkpoint `facebook/data2vec-vision-base` and loads it with `transformers`'
`Data2VecVisionModel`, CLS token. That checkpoint is Meta AI's released data2vec
**vision** backbone (a BEiT-architecture ViT-B/16, IN-1k linear ~64.5%); its
HuggingFace model card cites `arXiv:2202.03555` (the original data2vec) and
`arXiv:2106.08254` (BEiT). This port pins that exact checkpoint, as the capture
doc specifies, and takes its paper axis from the doc (data2vec 2.0). All of this
is recorded in `provenance.json`.

## Why this method, and what is new here

**data2vec2 is the first `transformers`-sourced eval-only port.** Its only stage
is `linear_eval`. Unlike the timm-sourced frozen-backbone siblings (`eva02` /
`aimv2`) or the submodule-sourced `beitv2`, the model class here is
`transformers`' `Data2VecVisionModel` — a pinned pip dependency **new to the
fleet**, not a git submodule.

**There is no author submodule.** The model class is transformers' — imported and
never copied — and the pretrained weights are a **sha256-pinned download** recorded
as `backbone_artifact` in `provenance.json`. So this port records **no** upstream
block and defines **no** `UPSTREAM` in the adapter (they must agree, and here both
are absent). The lock closes over transformers' own dependency tree
(huggingface-hub, tokenizers, safetensors, regex, …).

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds a `Data2VecVisionModel` from the config's architecture
keys, loads the downloaded checkpoint into it frozen, and fits a single linear
layer on the CLS feature. The manifest therefore carries `encoder_absent_reason`
rather than an encoder, and the backbone it read is named in the config
(`train.ckpt`).

Changed during the port (see `provenance.json`): the device is **resolved**
(`resolve_device`) rather than assumed CUDA; features are extracted in **fp32** (no
autocast), so the frozen-feature probe runs identically on a CPU or a pre-Ampere
GPU; the model is built with `add_pooling_layer=False` and the **CLS token** of
`last_hidden_state` is probed (the checkpoint is a pretraining model with
`use_mean_pooling=False` and no trained pooler); the checkpoint loads with
`strict=False` because it carries a derived `relative_position_index` buffer
transformers rebuilds; the input normalisation follows the backbone's **own
preprocessor** config (symmetric mean/std 0.5, a bicubic square resize, no centre
crop); the probe follows this port's shared frozen-backbone protocol (mean-centre +
L2-normalise, one linear layer with SGD + cosine).

## The representation, and the caveat

The probe reads data2vec-vision's CLS token, frozen. A real number therefore
measures the **pretrained** backbone, not something this port trained. The official
checkpoint is a **download pinned by sha256** in `provenance.json`, fetched and
hash-verified by `bin/fetch-weights.py`. The hermetic smoke leaves `train.ckpt`
empty and builds a **random tiny** Data2VecVisionModel at a small resolution, so
nothing is downloaded and its accuracy is meaningless — only the pipeline is
exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit.
- **Not a full run:** `configs/linear_eval.yaml` pins the official
  data2vec-vision-base recipe, not a completed run.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and
  runs where a device is visible.

## Environment

The eval stack is torch / torchvision / transformers / numpy / PyYAML; transformers
supplies the model class and its config, and its own dependency closure
(huggingface-hub, tokenizers, safetensors, regex, …) is pinned in the lock. There
is **no submodule** to check out.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/data2vec2/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/data2vec2/provenance.json \
        --out .weights/data2vec2 --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/data2vec2/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set D2V_CKPT=.weights/data2vec2/data2vec-vision-base.pytorch_model.bin \
        --out resolved.json
    cd methods/data2vec2 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
