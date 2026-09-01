# sam3 — as-is linear eval on the frozen SAM 3 vision encoder (eval-only)

Meta **SAM 3** (*Segment Anything with Concepts*, Meta AI, 2025;
[ai.meta.com/research/sam3](https://ai.meta.com/research/sam3/);
[github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)), a
promptable segmentation foundation model whose vision encoder is a ViTDet-style
ViT with 2D rotary position embeddings and windowed attention.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. SAM 3's as-is comparison freezes the official
promptable-segmentation model's vision encoder and fits a linear probe on its
**mean-pooled patch tokens** (no text/box prompts), because SAM 3's from-scratch
training is the excluded step. The probed representation is a genuine learned
feature, so the number **is** comparable — the multimodal "pretrained-backbone
reuse" row, the `transformers`-sourced sibling of `data2vec2`.

## The checkpoint, stated plainly

The capture's Step-3 CompEval comparison (`methods_step3/SegFM/SAM3`,
`methods_compeval/adapters/sam3_adapter.py`) loads the official `facebook/sam3`
checkpoint (`sam3.pt`, revision `3c879f3…`) and reads its vision encoder through
`transformers`' `Sam3ViTModel`. The official file stores its trunk under
`detector.backbone.vision_backbone.trunk.*` (ViTDet style: fused `attn.qkv`, a CLS
row in `pos_embed`, a bias on the patch-embed conv), which does **not** match the
HF class — a plain load would leave the backbone randomly initialised. This port
converts those tensors onto `Sam3ViTModel` (`sam3_trunk.py`) and infers the
architecture from the checkpoint itself (hidden 1024, depth 32, and the non-4×
`intermediate_size` 4736), so the released ViT-L loads without a size mismatch.
RoPE tables are non-persistent buffers `transformers` rebuilds and are not copied.
All of this is recorded in `provenance.json`.

The weights are Meta's, **gated** (the Meta SAM License). Nothing under it is
copied here; the model constructor is imported from the pinned `transformers`
dependency, and the checkpoint is a sha256-pinned file you must already be entitled
to (`bin/fetch-weights.py` verifies the hash of a copy you supply).

## Why this method, and what is new here

**sam3 is a multimodal, `transformers`-sourced eval-only port**, the sibling of
`data2vec2`. Its only stage is `linear_eval`. The model class is `transformers`'
`Sam3ViTModel` — a pinned pip dependency (`transformers==5.16.1`, the fleet pin
that first ships the `sam3` module), not a git submodule. What is new relative to
`data2vec2`: the official checkpoint is not directly loadable by the HF class, so
the port carries a **trunk converter** (`sam3_trunk.py`), unit-tested on synthetic
tensors so the real-run path is covered without the gated weights.

**There is no author submodule.** The model class is transformers' — imported and
never copied — and the pretrained weights are a **sha256-pinned, gated download**
recorded as `backbone_artifact` in `provenance.json`. So this port records **no**
upstream block and defines **no** `UPSTREAM` in the adapter (they must agree, and
here both are absent). The lock closes over transformers' own dependency tree
(huggingface-hub, tokenizers, safetensors, regex, …).

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds a `Sam3ViTModel`, loads the converted checkpoint into it
frozen, and fits a single linear layer on the mean-pooled patch feature. The
manifest therefore carries `encoder_absent_reason` rather than an encoder, and the
backbone it read is named in the config (`train.ckpt`).

Changed during the port (see `provenance.json`): the device is **resolved**
(`resolve_device`) rather than assumed CUDA; features are extracted in **fp32** (no
autocast / no bf16), so the frozen-feature probe runs identically on a CPU or a
pre-Ampere GPU, rather than the checkpoint's bf16; the feature is the vision
encoder's patch tokens **mean-pooled** over the sequence (SAM 3's ViT has no CLS
token), so `feature_dim = hidden_size`; the input resolution is **336**
(`pretrain_image_size`), not the native 1008, so the probe is executable; the input
normalisation is ImageNet's (the capture's SAM3 adapter); the probe follows this
port's shared frozen-backbone protocol (mean-centre + L2-normalise, one linear
layer with SGD + cosine).

## The representation, and the caveat

The probe reads SAM 3's mean-pooled patch tokens, frozen. A real number therefore
measures the **pretrained** backbone, not something this port trained. The official
checkpoint is a **gated download pinned by sha256** in `provenance.json`, whose
hash `bin/fetch-weights.py` verifies against a copy you supply. The hermetic smoke
leaves `train.ckpt` empty and builds a **random tiny** `Sam3ViTModel` at a small
resolution, so nothing is downloaded and its accuracy is meaningless — only the
pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit. The trunk
  converter is unit-tested on synthetic official-format tensors: the conversion
  loads into a `Sam3ViTModel` with **no** backbone weight left missing and a finite
  forward, and a checkpoint with no trunk keys is refused.
- **Not a full run:** `configs/linear_eval.yaml` pins the official ViT-L SAM 3
  recipe, not a completed run; the real checkpoint is gated and was not downloaded
  during the port.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and
  runs where a device is visible.

## Environment

The eval stack is torch / torchvision / transformers / safetensors / numpy /
PyYAML; transformers supplies the model class and its config, and its own
dependency closure (huggingface-hub, tokenizers, regex, …) is pinned in the lock.
There is **no submodule** to check out.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/sam3/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # verify a copy of the gated official checkpoint (pinned by sha256 in
    # provenance.json); the file is not auto-downloaded -- supply your own,
    # entitled under the Meta SAM License
    python bin/fetch-weights.py --provenance methods/sam3/provenance.json \
        --out .weights/sam3 --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/sam3/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set SAM3_CKPT=.weights/sam3/sam3.pt \
        --out resolved.json
    cd methods/sam3 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
