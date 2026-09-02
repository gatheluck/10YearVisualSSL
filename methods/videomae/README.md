# videomae — as-is linear eval on the frozen pretrained backbone (eval-only)

VideoMAE (Tong, Song, Wang & Wang, *VideoMAE: Masked Autoencoders are
Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022;
[arXiv:2203.12602](https://arxiv.org/abs/2203.12602)), a masked-video method that
reconstructs masked spatiotemporal tubes with an extremely high masking ratio,
learning a transferable video representation with a plain ViT and a Conv3d
tubelet embedding.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. VideoMAE's "as-is" comparison freezes the pretrained video
encoder (a ViT-B/16 with a tubelet Conv3d patch embed) and fits a linear probe on
its **temporally-averaged patch tokens**, because the from-scratch masked-video
pretraining (Kinetics / SSv2, many GPU-days) is the excluded step. The probed
representation is a genuine SSL feature, so the number **is** comparable — the
"pretrained-backbone reuse" row, analogous to `eva02` / `aimv2` / `data2vec2` /
`cae`.

## The representation, on a still image

VideoMAE consumes a video clip `(B, 3, T, H, W)`. The capture's ImageNet linear
eval feeds a **still image replicated `num_frames` times** along the temporal
axis (never PyAV / a video dataset), runs the ViT, and mean-pools the tokens.
**This port keeps exactly that:** `extract_feature` expands each image to a clip,
runs the encoder, and returns one temporally-averaged vector per image. The
`num_frames` / `tubelet_size` the clip is built with are read from the config, so
the hermetic smoke can shrink them.

## The checkpoint, stated plainly

The weights are the authors' **official public** `MCG-NJU/videomae-base`
checkpoint (a `VideoMAEForPreTraining` safetensors), pinned by sha256 in
`provenance.json` and fetched by `bin/fetch-weights.py`. They are the paper
authors' released weights — **not** a reproduction — distributed under
**CC-BY-NC-4.0** (non-commercial), like this repository's other non-commercial
ports. `provenance.json` records the licence plainly; nothing under it is copied
into this repository.

## Why this method, and what is new here

Unlike the `timm`-sourced siblings (`eva02` / `aimv2`) or the
`transformers`-sourced `data2vec2`, **neither timm nor transformers carries a
VideoMAE model class** here. So the encoder is a **small self-contained
Conv3d-patch-embed VideoMAE ViT** in `evaluate_linear_videomae.py`, built from the
config's architecture keys, whose module names mirror the checkpoint's
`videomae.*` key hierarchy. The checkpoint's `videomae.*` encoder tensors load
into it directly (the `videomae.` prefix stripped) and the `decoder.*`
pretext-decoder keys are dropped.

**No position-embedding parameters.** VideoMAE uses fixed sin-cos position
embeddings computed at run time (held as a non-persistent buffer), so the official
checkpoint stores none and the self-contained ViT omits them too — the stripped
state therefore loads with **no missing and no unexpected** key.

**There is no author submodule and no model-carrying pip dependency.** So this
port records **no** upstream block and defines **no** `UPSTREAM` in the adapter
(they must agree, and here both are absent). The eval stack is just torch /
torchvision / safetensors / numpy / PyYAML.

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds the VideoMAE ViT from the config's architecture keys,
loads the downloaded checkpoint's `videomae.*` tensors into it frozen, and fits a
single linear layer on the temporally-averaged patch feature. The manifest
therefore carries `encoder_absent_reason` rather than an encoder, and the backbone
it read is named in the config (`train.ckpt`).

Changed during the port (see `provenance.json`): the encoder is a self-contained
Conv3d-patch-embed ViT whose module names mirror the checkpoint's `videomae.*`
keys, so the tensors load with no remapping; the checkpoint is read directly from
safetensors (the `videomae.` prefix stripped, the `decoder.*` keys dropped); any
**missing or unexpected** encoder key on load is a hard error (a half-loaded
backbone would silently misreport what ran); the capture's **silent fallback to
random weights on a failed download is removed** (a `ckpt` that is not a loadable
VideoMAE checkpoint raises); the device is **resolved** (`resolve_device`) rather
than assumed CUDA; features are extracted in **fp32** (no autocast); the input
normalisation follows the backbone's **own preprocessor** config (ImageNet
mean/std, a bicubic square resize, no centre crop); the probe follows this port's
shared frozen-backbone protocol (mean-centre + L2-normalise, one linear layer with
SGD + cosine).

## The representation, and the caveat

The probe reads the VideoMAE ViT's temporally-averaged patch tokens, frozen. A
real number therefore measures the **pretrained** backbone, not something this
port trained, and it is an **image** proxy of a **video** model (a still image
replicated across the temporal axis). The checkpoint is a **download pinned by
sha256** in `provenance.json`, fetched and hash-verified by `bin/fetch-weights.py`.
The hermetic smoke leaves `train.ckpt` empty and builds a **random tiny** VideoMAE
ViT at a small resolution, so nothing is downloaded and its accuracy is
meaningless — only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit. A well-formed
  `videomae.*` checkpoint is loaded and probed, and a checkpoint missing an encoder
  weight is refused (both without the 360 MB download).
- **Not a full run:** `configs/linear_eval.yaml` pins the official
  MCG-NJU/videomae-base ViT-B recipe, not a completed run.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and
  runs where a device is visible.

## Environment

The eval stack is torch / torchvision / safetensors / numpy / PyYAML — no
transformers, no timm. There is **no submodule** to check out.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/videomae/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official VideoMAE backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/videomae/provenance.json \
        --out .weights/videomae --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/videomae/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set VIDEOMAE_CKPT=.weights/videomae/model.safetensors \
        --out resolved.json
    cd methods/videomae && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
