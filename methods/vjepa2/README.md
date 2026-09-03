# vjepa2 — as-is linear eval on the frozen pretrained backbone (eval-only)

V-JEPA 2 (Assran et al., Meta FAIR, *V-JEPA 2: Self-Supervised Video Models Enable
Understanding, Prediction and Planning*, 2025;
[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)), a latent-prediction video
method that extends V-JEPA's masked joint-embedding objective to internet-scale
video, learning a transferable video representation with a plain ViT and a Conv3d
tubelet embedding.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. V-JEPA 2's "as-is" comparison freezes the pretrained video
encoder (a ViT-L/16 with a tubelet Conv3d patch embed) and fits a linear probe on
its **mean-pooled patch tokens**, because the from-scratch latent-prediction
pretraining (internet-scale video, many GPU-days) is the excluded step. The probed
representation is a genuine SSL feature, so the number **is** comparable — the
"pretrained-backbone reuse" row, analogous to `eva02` / `aimv2` / `data2vec2` /
`cae` / `videomae`.

## The representation, on a still image

V-JEPA 2 consumes a video clip `(B, 3, T, H, W)`. The capture's ImageNet linear
eval feeds a **still image replicated `num_frames` times** along the temporal axis
(never PyAV / a video dataset), runs the ViT, and mean-pools the tokens. **This
port keeps exactly that:** `extract_feature` expands each image to a clip, runs the
encoder, and returns one mean-pooled vector per image. The `num_frames` /
`tubelet_size` the clip is built with are read from the config, so the hermetic
smoke can shrink them.

## The checkpoint, stated plainly

The weights are the authors' **official public** `facebook/vjepa2-vitl-fpc64-256`
checkpoint (a `VJEPA2Model` safetensors), pinned by sha256 in `provenance.json` and
fetched by `bin/fetch-weights.py`. They are the paper authors' released weights —
**not** a reproduction — distributed under **MIT** (measured from the HuggingFace
model-card front-matter; unlike V-JEPA 1 / `videomae`, which are CC-BY-NC). The
capture's own backbone wrapper carries a stale `CC BY-NC 4.0` comment inherited
from V-JEPA 1; `provenance.json` records the **measured** MIT licence and the
correction plainly. Nothing under it is copied into this repository.

## Why this method, and what is new here

Unlike the `timm`-sourced siblings (`eva02` / `aimv2`) or the
`transformers`-sourced `data2vec2`, **neither timm nor transformers carries this
V-JEPA 2 model class** here. So the encoder is a **small self-contained
Conv3d-patch-embed V-JEPA 2 ViT** in `evaluate_linear_vjepa2.py`, built from the
config's architecture keys, whose module names mirror the checkpoint's `encoder.*`
key hierarchy. The checkpoint's `encoder.*` tensors load into it directly (the
`encoder.` prefix stripped) and the `predictor.*` JEPA-predictor keys are dropped.

**No position-embedding parameters.** V-JEPA 2 applies rotary position embeddings at
run time and stores no learned position parameters, so the official checkpoint
carries none and the self-contained ViT omits them too — the stripped state loads
with **no missing and no unexpected** key.

**A faithfulness caveat: plain attention.** The capture's backbone wrapper runs the
encoder with **plain (non-rotary) attention** — it loads the weights and mean-pools
without re-deriving V-JEPA 2's rotary mechanism. This port mirrors the capture's
forward exactly, so the probe number reproduces what the capture's eval produced;
it is an **approximation** of the full rotary forward, and that is stated plainly
here and in `provenance.json` rather than glossed.

**There is no author submodule and no model-carrying pip dependency.** So this port
records **no** upstream block and defines **no** `UPSTREAM` in the adapter (they
must agree, and here both are absent). The eval stack is just torch / torchvision /
safetensors / numpy / PyYAML.

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds the V-JEPA 2 ViT from the config's architecture keys, loads
the downloaded checkpoint's `encoder.*` tensors into it frozen, and fits a single
linear layer on the mean-pooled patch feature. The manifest therefore carries
`encoder_absent_reason` rather than an encoder, and the backbone it read is named in
the config (`train.ckpt`).

Changed during the port (see `provenance.json`): the encoder is a self-contained
Conv3d-patch-embed ViT whose module names mirror the checkpoint's `encoder.*` keys,
so the tensors load with no remapping; the checkpoint is read directly from
safetensors (the `encoder.` prefix stripped, the `predictor.*` keys dropped); any
**missing or unexpected** encoder key on load is a hard error (a half-loaded
backbone would silently misreport what ran); the capture's **silent fallback to
random weights on a failed download is removed** (a `ckpt` that is not a loadable
V-JEPA 2 checkpoint raises); the device is **resolved** (`resolve_device`) rather
than assumed CUDA; features are extracted in **fp32** (no autocast); the input
normalisation is ImageNet mean/std with a bicubic square resize; the probe follows
this port's shared frozen-backbone protocol (mean-centre + L2-normalise, one linear
layer with SGD + cosine).

## The representation, and the caveat

The probe reads the V-JEPA 2 ViT's mean-pooled patch tokens, frozen. A real number
therefore measures the **pretrained** backbone, not something this port trained,
and it is an **image** proxy of a **video** model (a still image replicated across
the temporal axis) run with **plain attention** (see the caveat above). The
checkpoint is a **download pinned by sha256** in `provenance.json`, fetched and
hash-verified by `bin/fetch-weights.py`. The hermetic smoke leaves `train.ckpt`
empty and builds a **random tiny** V-JEPA 2 ViT at a small resolution, so nothing
is downloaded and its accuracy is meaningless — only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit. A well-formed
  `encoder.*` checkpoint (carrying a decoy `predictor.*` key that must be dropped)
  is loaded and probed, and a checkpoint missing an encoder weight is refused (both
  without the 1.3 GB download).
- **Not a full run:** `configs/linear_eval.yaml` pins the official
  facebook/vjepa2-vitl-fpc64-256 ViT-L recipe, not a completed run.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and
  runs where a device is visible.

## Environment

The eval stack is torch / torchvision / safetensors / numpy / PyYAML — no
transformers, no timm. There is **no submodule** to check out.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/vjepa2/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official V-JEPA 2 backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/vjepa2/provenance.json \
        --out .weights/vjepa2 --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/vjepa2/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set VJEPA2_CKPT=.weights/vjepa2/model.safetensors \
        --out resolved.json
    cd methods/vjepa2 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This stage
writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
