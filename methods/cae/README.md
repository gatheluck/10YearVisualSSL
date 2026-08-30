# cae — as-is linear eval on the frozen pretrained backbone (eval-only)

CAE (Chen, Ding, Wang, Xie, Lu, Yuan, Chen, Bai & Zhang, *Context Autoencoder for
Self-Supervised Representation Learning*, 2022;
[arXiv:2202.04200](https://arxiv.org/abs/2202.04200)), a masked-image method that
predicts the latent representations of masked patches — in a latent space *aligned*
to the encoder's — with an explicit "latent regressor", separating representation
learning from the pretext decoding.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. CAE's "as-is" comparison freezes the pretrained vision
backbone (a BEiT-architecture ViT-B/16) and fits a linear probe on its CLS token,
because the from-scratch context-autoencoder pretraining (IN-1k, many GPU-days) is
the excluded step. The probed representation is a genuine SSL feature, so the number
**is** comparable — the "pretrained-backbone reuse" row, analogous to `eva02` /
`aimv2` / `data2vec2` / `36_franca`.

## The checkpoint, stated plainly

The capture design-of-record logs CAE as a **deficiency (DEF-02)**: the checkpoint
its own doc names (`hujinwen/cae-base`) is **not** a valid HuggingFace identifier
(the capture's own eval scripts confirm it never resolves), and the paper authors'
official weights are distributed **only via Baidu**. So the capture pipeline fell
back to a **BEiT-v2 proxy** and explicitly refused to label that number CAE.

**This port ships no proxy.** It pins a real, publicly downloadable CAE ViT-B
checkpoint the capture missed: **OpenMMLab's mmselfsup reproduction**
(`download.openmmlab.com/.../cae_vit-base-p16_...`, Apache-2.0). It is a **faithful
reproduction, not the paper authors' released weights**, and `provenance.json` says
so plainly. Pinning a real reproduction and labelling it honestly is the measured
correction to the capture's stale `hujinwen/cae-base` pointer (per `CLAUDE.md`: fix
whichever doc is stale).

## Why this method, and what is new here

**cae is the first port whose backbone is a self-contained model in the method
itself.** Its only stage is `linear_eval`. Unlike the `timm`-sourced siblings
(`eva02` / `aimv2`), the `transformers`-sourced `data2vec2`, or the submodule-sourced
`beitv2`, **neither timm nor transformers carries a CAE model class**, and the
checkpoint is in OpenMMLab mmselfsup format. So the encoder is a **small
self-contained BEiT-style ViT** in `evaluate_linear_cae.py`, built from the config's
architecture keys, and the checkpoint's `backbone.*` tensors load into it directly.

**No mmcv / mmpretrain in the fleet.** The OpenMMLab checkpoint pickles mmengine
bookkeeping (a `HistoryBuffer` in its `meta` / `message_hub`). It is read with a
**standard-library tolerant unpickler** that maps any pickled `mmengine` / `mmcv`
class to a discarded placeholder, so only the tensors are read and no heavy, fragile
OpenMMLab package is a dependency. The eval stack is just torch / torchvision /
numpy / PyYAML.

**There is no author submodule and no model-carrying pip dependency.** So this port
records **no** upstream block and defines **no** `UPSTREAM` in the adapter (they must
agree, and here both are absent).

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds the CAE ViT from the config's architecture keys, loads the
downloaded checkpoint's `backbone.*` tensors into it frozen, and fits a single linear
layer on the CLS feature. The manifest therefore carries `encoder_absent_reason`
rather than an encoder, and the backbone it read is named in the config
(`train.ckpt`).

Changed during the port (see `provenance.json`): the encoder is a self-contained
BEiT-style ViT whose module names mirror the checkpoint's `backbone.*` keys, so the
tensors load with no remapping; the checkpoint is read **without** mmcv/mmpretrain via
a tolerant unpickler; any **missing or unexpected** `backbone.*` key on load is a hard
error (a half-loaded backbone would silently misreport what ran); the device is
**resolved** (`resolve_device`) rather than assumed CUDA; features are extracted in
**fp32** (no autocast); the input normalisation follows the backbone's **own
preprocessor** config (ImageNet mean/std, a bicubic square resize, no centre crop),
with the resolution read from the config; the probe follows this port's shared
frozen-backbone protocol (mean-centre + L2-normalise, one linear layer with SGD +
cosine).

## The representation, and the caveat

The probe reads the CAE ViT's CLS token, frozen. A real number therefore measures the
**pretrained** backbone, not something this port trained. The checkpoint is a
**download pinned by sha256** in `provenance.json`, fetched and hash-verified by
`bin/fetch-weights.py`. The hermetic smoke leaves `train.ckpt` empty and builds a
**random tiny** CAE ViT at a small resolution, so nothing is downloaded and its
accuracy is meaningless — only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit. A well-formed
  `backbone.*` checkpoint is loaded and probed, and a checkpoint missing a backbone
  weight is refused (both without the 1.1 GB download).
- **Not a full run:** `configs/linear_eval.yaml` pins the OpenMMLab CAE ViT-B recipe,
  not a completed run.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and runs
  where a device is visible.

## Environment

The eval stack is torch / torchvision / numpy / PyYAML — no mmcv, no transformers, no
timm. There is **no submodule** to check out.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/cae/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the OpenMMLab CAE backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/cae/provenance.json \
        --out .weights/cae --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/cae/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set CAE_CKPT=.weights/cae/cae_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221230-808170f3.pth \
        --out resolved.json
    cd methods/cae && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
