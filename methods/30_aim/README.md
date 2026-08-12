# 30_aim — linear evaluation only (frozen pretrained AIM backbone)

El-Nouby et al., *Scalable Pre-training of Large Autoregressive Image Models*,
2024 ([arXiv:2401.08541](https://arxiv.org/abs/2401.08541)).

AIM pretrains a ViT **autoregressively** (predict image patches in raster order),
at scale, on DFN-2B+ (~2B uncurated images). This is an **eval-only** port: like
`28_dinov2` and `36_franca`, it fits a linear probe on the **frozen official
pretrained backbone** (AIM-600M, ViT-H/14) and reports a comparable number,
because the from-scratch pretraining data (DFN-2B+) is not public.

## Scope — the eval-only "as-is SSL comparison"

The capture's AIM "Step 1" is a frozen-backbone probe: download the official
AIM-600M and probe it. That is what this port covers. AIM's from-scratch
autoregressive pretraining on the unavailable DFN-2B+ (and the capture's own
from-scratch step 2, `models/aim_vit.py`) is the **excluded step**, as in every
port. This port **trains nothing** and produces **no `encoder.pt`**; it probes a
frozen, hash-pinned downloaded backbone.

## Licence — non-commercial research use only

The AIM code (`apple/ml-aim`) and the AIM-600M weights (`apple/AIM` on
HuggingFace) are released by Apple under the **Apple ML Research License
(apple-amlr)**, which permits use for **non-commercial research purposes only**.
This port is used solely for academic research. **Nothing under apple-amlr is
copied into this repository**: the code is referenced as a pinned git submodule
(`third_party/ml-aim`, imported through PYTHONPATH, never installed) and the
weights are a **hash-pinned download that CI never fetches**. This port's own files
(the adapter, the evaluation, the configs, the tests) are original and carry this
repository's licence; they only *reference* the Apple artifacts. See
`provenance.json` (`licence_note`, `upstream`, `backbone_artifact`).

## What the representation is, and why it is comparable

The frozen feature is the **average of the last 6 transformer blocks**, mean-pooled
over the patch tokens — one 1536-d vector per image (the capture's
`AIMBackboneWrapper` behaviour, kept faithfully via forward hooks). This is a
genuine SSL representation, so the linear-probe number is comparable. The probe
follows the lab's shared ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule),
which makes the number comparable across the ported methods. (The capture's own
AIM eval fits an *attentive*-pooling probe, which it marks reference-only; using
the shared single-feature linear probe instead is a documented deviation, the same
as `28_dinov2` / `36_franca`.)

The backbone is built from AIM-600M's dims via the pinned upstream (ml-aim's
`_aim`), and the official backbone checkpoint loads **strict** (measured: 0 missing,
0 unexpected on the backbone; only the unused attentive-probe head is absent).

## What has and has not been exercised

- **Exercised (linear_eval):** a hermetic smoke — a random **tiny** AIM (4 small
  blocks at 32px, embed_dim 32, the last 2 blocks averaged; `ckpt` empty, so
  nothing is downloaded) — runs through `python -m adapter` on a CPU, passes
  `contract-test`, writes the comparable `linear_probe` accuracies, writes **no**
  `encoder.pt`, and the manifest carries `encoder_absent_reason` and the pinned
  `upstream`.
- **Measured (not a full run):** the official AIM-600M backbone was downloaded once
  and confirmed to load into the `_aim`-built model with 0 missing / 0 unexpected
  (backbone), and its sha256 recorded in `provenance.json`. A full ImageNet probe
  is the recipe in `configs/linear_eval.yaml`, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/30_aim-linear-eval.json`).

## Environment

torch / torchvision / numpy / PyYAML, **plus the pinned upstream's dependency
`huggingface_hub`** (ml-aim's model module imports `PyTorchModelHubMixin`) and its
closure. `requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA
13.0) are the hashed closures. The `aim` package itself is the submodule under
`third_party/ml-aim`, imported through PYTHONPATH and never installed, so it is not
in the lock.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/30_aim/requirements.lock.txt -r requirements-tools.lock.txt

    git submodule update --init third_party/ml-aim   # the AIM code (apple-amlr)

## Running

    # fetch + hash-verify the official AIM-600M backbone (apple-amlr; research use)
    python bin/fetch-weights.py --provenance methods/30_aim/provenance.json \
        --artifact backbone_artifact --out /path/to/weights

    # linear eval: DATA_ROOT has train/ and val/; CKPT is the fetched backbone
    python bin/resolve-config.py --config methods/30_aim/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set CKPT=/path/to/weights/aim_600m_2bimgs_attnprobe_backbone.pth --out eval.json
    cd methods/30_aim && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason` and the pinned `upstream`.
