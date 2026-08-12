# 36_franca — linear evaluation (frozen pretrained backbone)

Franca ([arXiv:2507.14137](https://arxiv.org/abs/2507.14137)), a self-supervised
ViT foundation model in the DINOv2 lineage.

## Why this method, and what is new here

**This is the first eval-only port: a `linear_eval` stage and no step 1.** In the
capture (`methods/36_franca/README.md`), Franca's "Step 1" is a *caveat eval* --
freeze the official pretrained Franca **ViT-B/14 In21K** backbone and fit a
linear probe on frozen CLS features, "analogous to DINOv2 ... **not local Franca
pretraining**". The from-scratch SSL pretraining (DINO/iBOT/Sinkhorn/KoLeo,
H100-class) is the capture's Step 2, and is excluded like every method's step 2.

So this port **trains nothing** and produces **no `encoder.pt`**; it probes a
frozen, downloaded backbone. This is the frozen-backbone / weight-download shape
that CONTRACT section 7 left open — see `docs/EVAL_DOWNLOAD.md`. Unlike var (which
probes a tokeniser), the representation here is a genuine SSL ViT (Franca's
pretrained CLS token), so the number **is** comparable.

The model is the pinned upstream `valeoai/Franca` under `third_party/franca`,
imported and never copied, and pinned **directly** (no fork): the frozen forward
has no hardcoded device. The inventory's `submodule+patch` (9984B) is for the
excluded Step 2.

Changed during the port (see `provenance.json`): the device is resolved rather
than assumed CUDA; features are extracted in fp32 (the capture used a bfloat16
autocast, a GPU speed path with no meaningful effect on a frozen-feature probe
and not portable to a CPU or pre-Ampere GPU); RASA is disabled (the capture notes
the official RASA loader mismatches ViT-B/14, and the CLS probe does not use it).

## The representation, and the caveat

The probe reads `forward_features(x)["x_norm_clstoken"]` — Franca's pretrained
CLS token, frozen. A real number therefore measures Franca's pretrained backbone
(the "pretrained-backbone reuse" row), not something this port trained. The
official checkpoint is a **download pinned by sha256** in `provenance.json`,
fetched and hash-verified by `bin/fetch-weights.py`. The hermetic smoke builds a
**random** ViT-B/14 (`pretrained=False`) at a tiny resolution, so nothing is
downloaded and its accuracy is meaningless — only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`).
- **Not a full run:** `configs/linear_eval.yaml` is the ViT-B/14 In21K + ARSSL
  probe recipe, not a completed run.
- **GPU:** the device resolution and a real-backbone probe are verified on real
  hardware.

## Environment

The eval stack is torch / torchvision / numpy / PyYAML; the pinned upstream also
imports tqdm, which is in the lock (`requirements.lock.in`). The heavier
dependencies in the upstream's own `requirements.txt` (pytorch-lightning,
omegaconf, webdataset, …) are for the excluded Step 2 training and are not needed
to build and probe the frozen backbone.

    git submodule update --init third_party/franca
    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/36_franca/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch + verify the official backbone (pinned by sha256 in provenance.json)
    python bin/fetch-weights.py --provenance methods/36_franca/provenance.json \
        --out .weights/franca --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each)
    python bin/resolve-config.py --config methods/36_franca/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set FRANCA_CKPT=.weights/franca/franca_vitb14_In21K.pth --out resolved.json
    cd methods/36_franca && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
