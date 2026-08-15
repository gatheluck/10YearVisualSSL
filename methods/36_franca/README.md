# 36_franca — as-is Step-1 probe + unified ViT-B/16 Step-2 pretrain + linear eval

Franca ([arXiv:2507.14137](https://arxiv.org/abs/2507.14137)), a self-supervised
ViT foundation model in the DINOv2 lineage.

Two comparisons live here:

- **Step 1 (as-is)** — a `linear_eval` with no `recipe`: freeze the official
  pretrained Franca **ViT-B/14 In21K** backbone (downloaded, `third_party/franca`)
  and probe its frozen CLS token, "analogous to DINOv2 ... **not local Franca
  pretraining**". This trains nothing and produces no `encoder.pt`.

- **Step 2 (unified)** — `pretrain` trains a **ViT-B/16 from scratch** on
  ImageNet-1k with Franca's objective, then `linear_eval` with `recipe: unified`
  probes the trained `encoder.pt` at its CLS token. This puts Franca on the same
  axis as every other method's Step 2 (all train on ImageNet-1k, so the data
  confound is gone). (Added after the initial port, which probed only the
  downloaded Step-1 backbone; see the Step 2 section.)

## Step 2 (unified ViT-B/16), from scratch

Franca shares DINOv2's DINO+iBOT+KoLeo core but is a distinct method: its
contribution is the **nested Matryoshka projection heads** (coarse-to-fine
prototype sets over `nesting_dims = [48, 96, 192, 384, 768]`) and **Sinkhorn-Knopp**
teacher normalisation (instead of DINOv2's EMA centering). `train_pretrain_vit_franca.py`
is a single-process port of the capture's `train_step2_vit.py` (per-step
warmup→cosine LR, per-step teacher-momentum and teacher-temp schedules, the
Sinkhorn DINO/iBOT losses + KoLeo, gradient-norm clipping, milestone saves). The
capture's DDP / bf16 autocast / TensorBoard / health-gate are dropped; the device
is resolved. `models/` reuses the DINOv2 ViT backbone (shared design with
28_dinov2, vendored so the method is self-contained) and adds `MatryoshkaHead` +
the Sinkhorn losses; the multi-crop / cyclic-mask data is vendored too. **timm is a
Step-2 dependency** (the ViT is built from scratch, so CI stays hermetic).

`encoder.pt` (Step 2) is the EMA **teacher backbone** (`teacher_bb.*`), with the
nested heads and student excluded; the adapter also writes
`encoder_epoch{100,200,300}.pt` for the milestone sweep. Probe a Step-2 encoder
with `configs/linear_eval_vit.yaml` (`recipe: unified`, CLS token).

## The eval-only Step 1 (unchanged)

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. Unlike var (which probes a tokeniser), the
representation is a genuine SSL ViT (Franca's pretrained CLS token), so the number
**is** comparable.

The model is the pinned upstream `valeoai/Franca` under `third_party/franca`,
imported and never copied, and pinned **directly** (no fork): the frozen forward
has no hardcoded device. The inventory's `submodule+patch` (9984B) is the
capture's own Step-2 implementation; this port re-implements the unified Step 2
additively as port-authored files (see the Step 2 section) rather than applying
that patch.

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
omegaconf, webdataset, …) drive the capture's own Step-2 training pipeline; this
port's additive Step 2 is port-authored and does not use them (it needs only
`timm`), so they are absent from the lock.

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
