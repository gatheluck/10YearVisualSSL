# 31_dinov3 — step 1 (DINOv3 core pretext) + linear evaluation

Siméoni et al., *DINOv3*, 2025
([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)).

DINOv3 is self-distillation at scale. A **student** ViT (with **register tokens**
and **axial RoPE**) and an EMA **teacher** see multi-crop views; the core objective
combines three losses:

    loss = L_DINO + L_iBOT + koleo_weight · L_KoLeo

**L_DINO** is a cross-view distillation with **Sinkhorn-Knopp (SwAV) centering** of
the teacher assignments; **L_iBOT** is a masked-patch distillation (block masks on
the global crops); **L_KoLeo** spreads the batch's CLS features. Step 1 is that
pretext.

## Scope — the from-scratch step 2, core objective

The capture's DINOv3 "step 1" **downloads the official pretrained weights** (the
from-scratch data, LVD-1689M, is not public) — and those weights are **HuggingFace
login-gated**, so that download is the excluded step. What this port covers is the
capture's **step 2**: the from-scratch **unified SSL comparison** that trains a
DINOv3 representation on ImageNet-1k. It is **self-contained torch-only** code (the
ViT, the losses and the multi-crop dataset are the lab's own; no timm/transformers).

The released DINOv3 recipe adds a **Gram anchoring** second stage (epochs 251–300,
a snapshotted Gram teacher). The capture exposes this as `gram.mode`, with
**`core_only`** as a first-class mode; this port runs `core_only` and **excludes
the Gram anchoring stage**, as every port excludes a secondary stage. The
`GramLoss` module is shipped for completeness but is not wired into the loss.

## Licence

This port is a **self-contained re-implementation** (the lab's own code,
referencing the paper); it trains **from scratch** and uses **no Meta-released
DINOv3 code or weights**. The official `facebookresearch/dinov3` code and its
pretrained weights carry Meta's custom, gated *DINOv3 License*; none of it is
copied, downloaded, or required here (the capture's gated-weights step 1 is
excluded). So no third-party licence attaches to this port's files. See
`provenance.json` (`licence_note`).

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **EMA teacher's ViT backbone** (`backbone.*` from
`teacher_state_dict`, the prefix stripped so it loads into a plain
`VisionTransformer`): patch embed, cls/register tokens, RoPE blocks and the final
norm — one embed_dim CLS feature per image (768 for ViT-B/16). The DINO and iBOT
heads and the student are training machinery and are excluded (the teacher is the
representation DINO-family methods are known for), and the round trip is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. The probe
follows the lab's shared ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule).
(The capture's own DINOv3 eval trains plain linear heads on the frozen backbone;
using the shared single-feature probe instead is a documented deviation.)

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny ViT at 32px (a 2×2 patch grid,
  embed_dim 32, 2 blocks, 2 register tokens), 2 global + 2 local crops, a few
  fabricated images — runs through `python -m adapter` on a CPU (exercising the
  multi-crop forward, the Sinkhorn-centred DINO loss, the block-masked iBOT loss,
  KoLeo and the EMA teacher), passes `contract-test`, and the encoder round-trip
  and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1 encoder
  over a two-class ImageFolder, passes `contract-test`, writes the comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the DINOv3 core recipe (ViT-B/16,
  224px, 2+8 crops, 300 epochs, batch 1024, AdamW), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/31_dinov3-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML (the same torch-only closure as
`05_jigsaw_puzzle`). `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures. No timm: the
ViT is the lab's own; `transformers`/`huggingface_hub` are only for the capture's
excluded step 1, so they are not dependencies here.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/31_dinov3/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT contains a train/ subdirectory of images
    python bin/resolve-config.py --config methods/31_dinov3/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/imagenet/train --out resolved.json
    cd methods/31_dinov3 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/31_dinov3/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/31_dinov3 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
