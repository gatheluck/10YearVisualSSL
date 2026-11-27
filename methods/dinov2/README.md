# dinov2 — released-checkpoint probe + controlled ViT-B/16 pre-training + linear eval

> See the [guide to original settings (ASIS), controlled pre-training (CTRL) and Unified evaluation](../../docs/PAPER_TERMINOLOGY.md) before matching a recipe to a paper result.

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](../../docs/SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023
([arXiv:2304.07193](https://arxiv.org/abs/2304.07193)).

Two comparisons live here:

- **Original settings (ASIS)** — a `linear_eval` with no `recipe`: the official pretrained
  **ViT-g/14** (LVD-142M) is downloaded (`third_party/dinov2`) and a linear probe
  is fit on its frozen **CLS token**. DINOv2's from-scratch SSL data (LVD-142M) is
  not public, so the as-is row reuses the official backbone. This trains nothing.

- **Controlled pre-training** — `pretrain` trains a **ViT-B/16 from scratch** on
  ImageNet-1k with the DINOv2 objective (DINO + iBOT + KoLeo), then `linear_eval`
  with `recipe: unified` freezes that trained `encoder.pt` and probes its CLS
  token. The common backbone and dataset let the paper compare training objectives under CTRL.

## Controlled ViT-B/16 pre-training, from scratch

`train_pretrain_vit_dinov2.py` is a single-process port of the capture's
`train_step2_vit.py`: a student and its EMA teacher (timm ViT-B/16), the shared
DINO/iBOT prototype head, DINO + iBOT + KoLeo losses, a per-epoch warmup→cosine LR
(peak = `base_lr × batch/1024`), a per-step teacher-momentum cosine, teacher-temp
warmup, `freeze_last_layer` for the first epoch and gradient-norm clipping — all
faithful. The capture's DDP, TensorBoard and distributed collapse-gate are dropped
for a single-process fp32 run; the device is resolved rather than assumed CUDA. The
DINOv2 model/loss/data are the capture's own (`models/`, `data/`), timm-based, so
**timm is a controlled recipe dependency** (built from scratch, no download → hermetic).

`encoder.pt` (controlled recipe) is the EMA **teacher backbone** (`teacher_bb.*`, prefix
stripped so it loads into a plain `DINOv2Backbone`); the heads, the student and the
centering buffers are excluded. The adapter also writes `encoder_epoch{100,200,300}.pt`
for the milestone probe sweep. Probe an encoder from controlled pre-training with
`configs/linear_eval_vit.yaml` (`recipe: unified`, CLS token).

## What is probed, and why it's comparable

The representation is DINOv2's pretrained CLS token
(`forward_features(x)["x_norm_clstoken"]`), frozen — a genuine SSL representation,
so the number is comparable across the ported methods (the "pretrained-backbone
reuse" row, like `franca`; unlike `var`, which probes a fixed tokeniser). The
probe follows the shared ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule).

## The backbone: pinned code + hash-pinned weights

The model is the pinned upstream `facebookresearch/dinov2` under
`third_party/dinov2`, **imported not copied**, pinned **directly** (no fork). It
is built with the **xformers path disabled** (`XFORMERS_DISABLED=1`) so the forward
is **torch-only** and reproducible on a CPU or any GPU — the giant's SwiGLU falls
back to a torch implementation whose `w12`/`w3` weight keys match the official
fused checkpoint, so the weights load strict (0 missing, 0 unexpected). The model
is built at the checkpoint's native **518px** (a 37×37 position-embedding grid) and
the input is interpolated to the eval resolution.

The weights are a **hash-pinned download** (`provenance.json` → `backbone_artifact`):
the official `dinov2_vitg14_pretrain.pth`, pinned by sha256. **CI never
downloads** — the hermetic smoke builds a random `dinov2_vits14` at a tiny
resolution, so nothing is fetched and only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a **random** ViT-S/14
  (`ckpt` empty → `pretrained=False`) at 28px over a two-class ImageFolder, runs
  through `python -m adapter`, passes `contract-test`, writes the comparable
  `linear_probe` accuracies (meaningless for a random backbone — only the pipeline
  is verified), writes **no** `encoder.pt`, and records the pinned upstream commit
  in the manifest.
- **Not exercised here:** a real linear-probe number, which needs the official
  4.5 GB `dinov2_vitg14_pretrain.pth` (fetch it with `bin/fetch-weights.py`).
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/dinov2-linear-eval.json`).

## Environment

torch / torchvision / numpy / PyYAML, plus `timm` for the controlled recipe ViT-B/16 backbone
(same closure as `franca`; the pinned upstream adds `tqdm`, imported through
`PYTHONPATH` and never installed, so it is not in the lock). The original backbone
under `third_party/dinov2` runs torch-only, with `xformers` disabled.

    git submodule update --init third_party/dinov2   # populate the pinned model
    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/dinov2/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch the hash-pinned official backbone (a real run; CI never does this)
    python bin/fetch-weights.py --provenance methods/dinov2/provenance.json \
        --artifact backbone_artifact --out /path/to/weights

    # linear eval: DATA_ROOT has train/ and val/; DINOV2_CKPT is the .pth above
    python bin/resolve-config.py --config methods/dinov2/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set DINOV2_CKPT=/path/to/weights/dinov2_vitg14_pretrain.pth --out eval.json
    cd methods/dinov2 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The port
writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason` and the pinned `upstream` commit.
