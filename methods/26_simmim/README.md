# 26_simmim — step 1 (SimMIM Swin-B pretext) + unified ViT-B/16 step 2 + linear evaluation

Xie et al., *SimMIM: A Simple Framework for Masked Image Modeling*, 2022
([arXiv:2111.09886](https://arxiv.org/abs/2111.09886)).

SimMIM is **masked image modeling**. Images are patch-embedded by a Swin encoder;
a random block of patch tokens (mask units of 32×32 pixels, 60% masked) is
replaced by a learned **mask token**; the full token grid is encoded; a
lightweight **1×1 conv + PixelShuffle** decoder reconstructs pixels; an **L1
loss** is taken only on the masked pixels. Step 1 is that pretext.

## Scope — the Swin-B step 1 and the unified ViT-B/16 step 2

This port covers SimMIM's **Swin-B** step 1 and, added additively, the capture's
unified **ViT-B/16** step 2 (see the Step 2 section below). SimMIM's step 1 is
genuinely **Swin**-based, so **`timm` is a pretrain dependency** (it supplies the
`SwinTransformer`, and — for step 2 — the `VisionTransformer`) — this is the repo's
**first Swin backbone**. The `transformers` package in the capture's
`requirements.txt` is **docstring-only** (a note about HuggingFace pretrained
weights) and is **never imported** (measured), so it is not a dependency here. Both
backbones are built **from scratch** (no pretrained download), so the run stays
hermetic.

## Step 2 (unified ViT-B/16), added additively

Unlike the other ViT Step-2 ports, SimMIM's native backbone is a **Swin**, not a
ViT — so the unified Step 2 is a genuinely **different backbone**, not a re-tuning
of the same one. The capture plugs the same masked-image-modelling objective into
a **timm ViT-B/16** (`models/simmim_vit.py`): 300 epochs, batch 1024, a direct
`lr` 6e-4, betas (0.9, 0.95), fixed weight decay 0.05, 10-epoch warmup then cosine
to `min_lr`, checkpointed at 100/200/300. It is selected by a `recipe: unified`
key in `train`; **absent `recipe` is the native Swin-B path, unchanged**, and the
native and unified config key sets are disjoint (native-only `window_size`/`depths`
/multistep-decay knobs ↔ unified-only `depth`/`mlp_ratio`/`min_lr`/`save_at_epochs`).

The masking is pixel-space, as in the Swin variant (masked pixels replaced by a
learned mask token, one mask unit = one ViT patch); the dataloader already
supports the pixel-mask path (`return_pixel_mask=True`). `train_pretrain_vit_simmim.py`
is single-process, reusing the native trainer's device resolution and seeding
(one implementation per method). `encoder.pt` is the bare timm ViT (`encoder.*`).

**This is the only Step-2 port that is non-additive to the linear evaluation.**
The native probe mean-pools the Swin tokens; the ViT's representation is its **CLS
token** (the capture's own choice), so `evaluate_linear_simmim.py` gains a `pool`
argument (`cls` for the ViT, `mean` for the Swin), fixed from the recipe by the
adapter. The adapter writes `encoder.pt` plus `encoder_epoch{100,200,300}.pt`, one
frozen ViT per milestone; probe a unified encoder with `configs/linear_eval_vit.yaml`.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/26_simmim` (the lab's own SimMIM, following
[microsoft/SimMIM](https://github.com/microsoft/SimMIM): the `SimMIMSwinB` model
wrapping timm's `SwinTransformer`, the block-mask generator + dataset, the trainer
and the probe) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP autocast and logs
to TensorBoard; none is needed for a single-process run, so
`train_pretrain_simmim.py` owns a thin fp32 loop, the device is **resolved** rather
than assumed CUDA, and AMP / TensorBoard / tqdm are dropped. The mask generator,
the mask-token replacement, the Conv+PixelShuffle decoder and the masked-pixel L1
loss are kept faithfully; `build_swin_encoder` is the single construction path
shared by the model and the linear-eval loader; `img_size` and the Swin dims are
**threaded** (the capture hard-coded 192) so a small hermetic CPU smoke runs a
tiny 2-stage Swin at a lower resolution.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the bare **Swin encoder** (`encoder.*`, the prefix stripped so it
loads straight into a plain timm `SwinTransformer`): the patch embed, the Swin
stages and the final norm — one encoder_dim mean-pooled feature per image (1024
for Swin-B). The learned mask token and the reconstruction decoder are training
machinery and are excluded, and the round trip (write it, load it back into a
rebuilt Swin, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation SimMIM trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods. (The capture's own SimMIM eval standardises the cached
features then fits a head; using the shared single-feature probe instead is a
documented deviation, the same as every other port.)

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny 2-stage Swin at 16px (an 8×8
  token grid, encoder_dim 32), a few fabricated images — runs through `python -m
  adapter` on a CPU (exercising the mask generation, the mask-token replacement
  and the masked-pixel L1 loss), passes `contract-test`, and the encoder
  round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the SimMIM recipe (Swin-B, 192px,
  800 epochs, batch 2048, AdamW, warmup 10 → multistep at 700), a recipe, not a
  completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/26_simmim-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML **plus `timm`** (and its transitive closure)
— SimMIM's step 1 is genuinely Swin-based. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same closure
as `22_mocov3`: identical timm resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/26_simmim/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT contains a train/ subdirectory of images
    python bin/resolve-config.py --config methods/26_simmim/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/imagenet --out resolved.json
    cd methods/26_simmim && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/26_simmim/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/26_simmim && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
