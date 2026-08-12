# 29_ijepa — step 1 (I-JEPA ViT pretext) + linear evaluation

Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture* (I-JEPA), 2023
([arXiv:2301.08243](https://arxiv.org/abs/2301.08243)).

I-JEPA predicts **in latent space**, not pixels. A **context encoder** (a ViT) sees
a large context block of patches; a narrow **predictor** predicts the
representations of several masked **target blocks**; the targets come from an
**EMA target encoder** (a momentum copy of the context encoder, no gradient),
layer-normalised, and the loss is a **smooth-L1** in latent space. No pixel
reconstruction, no hand-crafted augmentation invariances. Step 1 is that pretext.

## Scope — the ViT step 1 only

This port covers I-JEPA's ViT step 1. The capture's step 2 (ViT-B) is excluded, as
in every port. I-JEPA ships **its own** Vision Transformer
(`models/vision_transformer.py` — measured: it imports only `torch`, **not
`timm`**) and trains **from scratch on ImageNet-1k**, so this port is
**torch-only** and hermetic.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/29_ijepa` (the lab's own I-JEPA, following
[facebookresearch/ijepa](https://github.com/facebookresearch/ijepa): the ViT
context/target encoder and narrow predictor, the multi-block mask collator, the
cosine schedulers, the dataset, the trainer and the probe) — no `third_party/`
submodule.

The lab wrapper trains under `DistributedDataParallel` with bf16 autocast and
gradient accumulation and logs to TensorBoard; none is needed for a single-process
run, so `train_step1_ijepa.py` owns a thin fp32 loop, the device is **resolved**
rather than assumed CUDA, and AMP / TensorBoard / tqdm / no_sync / accumulation are
dropped. The mask collator, the context/target encoders, the narrow predictor, the
layer-normed latent targets, the smooth-L1 loss, the EMA target update and the
cosine LR / weight-decay / EMA schedules are kept faithfully. The port adds a
`vit_tiny` arch and builds the predictor with `pred_num_heads = max(1, pred_dim //
64)` so a small hermetic CPU smoke runs at a lower resolution.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **target** ViT encoder (`target_encoder.*`, the prefix stripped
so it loads straight into a plain `VisionTransformer`): the patch-embed conv, the
learnable position embedding, the transformer blocks and the final norm — one
embed_dim mean-pooled feature per image (1280 for `vit_huge`; I-JEPA has no CLS
token). The context encoder, the predictor and the mask token are training
machinery and are excluded, and the round trip (write it, load it back into a
rebuilt ViT, compare the weights) is tested. **The target encoder, not the context
encoder, is shipped** — it is the representation I-JEPA is evaluated on (the
capture's own linear eval uses the `target_encoder` key).

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation I-JEPA trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny ViT at 32px / patch 8 (a 4×4
  token grid, embed_dim 48), a narrow predictor, 2 target blocks, a few fabricated
  images — runs through `python -m adapter` on a CPU (exercising the mask
  collator, the context/predictor/target forward and the EMA update), passes
  `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/step1.yaml` is the I-JEPA recipe (`vit_huge`,
  224px, 300 epochs, batch 2048, AdamW, warmup 40), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/29_ijepa-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained torch-only stack (no
submodule, no `timm`; I-JEPA's ViT is its own, and TensorBoard is dropped).
`requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are the
hashed closures (the same closure as `19_byol`: identical floors, identical
resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/29_ijepa/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT contains a train/ subdirectory of images
    python bin/resolve-config.py --config methods/29_ijepa/configs/step1.yaml \
        --set DATA_ROOT=/path/to/imagenet --out resolved.json
    cd methods/29_ijepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/29_ijepa/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/29_ijepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
