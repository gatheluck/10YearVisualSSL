# nepa — NEPA ViT pretext + controlled ViT-B/16 pre-training + linear evaluation

> See the [guide to original settings (ASIS), controlled pre-training (CTRL) and Unified evaluation](../../docs/PAPER_TERMINOLOGY.md) before matching a recipe to a paper result.

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](../../docs/SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

Xu et al., *NEPA: Next-Embedding Predictive Autoregression*, 2025
([arXiv:2512.16922](https://arxiv.org/abs/2512.16922)).

NEPA predicts the **next patch embedding, autoregressively, in latent space**.
Patch embeddings `z = f(x)` run through a **causal** transformer predictor `h` to
give `z_hat`; the loss is the **negative cosine similarity** between `z_hat[:, :-1]`
and a **stop-gradient** shifted target `z[:, 1:]` (SimSiam style). An EMA copy of
the whole model is kept for evaluation. The ViT uses **2D RoPE**, **QK-norm**,
**LayerScale** and GeLU (optionally SwiGLU).

## Scope — the ViT original recipe and controlled ViT-B/16 pre-training

This port covers NEPA's ViT original recipe **and** the capture's controlled ViT-B/16 pre-training
(see the controlled pre-training section below). NEPA ships **its own** Vision Transformer
(`models/nepa_vit.py` — measured: it imports only `torch`, **not `timm`**, despite
`timm` appearing as a floor in the capture's `requirements.txt`) and trains **from
scratch on ImageNet-1k**, so this port is **torch-only** and hermetic for both.

## Controlled ViT-B/16 pre-training

The controlled recipe is the **same `NEPAModel`** with the controlled hyperparameters — no new
model, no `arch` key. `configs/pretrain_vit.yaml` selects it: standard **ViT-B/16**
(`patch_size: 16` vs original recipe's 14), the controlled pre-training augmentation (RandomResizedCrop +
flip + ColorJitter), SwiGLU, AdamW with the controlled recipe (peak LR =
`base_lr × batch/256` = 6e-4, wd 0.05, 10-epoch warmup → cosine), and milestone
checkpoints via `save_at_epochs: [100, 200, 300]` (absent in the native config, so
the original recipe is unchanged). The adapter writes `encoder.pt` plus `encoder_epoch{100,200,
300}.pt`, one frozen EMA model per milestone. The controlled recipe linear probe pools the
**raw patch embeddings** (`configs/linear_eval_vit.yaml`, `pool: embed`, the
capture's controlled recipe choice) rather than the native AR-output average (`pool: avg`).

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/nepa` (the lab's own NEPA: the ViT with 2D RoPE / QK-norm / LayerScale
/ causal autoregressive predictor, the dataset, the trainer and the probe) — no
`third_party/` submodule. It is the repo's **first 2D-RoPE ViT** and its first
**autoregressive** latent-prediction objective.

The lab wrapper trains under `DistributedDataParallel` with bf16 autocast and logs
to TensorBoard; none is needed for a single-process run, so `train_pretrain_nepa.py`
owns a thin fp32 loop, the device is **resolved** rather than assumed CUDA, and
AMP / TensorBoard / tqdm are dropped. The 2D RoPE, QK-norm, LayerScale, the causal
autoregressive predictor, the stop-gradient negative-cosine loss and the EMA update
are kept faithfully. The port adds a `build_nepa_model` that constructs from
explicit dims (alongside the capture's `build_nepa_vit_base`/`_large`) so a small
hermetic CPU smoke runs a tiny ViT (head_dim divisible by 4 for the 2D RoPE).

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **EMA** model (`ema_model.*`, the prefix stripped so it loads
straight into a plain `NEPAModel`): the patch embed, the (optional) CLS token, the
causal transformer blocks and the final norm — the **whole model**, since NEPA's
representation is the autoregressive predictor output, not a separable backbone.
The online (gradient-trained) model is training machinery and is excluded, and the
round trip (write it, load it back into a rebuilt model, compare the weights) is
tested. **The EMA model, not the online model, is shipped** — it is the
representation NEPA is evaluated on (the capture's own eval uses the EMA model).

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes the
EMA model's **avg-pooled** causal predictor output (`extract_features(pool="avg")`,
the paper's linear-probe pooling). Images use the deterministic val pipeline
(resize + centre crop, NEPA's 0.5/0.5 normalisation — the same normalisation NEPA
trains with); the probe follows the lab's shared ARSSL protocol (features cached
once, mean-centred and L2-normalised, a single linear layer trained with SGD under
a cosine schedule), which makes the number comparable across the ported methods.

## What has and has not been exercised

- **Exercised (original recipe):** a hermetic smoke — a tiny ViT at 32px / patch 8 (a 4×4
  token grid + CLS, embed_dim 32, head_dim 8, depth 2), a few fabricated images —
  runs through `python -m adapter` on a CPU (exercising the 2D RoPE, the causal
  predictor, the stop-gradient loss and the EMA update), passes `contract-test`,
  and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Exercised (controlled ViT recipe):** a hermetic smoke trains the controlled ViT-B/16 (tiny
  dims) with `save_at_epochs`, writes milestone encoders, and probes the raw patch
  embeddings (`pool: embed`) through `contract-test` on a CPU.
- **Not a full run:** `configs/pretrain.yaml` is the NEPA recipe (ViT-B/14, 224px,
  1600 epochs, batch 4096, AdamW, warmup 40), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/nepa-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained torch-only stack (no
submodule, no `timm`; NEPA's ViT is its own, and TensorBoard is dropped).
`requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are the
hashed closures (the same closure as `byol`: identical floors, identical
resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/nepa/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # original recipe: DATA_ROOT contains a train/ subdirectory of images
    python bin/resolve-config.py --config methods/nepa/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/imagenet --out resolved.json
    cd methods/nepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is the encoder.pt from pre-training
    python bin/resolve-config.py --config methods/nepa/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/nepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
