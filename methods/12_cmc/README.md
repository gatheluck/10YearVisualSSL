# 12_cmc — step 1 (CMC AlexNet pretext) + linear evaluation

Tian, Krishnan & Isola, *Contrastive Multiview Coding*, 2019
([arXiv:1906.05849](https://arxiv.org/abs/1906.05849)).

An RGB image is converted to CIE **Lab** and split into its **L** (1-channel) and
**ab** (2-channel) views. A two-branch half-size AlexNet maps each view to an
L2-normalised embedding, and an **NCE** loss over two momentum **memory banks**
(one per view, scored cross-view) pulls the two views of an image together and
apart from K negatives. Step 1 is that pretext.

## Scope — the paper-faithful AlexNet path and the unified ViT-B/16 Step 2

The capture ships a CMC AlexNet step 1, a ViT step 2, and an optional ResNet
linear-classifier variant. This port brings across the **AlexNet** path (the
two-branch encoder, the two-bank NCE loss with alias-method negative sampling,
and the Lab dataset) **and** the capture's unified **ViT-B/16 Step 2**
(`configs/pretrain_vit.yaml`, `arch: vit`): two ViT branches — one for the L
channel (`in_chans` 1), one for ab (`in_chans` 2) — each with a 3-layer MLP
projector, trained from scratch by the same cross-view NCE memory-bank objective;
AdamW + warmup/cosine, no AMP/clip; checkpoints at 100/200/300 epochs, the linear
probe reading both branches' concatenated CLS features. The ViT path needs `timm`
(imported lazily); the native AlexNet path is byte-for-byte unchanged. The
capture's optional ResNet linear-classifier variant remains excluded, as in every
port.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/12_cmc` AlexNet files (the lab's own model, NCE machinery, Lab dataset,
trainer and probe, torch/torchvision only) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP and logs to
TensorBoard; none is needed for a single-process run, so `train_pretrain_cmc.py`
owns a thin fp32 loop, the device is **resolved** rather than assumed CUDA,
TensorBoard/tqdm are dropped, and NCE negatives come from within the batch (the
cross-rank all-gather / broadcast / all-reduce paths are kept but guarded by
`dist.is_initialized()`, so they are inert single-process). The two memory banks
live in the `NCEAverage` loss module, not the model.

**The Lab conversion is reimplemented in numpy** (sRGB → XYZ(D65) → CIE Lab),
verified against published CIE Lab reference values, so the port keeps the
torch-only closure — scikit-image is **not** a dependency (the capture uses
`skimage.color.rgb2lab` with a PIL fallback).

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **two-branch AlexNet encoder** (`encoder_l.*` /
`encoder_ab.*`). The NCE memory banks (`memory_l`, `memory_ab`) live in the
separate `NCEAverage` module, so the model's `state_dict` never carries them and
`encoder.pt` naturally excludes them (the same shape as the inst_disc port). The
round trip (write it, load it back into a rebuilt model, compare the weights) is
tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes the
**layer-6 (fc6) features of both branches, concatenated** (2 × 2048) — the paper's
best single-branch layer. Images use the deterministic val pipeline (resize +
centre crop, no augmentation); the probe follows the lab's ARSSL protocol
(features cached once, mean-centred and L2-normalised, a single linear layer
trained with SGD under a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a narrow encoder, a 64px input, a
  handful of negatives, a few fabricated images — runs through `python -m adapter`
  on a CPU, passes `contract-test`, and the encoder round-trip and a determinism
  check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the paper-target recipe (feat_dim
  128, K 16384, T 0.07, 240 epochs, 224px), a recipe, not a completed run.
- **Exercised (ViT Step 2):** a hermetic smoke trains the two-branch ViT under
  the two-bank NCE objective, writes milestone encoders, and probes the
  concatenated CLS features through `contract-test` (tiny ViT dims, CPU).
- **Not ported:** the optional ResNet linear-classifier variant.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/12_cmc-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule (the Lab conversion is numpy, not scikit-image) — plus `timm` for the
unified ViT-B/16 Step-2 path (imported lazily, so the native AlexNet path never
needs it). `requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt`
(CUDA 13.0) are the hashed closures (the same closure as `14_simclrv1`: identical
floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/12_cmc/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/12_cmc/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/12_cmc && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/12_cmc/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/12_cmc && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
