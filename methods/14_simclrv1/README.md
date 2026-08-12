# 14_simclrv1 — step 1 (SimCLR v1 ResNet-50 pretext) + linear evaluation

Chen, Kornblith, Norouzi & Hinton, *A Simple Framework for Contrastive Learning
of Visual Representations* (SimCLR v1), ICML 2020
([arXiv:2002.05709](https://arxiv.org/abs/2002.05709)).

Two independently-augmented views of an image feed a shared **ResNet-50** encoder
and a 2-layer MLP projection head (2048→2048→`out_dim`, BN + ReLU, L2-normalised).
The **NT-Xent** loss pulls the two views of an image together and pushes every
other view in the batch apart. Training uses **LARS** under a cosine schedule with
linear warmup. Step 1 is that pretext.

## Scope — the ResNet-50 path only

This port covers the paper-faithful **ResNet-50** step 1. The capture's ViT
variant (`models/vit_simclr.py`, step 2) imports `timm`, and is excluded as in
every port — which also drops the `timm`, `tensorboard` and `tqdm` dependencies.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/14_simclrv1` (the lab's own ResNet-50 model + projection head, NT-Xent
loss, LARS optimizer and two-view dataset, torch/torchvision only) — no
`third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with `SyncBatchNorm` and
logs to TensorBoard; none is needed for a single-process run, so
`train_step1_simclr.py` owns a thin fp32 loop, the device is **resolved** rather
than assumed CUDA, and TensorBoard is dropped. The NT-Xent `all_gather_with_grad`
path is kept but guarded by `dist.is_initialized()`, so it is inert
single-process (negatives come from the local batch). LARS and the
cosine-with-warmup schedule are kept.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **ResNet-50 backbone** (`encoder.*`) — the torchvision
`resnet50` up to and including avgpool, one 2048-d feature per image. The
projection head is training machinery and is excluded, and the round trip (write
it, load it back into a rebuilt model, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use
SimCLR's own eval pipeline (bicubic resize + centre crop, **no** ImageNet mean/std
normalisation — the encoder was trained on unnormalised [0,1] inputs, so the probe
feeds it the same). The probe then follows the lab's shared ARSSL protocol
(features cached once, mean-centred and L2-normalised, a single linear layer
trained with SGD under a cosine schedule), which is what makes the number
comparable across the ported methods.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small ResNet-50 SimCLR, a few
  fabricated images, a 32px input and `out_dim=32` — runs through `python -m
  adapter` on a CPU, passes `contract-test`, and the encoder round-trip and a
  determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/step1.yaml` is the SimCLR recipe (out_dim 128,
  1000 epochs, batch 4096, LARS lr 4.8), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/14_simclrv1-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra (the ViT step 2's `timm` is not ported).
`requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are
the hashed closures (the same closure as `13_mocov1`: identical floors, identical
resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/14_simclrv1/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py --config methods/14_simclrv1/configs/step1.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/14_simclrv1 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/14_simclrv1/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/14_simclrv1 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
