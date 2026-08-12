# 15_mocov2 — step 1 (MoCo v2 ResNet-50 pretext) + linear evaluation

Chen, Fan, Girshick & He, *Improved Baselines with Momentum Contrastive Learning*
(MoCo v2), 2020 ([arXiv:2003.04297](https://arxiv.org/abs/2003.04297)).

Two independently-augmented views of an image feed a **query encoder** (ResNet-50
+ a 2-layer MLP projection head, L2-normalised) and a **momentum key encoder** (an
EMA copy of the query, no gradient). An **InfoNCE** loss contrasts the query
against the matching key (the positive) and a FIFO **queue** of K past keys (the
negatives). Step 1 is that pretext.

**MoCo v2 = MoCo v1 + three changes:** a 2-layer MLP projection head (v1 used a
single linear), Gaussian-blur augmentation, and a cosine LR schedule. Everything
else (queue K, EMA m, InfoNCE) is unchanged; the temperature is τ=0.2.

## Scope — the ResNet-50 path only

This port covers the paper-faithful **ResNet-50** step 1. The capture's ViT
variant (`models/vit_mocov2.py`, step 2) imports `timm`, and is excluded as in
every port — which also drops the `timm`, `tensorboard` and `tqdm` dependencies.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/15_mocov2` (the lab's own ResNet-50 MoCo v2 model, two-view dataset,
trainer and probe, torch/torchvision only) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with SyncBatchNorm and logs
to TensorBoard; none is needed for a single-process run, so
`train_pretrain_mocov2.py` owns a thin fp32 loop, the device is **resolved** rather
than assumed CUDA, TensorBoard is dropped, and the queue is filled from within the
batch (the shuffle-BN / all-gather paths are kept but inert single-process). The
cosine LR schedule is kept.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the query **ResNet-50 backbone** (`encoder_q.backbone.*`), one
2048-d feature per image. The MLP projection head, the momentum key encoder and
the queue are training machinery and are excluded, and the round trip (write it,
load it back into a rebuilt model, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation MoCo v2 trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small ResNet-50 MoCo v2, a few
  fabricated images, a 32px input, a tiny queue (K=4) — runs through `python -m
  adapter` on a CPU, passes `contract-test`, and the encoder round-trip and a
  determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the MoCo v2 recipe (feature_dim 128,
  K 65536, τ 0.2, 200 epochs, cosine), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/15_mocov2-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra (the ViT step 2's `timm` is not ported).
`requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are
the hashed closures (the same closure as `13_mocov1`: identical floors, identical
resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/15_mocov2/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py --config methods/15_mocov2/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/15_mocov2 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/15_mocov2/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/15_mocov2 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
