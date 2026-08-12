# 13_mocov1 — step 1 (MoCo v1 ResNet-50 pretext) + linear evaluation

He, Fan, Wu, Xie & Girshick, *Momentum Contrast for Unsupervised Visual
Representation Learning*, 2019
([arXiv:1911.05722](https://arxiv.org/abs/1911.05722)).

Two augmented views of an image feed a **query encoder** (ResNet-50 + a single
Linear(2048, 128) projection, L2-normalised — no MLP, that is v2) and a
**momentum key encoder** (an EMA copy of the query, no gradient). An **InfoNCE**
loss contrasts the query against the matching key (the positive) and a FIFO
**queue** of K past keys (the negatives). Step 1 is that pretext.

## Scope — the paper-faithful ResNet-50 path only

This port brings across the **ResNet-50** path: the query/key encoders, the
momentum queue, and the InfoNCE loss. The captured step 2 (a ViT variant) is
excluded, as in every port.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/13_mocov1` ResNet-50 files (the lab's own model, two-view dataset,
trainer and probe, torch/torchvision only) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` and logs to TensorBoard;
none is needed for a single-process run, so `train_pretrain_mocov1.py` owns a thin
fp32 loop, the device is **resolved** rather than assumed CUDA, TensorBoard is
dropped, and the queue is filled from within the batch. The model's shuffle-BN /
all-gather / broadcast branches are kept but guarded by `dist.is_initialized()`,
so they are inert single-process. The SGD updates only the query encoder; the key
encoder is updated by EMA.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **query ResNet-50 backbone** (`encoder_q.backbone.*`). The
128-d projection head (`encoder_q.proj.*`), the momentum key encoder
(`encoder_k.*`) and the queue (`queue`, `queue_ptr`) are training machinery and
are excluded — the standard SSL convention of probing the backbone, not the head.
The round trip (write it, load it back into a rebuilt model, compare the weights)
is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes the
backbone's 2048-d feature. Images use the deterministic val pipeline (resize +
centre crop, ImageNet normalisation); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule) — the same probe the other
ported methods use, so the number is comparable across them.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a narrow projection, a 32px input, a
  tiny queue, a few fabricated images — runs through `python -m adapter` on a CPU,
  passes `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the paper-target recipe (feature_dim
  128, K 65536, m 0.999, 200 epochs, 224px), a recipe, not a completed run.
- **Not ported:** the ViT step 2.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/13_mocov1-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
closure as `image_gpt`: identical floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/13_mocov1/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/13_mocov1/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/13_mocov1 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/13_mocov1/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/13_mocov1 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
