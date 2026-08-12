# 06_rotation_prediction — step 1 (rotation pretext) + linear evaluation

Gidaris, Singh & Komodakis, *Unsupervised Representation Learning by Predicting
Image Rotations*, ICLR 2018 ([arXiv:1803.07728](https://arxiv.org/abs/1803.07728)).

An image is rotated by one of {0°, 90°, 180°, 270°} and an **AlexNet-BN**
predicts which rotation was applied (a 4-class pretext). Step 1 is that pretext.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/6_rotation_prediction` (the lab's own model + dataset following the
official RotNet AlexNet-BN, torch/torchvision only) — no `third_party/`
submodule, the same treatment the other re-implemented methods got. The capture's
step 2 (a ViT variant) is excluded, as in every port, which also drops its `timm`
dependency.

The lab wrapper trains under `DistributedDataParallel` and logs to TensorBoard;
neither is needed for a single-process run, so `train_pretrain_rotation.py` owns a
thin loop, the device is **resolved** rather than assumed CUDA, and TensorBoard
is dropped.

One thing is added in the port: an `AdaptiveAvgPool2d((6, 6))` before the FC
block, so the encoder accepts any input size. At the paper's 224px input the
pool5 map is already 6×6, so the adaptive pool is an **identity** there; it only
matters for the smaller inputs a CPU smoke uses.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **AlexNet-BN encoder** (`encoder.*`) — the five convolutional
blocks and the two FC blocks, giving one 4096-d feature per image. The 4-class
rotation head is pretext machinery and is excluded, and the round trip (write it,
load it back into a rebuilt model, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images are
resized to the encoder's training input size; the probe follows the lab's ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule). The capture's research eval
could select an earlier conv layer for the probe; the port fixes the
representation at the trained encoder's output so the number is a single,
comparable linear probe, as in every other self-contained port here.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small AlexNet-BN, a few fabricated
  images, the four rotations — runs through `python -m adapter` on a CPU, passes
  `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the AlexNet-BN recipe (224px, 50
  epochs), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/06_rotation_prediction-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
closure as `05_jigsaw_puzzle`: identical floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/06_rotation_prediction/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/06_rotation_prediction/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/06_rotation_prediction && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/06_rotation_prediction/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/06_rotation_prediction && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
