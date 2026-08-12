# 05_jigsaw_puzzle — step 1 (jigsaw pretext) + linear evaluation

Noroozi & Favaro, *Unsupervised Learning of Visual Representations by Solving
Jigsaw Puzzles*, ECCV 2016 ([arXiv:1603.09246](https://arxiv.org/abs/1603.09246)).

An image is cut into a 3×3 grid of tiles, the tiles are reordered by one of a
fixed set of high-Hamming-distance permutations, and a **Context-Free Network**
(a siamese AlexNet whose FC layers are 1×1 convolutions, shared over the 9 tiles)
predicts which permutation was applied. Step 1 is that pretext.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/5_jigsaw_puzzle` (the lab's own model + dataset, torch/torchvision only)
— no `third_party/` submodule, the same treatment the other re-implemented
methods got. The capture's step 2 (a ViT variant) is excluded, as in every port,
which also drops its `timm` dependency.

The lab wrapper trains under `DistributedDataParallel` and logs to TensorBoard;
neither is needed for a single-process run, so `train_pretrain_jigsaw.py` owns a
thin loop, the device is **resolved** rather than assumed CUDA, and TensorBoard
is dropped.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the shared **CFN encoder** (`encoder.*`) — the AlexNet conv stack,
the 1×1-conv CFN layers and the adaptive pool, giving one 512-d feature per
image. The permutation classifier is pretext machinery and is excluded, and the
round trip (write it, load it back into a rebuilt model, compare the weights) is
tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images are
resized to the encoder's tile size; the probe follows the lab's ARSSL protocol
(features cached once, mean-centred and L2-normalised, a single linear layer
trained with SGD under a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny CFN, a few fabricated images,
  a 4-permutation puzzle — runs through `python -m adapter` on a CPU, passes
  `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the AlexNet/CFN recipe (1000
  permutations, 300 epochs), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/05_jigsaw_puzzle-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/05_jigsaw_puzzle/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/05_jigsaw_puzzle/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/05_jigsaw_puzzle && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/05_jigsaw_puzzle/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/05_jigsaw_puzzle && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
