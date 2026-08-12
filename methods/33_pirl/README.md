# 33_pirl — step 1 (PIRL ResNet-50 pretext) + linear evaluation

Misra & van der Maaten, *Self-Supervised Learning of Pretext-Invariant
Representations* (PIRL), CVPR 2020
([arXiv:1912.01991](https://arxiv.org/abs/1912.01991)).

PIRL learns representations that are **invariant** to a pretext transformation
(here, a jigsaw shuffle). A ResNet-50 trunk encodes an image and a
jigsaw-shuffled view of the same image (nine patches through the shared trunk,
projected and concatenated). Both are contrasted against a momentum-updated
**memory bank** (one row per training image) with an **NCE** cross-entropy, and
the loss is a convex combination of the image-NCE and the jigsaw-NCE. The bank is
updated with the image representation each step. Step 1 is that pretext.

## Scope — the ResNet-50 step 1 only

This port covers PIRL's ResNet-50 step 1. The capture's step 2 (ViT) is excluded,
as in every port — so no `timm`; the port is **torch-only** (torchvision supplies
ResNet-50).

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/33_pirl` (the lab's own PIRL: the ResNet-50 model with image + jigsaw
projection heads, the memory-bank NCE loss, the jigsaw dataset, the trainer and
the probe) — no `third_party/` submodule. It is the repo's **third memory-bank
method**, sharing the instance-discrimination bank shape with `10_inst_disc` and
`12_cmc`, plus the jigsaw-invariance branch.

The lab wrapper trains under `DistributedDataParallel` and logs to TensorBoard;
none is needed for a single-process run, so `train_pretrain_pirl.py` owns a thin fp32
loop, the device is **resolved** rather than assumed CUDA, and TensorBoard / tqdm
are dropped. The ResNet-50 model, the memory-bank NCE (with its EMA bank update),
the jigsaw view (resize → crop → 3×3 cells → per-cell random patch → colour jitter
→ shuffle) and the convex-combination loss are kept faithfully; the `DataLoader`
gets a seeded generator, and the image / jigsaw geometry is threaded so a small
hermetic CPU smoke runs at a lower resolution.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the ResNet-50 **trunk** (`encoder.*`), one 2048-d avg-pooled
feature per image. The image projector, the jigsaw projector and the **memory
bank** are excluded — the bank is a `register_buffer` in the loss module, not the
model, so it is never in `model.state_dict` (the instance-discrimination
convention). The round trip (write it, load it back into a rebuilt model, compare
the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation PIRL trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a ResNet-50 at 32px images / 16px
  jigsaw patches (3×3 grid), a narrow feature_dim, few negatives, a few fabricated
  images — runs through `python -m adapter` on a CPU (exercising the jigsaw view,
  the memory-bank NCE, the bank initialisation and the bank update), passes
  `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the PIRL recipe (ResNet-50, 224px,
  feature_dim 128, 32000 negatives, 800 epochs, SGD step decay), a recipe, not a
  completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/33_pirl-pretrain-device.json`).

## Environment

torch / torchvision / numpy / Pillow / PyYAML — the self-contained torch-only
stack (no submodule, no `timm`; the ViT step 2 is not ported). `requirements.lock.txt`
(CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the
same closure as `05_jigsaw_puzzle`: identical floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/33_pirl/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py --config methods/33_pirl/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/33_pirl && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/33_pirl/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/33_pirl && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
