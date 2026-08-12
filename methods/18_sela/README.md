# 18_sela — step 1 (SeLa ResNet pretext) + linear evaluation

Asano, Rupprecht & Vedaldi, *Self-labelling via simultaneous clustering and
representation learning* (SeLa), ICLR 2020
([arXiv:1911.05371](https://arxiv.org/abs/1911.05371)).

A ResNet backbone with `num_heads` linear **prototype heads** is trained to
predict pseudo-labels. The labels are (re)computed with **Sinkhorn-Knopp optimal
transport**, which forces a balanced (equipartitioned) assignment over K clusters,
and the network is trained with cross-entropy on the resulting hard targets,
averaged over the heads. Step 1 is that self-labelling pretext.

## Scope — the ResNet path only

This port covers the paper-faithful **ResNet** step 1 (the official ResNetV2-50 by
default, or a torchvision ResNet-50 via `arch`). The capture's ViT variant
(`models/vit_sela.py`, step 2) imports `timm`, and is excluded as in every port —
which also drops the `timm`, `tensorboard` and `tqdm` dependencies.

Unlike DeepCluster, the prototype heads are **not reset** each epoch and there is
**no Sobel front-end**; the assignment is balanced by optimal transport rather
than k-means.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/18_sela` (the lab's own ResNet model + prototype heads, the torch
Sinkhorn-Knopp optimal transport, the indexed dataset, the trainer and the probe)
— no `third_party/` submodule. The Sinkhorn is pure torch (no scipy/faiss).

The lab wrapper trains under `DataParallel`/`DistributedDataParallel` with AMP and
logs to TensorBoard; none is needed for a single-process run, so
`train_pretrain_sela.py` owns a thin fp32 loop, the device is **resolved** rather
than assumed CUDA, and TensorBoard/tqdm are dropped. The official `nopts`
label-optimisation schedule, the balanced-label initialisation, and the float64
Sinkhorn iteration are kept faithfully.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the ResNet **backbone** (`backbone.*`), one 2048-d feature per
image. The linear prototype heads (`top_layer.*`) are training machinery and are
excluded, and the round trip (write it, load it back into a rebuilt model, compare
the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation SeLa trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small ResNetV2, a few fabricated
  images, K=8, two heads, a short Sinkhorn schedule at 32px — runs through
  `python -m adapter` on a CPU (exercising the Sinkhorn reassignment path),
  passes `contract-test`, and the encoder round-trip and a determinism check
  pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the SeLa recipe (K 3000, 10 heads,
  λ 25, 400 epochs), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/18_sela-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra (the Sinkhorn is pure torch; the ViT step 2's `timm` is not
ported). `requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA
13.0) are the hashed closures (the same closure as `14_simclrv1`: identical
floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/18_sela/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py --config methods/18_sela/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/18_sela && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/18_sela/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/18_sela && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
