# 19_byol — step 1 (BYOL ResNet-50 pretext) + linear evaluation

Grill et al., *Bootstrap Your Own Latent: A New Approach to Self-Supervised
Learning* (BYOL), 2020 ([arXiv:2006.07733](https://arxiv.org/abs/2006.07733)).

An **online** network (ResNet-50 backbone → projector → predictor) is trained so
its prediction of one augmented view matches a **target** network's projection of
the other view. The target is an exponential-moving-average (EMA) copy of the
online backbone + projector, with no predictor and no gradient. The loss is a
**symmetric negative cosine similarity** with a stop-gradient on the target —
**no negatives, no queue**. The EMA momentum τ follows a cosine schedule from
0.996 to 1.0. Step 1 is that pretext.

## Scope — the ResNet-50 path only

This port covers the paper-faithful **ResNet-50** step 1. The capture's ViT
variant (`models/vit_byol.py`, step 2) imports `timm`, and is excluded as in
every port — which also drops the `timm`, `tensorboard` and `tqdm` dependencies.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/19_byol` (the lab's own online/target ResNet-50 networks, projector,
predictor, symmetric cosine loss, LARS optimizer, EMA schedule and two-view
dataset, torch/torchvision only) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP autocast and logs
to TensorBoard; none is needed for a single-process run, so `train_step1_byol.py`
owns a thin fp32 loop, the device is **resolved** rather than assumed CUDA, and
AMP / TensorBoard / tqdm are dropped. The EMA target update, the symmetric loss,
LARS, and the cosine LR + EMA-τ schedules are kept faithfully.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the online **ResNet-50 backbone** (`online_encoder.*`), one
2048-d feature per image. The projector, predictor and the entire target network
are training machinery and are excluded, and the round trip (write it, load it
back into a rebuilt model, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation BYOL trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small BYOL ResNet-50, a few
  fabricated images, narrow projector/predictor at 32px — runs through `python -m
  adapter` on a CPU (exercising the EMA target update), passes `contract-test`,
  and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/step1.yaml` is the BYOL recipe (4096/256 MLPs,
  1000 epochs, batch 4096, LARS, EMA τ 0.996→1.0), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/19_byol-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra (the ViT step 2's `timm` is not ported).
`requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are
the hashed closures (the same closure as `14_simclrv1`: identical floors,
identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/19_byol/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py methods/19_byol/configs/step1.yaml \
        --set DATA_ROOT=/path/to/images > resolved.json
    cd methods/19_byol && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py methods/19_byol/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt > eval.json
    cd methods/19_byol && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
