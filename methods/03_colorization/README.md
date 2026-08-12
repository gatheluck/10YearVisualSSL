# 03_colorization — step 1 (colorization pretext) + linear evaluation

Zhang, Isola & Efros, *Colorful Image Colorization*, ECCV 2016
([arXiv:1603.08511](https://arxiv.org/abs/1603.08511)).

An image is converted to CIE **Lab**. The **L** (lightness) channel is the input,
and a VGG-style CNN predicts the **ab** colour channels — quantised into **313**
in-gamut bins — as a per-pixel classification, trained with a (class-rebalanced)
cross-entropy. Step 1 is that pretext.

## Scope — the paper-faithful CNN path only

This port brings across the **CNN** path: the VGG-style encoder/decoder, the
numpy Lab conversion and 313-bin quantisation, and the class-rebalanced loss. The
captured step 2 (a ViT variant) is excluded, as in every port.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/03_colorization` CNN files (the lab's own model, dataset, quantisation,
trainer and probe) — no `third_party/` submodule.

Despite the capture's `requirements.txt` naming **opencv / scikit-image /
scikit-learn**, the lab's own code imports **none** of them: the RGB→Lab
conversion (sRGB → XYZ(D65) → Lab) and the 313-bin ab quantisation are **pure
numpy**, verified against published CIE Lab reference values. So the port keeps
the torch-only closure (the same as `image_gpt`).

The lab wrapper trains under `DistributedDataParallel` with AMP and logs to
TensorBoard; none is needed for a single-process run, so
`train_step1_colorization.py` owns a thin fp32 loop, the device is **resolved**
rather than assumed CUDA, and TensorBoard is dropped.

## The 313 ab-bin constant

Colorization turns ab-regression into a 313-way classification using a fixed
lookup table of in-gamut ab-bin centres. The capture's copy is a **symlink** into
the ABCI filesystem (not in the snapshot), so this port **vendors** the
authoritative constant (`data/pts_in_hull.npy`, 313×2, from
[richzhang/colorization](https://github.com/richzhang/colorization); SHA-256 in
`provenance.json`). There is **no runtime download** — a missing constant errors
loudly.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **CNN encoder trunk** (`encoder.*`, conv1–7). The decoder
(conv8) and the 313-bin head are pretext machinery and are excluded. The lab's
model names its layers flat (`conv1_1 … conv_out`); this port groups them into
`encoder` / `decoder` / `head`, so `encoder.pt` is a clean `encoder.*` prefix,
with the layer shapes, order and computation unchanged. The round trip (write it,
load it back into a rebuilt model, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes the
encoder's **512-d global-average-pooled** feature (the frozen encoder reads the L
channel). The probe follows the lab's shared ARSSL protocol (features cached
once, mean-centred and L2-normalised, a single linear layer trained with SGD
under a cosine schedule) — the same probe the other ported methods use, so the
number is comparable across them.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a 32px crop, one epoch, class
  rebalancing off, a few fabricated images — runs through `python -m adapter` on
  a CPU, passes `contract-test`, and the encoder round-trip and a determinism
  check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the paper-target recipe (313 bins,
  224px crop, 300 epochs, class rebalancing on), a recipe, not a completed run.
- **Not ported:** the ViT step 2.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/03_colorization-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra (the Lab conversion and ab quantisation are numpy, not
opencv/scikit-image). `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
closure as `image_gpt`).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/03_colorization/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/03_colorization/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/03_colorization && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/03_colorization/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/03_colorization && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
