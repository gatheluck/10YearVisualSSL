# 08_split_brain — step 1 (Split-Brain cross-channel pretext) + linear evaluation

Zhang, Isola & Efros, *Split-Brain Autoencoders: Unsupervised Learning by
Cross-Channel Prediction*, CVPR 2017
([arXiv:1611.09842](https://arxiv.org/abs/1611.09842)).

An image is converted to CIE **Lab** and split into its **L** and **ab**
channels. Two cross-channel AlexNet branches predict one from the other: **net1**
maps L → the quantised ab channels (313 bins), **net2** maps ab → the quantised L
channel (50 bins), each a per-pixel classification. Step 1 is that pretext.

## Scope — the AlexNet path only

This port brings across the **AlexNet** path: the two cross-channel branches, the
numpy Lab conversion and L/ab quantisation, and the summed cross-entropy loss.
The captured step 2 (a channel-split ViT, timm) is excluded, as in every port —
so this port has **no timm dependency**.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/8_split_brain` AlexNet files (the lab's own model, dataset, trainer and
probe) — no `third_party/` submodule.

The capture's dataset imports `scipy.spatial.cKDTree` and `skimage.color`, but
its **own comment** states the released ab target *is* NumPy argmin (the cKDTree
path only accelerates it and is corrected to match at Voronoi boundaries). So the
port does the RGB→Lab conversion (sRGB → XYZ(D65) → Lab, verified against
published CIE Lab values) and the L/ab quantisation in **pure numpy**, depending
on neither scipy nor scikit-image — the torch-only closure holds.

The capture ships **no AlexNet pretrain recipe** (its `train.py` trains the ViT
step 2 under DistributedDataParallel + AdamW with a canonical contract), so
`train_pretrain_split_brain.py` owns a thin single-process fp32 loop with a plain
Adam optimiser (its knobs exposed in the config). The device is **resolved**
rather than assumed CUDA; TensorBoard is dropped.

## The 313 ab-bin constant

The ab channels are quantised against a fixed 313-entry codebook
(`pts_in_hull.npy`). The capture pins its sha256 (`AB_CODEBOOK_SHA256`) and its
copy is a **symlink into the cluster's LIVE_ROOT** (not in the snapshot), so this
port **vendors** the authoritative constant (the same file
[richzhang/colorization](https://github.com/richzhang/colorization) ships, which
`03_colorization` also uses), loaded with a **sha256 check** — no runtime
download.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **two branch encoders** (`net1.encoder.*` /
`net2.encoder.*`). The two deconv decoders are pretext machinery and are
excluded. The round trip (write it, load it back into a rebuilt model, compare
the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes
`extract_features(l, ab)` — both encoders' spatially-averaged features
concatenated (256 + 256 = **512-d**). The probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule) — the same probe the other ported
methods use, so the number is comparable across them.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a 32px crop, one epoch, a few
  fabricated images — runs through `python -m adapter` on a CPU, passes
  `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is a recipe (224px crop, 200 epochs),
  not a completed run; its optimiser is the port's single-process choice (the
  capture ships no AlexNet pretrain recipe).
- **Not ported:** the channel-split ViT step 2.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/08_split_brain-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra (the Lab conversion and quantisation are numpy, not
scipy/scikit-image). `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
closure as `image_gpt`).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/08_split_brain/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/08_split_brain/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/08_split_brain && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/08_split_brain/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/08_split_brain && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
