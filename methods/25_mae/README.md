# 25_mae — step 1 (masked-autoencoder pretraining) + linear evaluation

He, Chen, Xie, Li, Dollár & Girshick, *Masked Autoencoders Are Scalable Vision
Learners*, CVPR 2022 ([arXiv:2111.06377](https://arxiv.org/abs/2111.06377)).

MAE learns by reconstruction: 75% of an image's patches are masked, the encoder
sees only the visible tokens, and a lightweight decoder reconstructs the masked
pixels (MSE loss). Step 1 is that pretraining. The representation is the
encoder, global-average-pooled over patch tokens.

## Scope — the ViT-L/16 step 1 and the unified ViT-B/16 Step 2

This port covers MAE's **ViT-L/16** step 1 (`configs/pretrain.yaml`, the paper
recipe) **and** the capture's unified **ViT-B/16 Step 2** (`configs/pretrain_vit.yaml`,
`recipe: unified`): the *same* MAE objective on a ViT-B/16 encoder (`build_mae`
supports it) under the unified recipe — AdamW (betas 0.9/0.95, fixed wd 0.05) with
a **cosine LR schedule + 10-epoch warmup to min_lr**, which the native fixed-LR
Step-1 trainer does not have — with milestone checkpoints at 100/200/300. It is
selected by an explicit `recipe: unified` key (absent = the native ViT-L/16 recipe,
byte-for-byte unchanged). A unified encoder is probed with
`configs/linear_eval_vit.yaml` (the ViT-B/16 dimensions).

## Why this method, and what is new here

**25_mae is a self-contained re-implementation.** The capture's
`methods/25_mae/models/mae_vit.py` is the lab's **own** MAE implementation
(torch-only, following He et al.), not a copy of the CC-BY-NC
`facebookresearch/mae` code, and it trains from scratch with no pretrained
weights. So MAE ports self-contained — the treatment the other re-implemented
methods got — with **no `third_party/` submodule**, no download, and no
noncommercial-licence entanglement.

The lab wrapper trains under `DistributedDataParallel` and logs to TensorBoard;
neither is needed for a single-process run, so `train_pretrain_mae.py` owns a thin,
full-precision loop, the device is **resolved** rather than assumed CUDA, and
TensorBoard is dropped.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the encoder side of the model — the patch embedding, the CLS
token, the encoder blocks and their norm. The decoder is reconstruction
machinery and is excluded, and the round trip (write it, load it back into a
rebuilt model, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: MAE's downstream representation is the
model this port trains (`MAEEncoder`, global-average-pooled patch features — He
et al. Section 4), so the probe number is a genuine, comparable linear probe. The
probe follows the lab's ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny MAE, a few fabricated images
  — runs through `python -m adapter` on a CPU, passes `contract-test`, and the
  encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the four
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the ViT-L/16 recipe (MAE pretrains
  for 1600 epochs); it is a recipe, not a completed run.
- **Exercised (unified Step 2):** a hermetic smoke — `recipe: unified`, the tiny
  vit_base dims, two epochs with `save_at_epochs: [1, 2]` — runs through
  `python -m adapter` on a CPU, writes `encoder.pt` and both `encoder_epoch{1,2}.pt`
  milestones, and a milestone probe passes `contract-test`. The full 300-epoch
  ViT-B/16 recipe has not been run here.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/25_mae-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/25_mae/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/25_mae/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/25_mae && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/25_mae/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/25_mae && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
