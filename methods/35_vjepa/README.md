# 35_vjepa — step 1 (V-JEPA pretext) + linear evaluation

Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations
from Video* (V-JEPA), 2024 ([arXiv:2404.08471](https://arxiv.org/abs/2404.08471)).

V-JEPA is **latent-space prediction**. A **context** encoder sees only the visible
tokens of an input, an EMA **target** encoder encodes the whole input, and a narrow
**predictor** predicts the target's representations at masked positions:

    loss = mean_over_masks( mean|z_pred − layernorm(target)|^loss_exp / loss_exp )
           + reg_coeff · mean(relu(1 − std(z_pred)))

Step 1 is that pretext.

## Scope — the from-scratch image adaptation (step 2), not the video caveat

V-JEPA is a **video** method (pretrained on VideoMix2M). The capture has two paths:
a **pretrain caveat** row (probe the released video ViT-H/16 on ImageNet images —
which the capture itself marks an appendix result, "very low accuracy, not a
main-table success"), and a **step-2** unified-comparison that trains a
V-JEPA-objective **image** ViT-B/16 from scratch on ImageNet at `num_frames=1`.
This port covers the **step-2** path — a genuine comparable row, like `31_dinov3`.

It's a **submodule-import** port: it pins
[`facebookresearch/jepa`](https://github.com/facebookresearch/jepa) as
**`third_party/jepa`** and **imports** its ViT + predictor
(`app.vjepa.utils.init_video_model`), its 3D mask collator
(`src.masks.multiblock3d`) and `apply_masks`. The imported modules are
torch/torchvision-only (measured), so the closure stays torch-only. The `src`/`app`
imports are **lazy** (inside `run`/`build`) so the in-process suite stays
collision-free with other submodule ports that also expose a `src` package.

## Licence — non-commercial research use only

`facebookresearch/jepa` is **CC BY-NC 4.0** (non-commercial). Used **solely for
academic research**; **nothing under it is copied** — the code is a pinned
submodule, imported through PYTHONPATH. This port's own files are original and carry
this repo's licence; they only *reference* the jepa code. See `provenance.json`
(`licence_note`, `upstream`).

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **EMA target encoder** (a bare V-JEPA ViT wrapper, keys
`backbone.*`; there is no separate projection head to strip). The context encoder
and the predictor are training machinery and are not saved. It loads into a rebuilt
V-JEPA encoder whose **mean-pooled tokens** are the probed feature (768 for
ViT-B), and the round trip is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. The probe
follows the lab's shared ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule).

## Milestone checkpoints for the frozen-backbone sweep

This config **is** the unified ViT-B/16 Step 2 already, so there is a single
recipe and no `recipe` selector. To support the Step-2 protocol's 100/200/300
frozen-backbone probe sweep, `train.save_at_epochs` lists the epochs at which the
trainer writes a `checkpoint_epoch_{N}.pth` (in addition to
`checkpoint_latest.pth`); the adapter hands each over as `encoder_epoch{N}.pt`, one
frozen target encoder per milestone, so `linear_eval` can be run at each. An empty
`save_at_epochs` writes only the final `encoder.pt` — the behaviour before this key
existed is unchanged.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny `vit_tiny` at 64px
  (`num_frames=1`, a 4×4 patch grid so the block masks fit), 2 mask configs, a few
  fabricated images — runs through `python -m adapter` on a CPU (exercising the
  context/target forward, the 3D masks, `apply_masks`, the latent-prediction loss
  and the EMA target), passes `contract-test`, and the encoder round-trip and a
  determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain encoder
  over a two-class ImageFolder, passes `contract-test`, writes the comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the recipe (ViT-B/16, 224px, 300
  epochs, batch 1024, AdamW), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/35_vjepa-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML (the same torch closure as `31_dinov3` /
`34_msn`). The `facebookresearch/jepa` code is the submodule under
`third_party/jepa`, imported not installed, so it is not in the lock.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/35_vjepa/requirements.lock.txt -r requirements-tools.lock.txt
    git submodule update --init third_party/jepa   # the V-JEPA code (CC BY-NC 4.0)

## Running

    # step 1: DATA_ROOT contains a train/ subdirectory of images
    python bin/resolve-config.py --config methods/35_vjepa/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/imagenet/train --out resolved.json
    cd methods/35_vjepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/35_vjepa/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/35_vjepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason` and the pinned `upstream`.
