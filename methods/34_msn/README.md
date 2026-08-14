# 34_msn — step 1 (MSN pretext) + unified deit_base/16 step 2 + linear evaluation

Assran et al., *Masked Siamese Networks for Label-Efficient Learning*, 2022
([arXiv:2204.07141](https://arxiv.org/abs/2204.07141)).

MSN is a **masked siamese** method. An **anchor** ViT sees several augmented,
patch-dropped multi-crop views (`imgs[1:]`) and an EMA **target** ViT sees one
un-dropped random view (`imgs[0]`); both are mapped to a set of learnable
**prototypes** via a soft-nearest-neighbour classifier, and the loss is the MSN
cross-entropy plus a **me-max** entropy regulariser:

    loss = ploss + memax_weight · me_max + ent_weight · ent

Step 1 is that pretext.

## Scope — a submodule-import port of the official MSN

The capture's MSN is a **thin wrapper** that drives the official
[`facebookresearch/msn`](https://github.com/facebookresearch/msn) repo (its
`main.py` under `DistributedDataParallel` + submitit, its `linear_eval.py` under
cyanure) — it ships no model or loss of its own. This port pins
`facebookresearch/msn` as the submodule **`third_party/msn`** and **imports** its
ViT (`src/deit.py`), MSN loss (`src/losses.py`) and optimiser
(`src/msn_train.init_opt`), running them in a **single-process** trainer. Measured:
the imported `src` modules use only torch / torchvision / numpy / PIL — no
apex / opencv / submitit / cyanure — so the closure stays torch-only.

Two things are rewritten in the port (documented in `provenance.json`):
- the **multi-view augmentation** is reimplemented (`data/msn_data.py`) with
  torchvision's `GaussianBlur`, because the upstream `make_transforms` passes a
  `torch.Tensor` radius to `PIL.ImageFilter.GaussianBlur`, which the pinned
  Pillow (12.x) rejects. The recipe (rand + focal crops, colour distortion,
  grayscale, blur, ImageNet norm) is kept faithfully;
- the linear probe is the shared **ARSSL** probe rather than the official cyanure
  logistic probe (the same deviation as every port).

## Step 2 (unified deit_base/16), added additively

The capture's Step 2 plugs the **same** MSN objective into the unified
**deit_base/16** backbone (embed_dim 768, depth 12, heads 12; 300 epochs, `lr`
6e-4, fixed weight decay 0.05, checkpointed at 100/200/300). It is selected by a
`recipe: unified` key in `train`; **absent `recipe` is the native deit_small
step-1 path, unchanged**, and the unified recipe's only extra key over the native
set is `save_at_epochs` (the milestone knob), refused by name on the native path.

**No separate trainer was needed — the native `train_pretrain_msn.py` already
implements the unified recipe.** The official `src.msn_train.init_opt` takes the
`lr` directly (the capture baked `lr = 1.5e-4 × 1024/256 = 6e-4` into the config,
so there is nothing to rescale), the arch is built from the config dims
(`deit_base` is just `embed_dim: 768`, `num_heads: 12`), and its cosine
weight-decay is **constant** when `weight_decay == final_weight_decay`. The only
addition is a milestone checkpoint: the trainer writes `checkpoint_epoch_{N}.pth`
at each `training.save_at_epochs`, a guarded block that is empty for the native
config (which never sets that key), so the native behaviour is byte-for-byte
unchanged — the DRY choice over duplicating the loop into a second file. The
adapter writes `encoder.pt` (the final anchor trunk) plus
`encoder_epoch{100,200,300}.pt`, one frozen backbone per milestone; probe a
unified encoder with `configs/linear_eval_vit.yaml` (deit_base dims).

## Licence — non-commercial research use only

The `facebookresearch/msn` code is **CC BY-NC 4.0** (Attribution-NonCommercial),
which permits **non-commercial research** use only. This port is used solely for
academic research. **Nothing under that licence is copied** into this repository —
the code is a pinned git submodule, imported through PYTHONPATH. This port's own
files (trainer, augmentation, eval, adapter, configs, tests) are original and carry
this repository's licence; they only *reference* the msn code. See
`provenance.json` (`licence_note`, `upstream`).

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **anchor ViT trunk** (`patch_embed`, `cls_token`, `pos_embed`,
`blocks`, `norm`) — one embed_dim CLS feature per image (384 for deit_small). The
projection head (`fc.*`) is training machinery and is excluded; it loads into a
bare ViT (`build_msn_backbone`, the head left as the default Identity) whose
`forward_features` returns the CLS token, and the round trip is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. The probe
follows the lab's shared ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny ViT at 32px rand / 16px focal
  (a 2×2 patch grid, embed_dim 32, 2 blocks), 1 rand + 2 focal views, 8
  prototypes, a few fabricated images — runs through `python -m adapter` on a CPU
  (exercising the multi-view forward, the patch-drop masking, the Sinkhorn MSN
  loss, me-max and the EMA target), passes `contract-test`, and the encoder
  round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain encoder
  over a two-class ImageFolder, passes `contract-test`, writes the comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the official deit_small MSN recipe
  (ViT-S/16, 224px, 1 rand + 10 focal, 1024 prototypes, 800 epochs, AdamW), a
  recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/34_msn-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML (the same torch closure as `31_dinov3`). The
`facebookresearch/msn` code is the submodule under `third_party/msn`, imported not
installed, so it is not in the lock.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/34_msn/requirements.lock.txt -r requirements-tools.lock.txt
    git submodule update --init third_party/msn   # the MSN code (CC BY-NC 4.0)

## Running

    # step 1: DATA_ROOT contains a train/ subdirectory of images
    python bin/resolve-config.py --config methods/34_msn/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/imagenet/train --out resolved.json
    cd methods/34_msn && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/34_msn/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/34_msn && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason` and the pinned `upstream`.
