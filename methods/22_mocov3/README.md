# 22_mocov3 — step 1 (MoCo v3 ViT pretext) + linear evaluation

Chen, Xie & He, *An Empirical Study of Training Self-Supervised Vision
Transformers* (MoCo v3), 2021
([arXiv:2104.02057](https://arxiv.org/abs/2104.02057)).

Two augmented views feed a **base** encoder (a Vision Transformer → 3-layer MLP
projector) and a **momentum** encoder (an exponential-moving-average copy of the
base + projector, no gradient). A 2-layer MLP **predictor** sits on the base
encoder. The loss is a **symmetric InfoNCE**: each predicted query is contrasted
against the other view's momentum key (temperature-scaled). The EMA momentum
follows a cosine schedule from its base to 1.0. Step 1 is that pretext.

## Scope — the ViT step 1 and the unified ViT-B/16 Step 2

This port covers MoCo v3's **ViT-B/16** step 1 (`configs/pretrain.yaml`, the paper
recipe) **and** the capture's unified **ViT-B/16 Step 2** (`configs/pretrain_vit.yaml`,
`recipe: unified`). MoCo v3 is already ViT-B/16, so the unified Step 2 is the *same*
objective/backbone under the unified recipe: a direct `lr` 6e-4 (step 1 uses
`learning_rate` = base LR × batch/256), a **fixed** EMA momentum (step 1 cosine-
anneals it to 1.0), gradient clipping, and the unified schedule (300 epochs, batch
1024, wd 0.05, 10-epoch warmup) with milestone checkpoints at 100/200/300, each
probed by the same frozen-backbone `linear_eval`. It is selected by an explicit
`recipe: unified` key (absent = the native paper recipe, byte-for-byte unchanged).
Unlike the ResNet ports, MoCo v3's step 1 is
genuinely ViT-based, so **`timm` is a real dependency** here — it supplies the
`VisionTransformer` base class. This is the **first ported method that needs
`timm`**. The ViT is built **from scratch** (no pretrained download), so the run
stays hermetic.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/22_mocov3` (the lab's own MoCo v3, following
[facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3): the ViT
wrapper with fixed 2D sin-cos position embeddings and the official init, the
base/momentum encoders, the MLP projector and predictor, the symmetric InfoNCE
loss and the two-view dataset) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP autocast and logs
to TensorBoard; none is needed for a single-process run, so
`train_pretrain_mocov3.py` owns a thin fp32 loop, the device is **resolved** rather
than assumed CUDA, and AMP / TensorBoard / tqdm are dropped. The EMA
momentum-encoder update, the symmetric InfoNCE loss, the 2D sin-cos position
embedding and the official ViT init are kept faithfully; the `concat_all_gather`
is kept but inert single-process. `img_size` is **threaded** through the model and
dataset (the capture hard-coded timm's 224 default) so the same code runs a small
hermetic CPU smoke at a lower resolution.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the base ViT **trunk** (`base_encoder.*` minus the projector
`base_encoder.head.*`): the patch-embed conv, the class token, the fixed sin-cos
position embedding, the transformer blocks and the final norm — one embed_dim CLS
feature per image (768 for `vit_base`, 384 for `vit_small`). The projector, the
predictor and the entire momentum encoder are training machinery and are excluded,
and the round trip (write it, load it back into a rebuilt model, compare the
weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation MoCo v3 trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small `vit_small` at 32px (a 2×2
  token grid), a narrow projector/predictor, a few fabricated images — runs
  through `python -m adapter` on a CPU (exercising the EMA momentum-encoder
  update), passes `contract-test`, and the encoder round-trip and a determinism
  check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the MoCo v3 recipe (`vit_base`,
  projector 4096→256, predictor, 300 epochs, batch 4096, AdamW, warmup 40), a
  recipe, not a completed run.
- **Exercised (unified Step 2):** a hermetic smoke — `recipe: unified`,
  `arch: vit_base` at 32px, two epochs with `save_at_epochs: [1, 2]` — runs
  through `python -m adapter` on a CPU, writes `encoder.pt` and both
  `encoder_epoch{1,2}.pt` milestones, and a milestone probe passes `contract-test`.
  The full 300-epoch ViT-B/16 recipe has not been run here.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/22_mocov3-pretrain-device.json`).

## Environment

torch / torchvision / numpy / Pillow / PyYAML **plus `timm`** (and its transitive
closure: `huggingface-hub`, `safetensors`, and so on) — the first ported method
that needs `timm`, because MoCo v3's ViT is genuinely `timm`-based.
`requirements.lock.txt` (CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are
the hashed closures.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/22_mocov3/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py --config methods/22_mocov3/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/22_mocov3 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/22_mocov3/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/22_mocov3 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
