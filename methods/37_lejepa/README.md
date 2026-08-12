# 37_lejepa — step 1 (LeJEPA pretext) + linear evaluation

Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics*, 2025
([arXiv:2511.08544](https://arxiv.org/abs/2511.08544)).

LeJEPA replaces the usual JEPA heuristics (stop-gradient, EMA teachers, predictor
tricks) with one explicit objective. Each image is seen as several augmented
**views**; a ViT backbone + a projection MLP maps each view to a projected
feature; the loss is a convex combination of **SIGReg** and a **cross-view
invariance** loss:

    loss = SIGReg(proj) · λ + invariance(proj) · (1 − λ)

**SIGReg** is an Epps-Pulley Gaussian regularizer: it pushes the projected
features toward an isotropic Gaussian by comparing the empirical characteristic
function of random 1-D **slices** of the batch against the standard-normal
`exp(−t²/2)` on a trapezoidal quadrature grid. Step 1 is that pretext.

## Scope — the ViT step 1 only

This port covers LeJEPA's **ViT-B/16** step 1. The capture's step 2 (ViT
fine-tune) is excluded, as in every port. `timm` supplies the ViT, built **from
scratch** (no pretrained download), so the run stays hermetic. **SIGReg is
reimplemented locally** — the capture's README states *"No external LeJEPA package
is imported at runtime"*, and the only third-party imports are torch / torchvision
/ timm / yaml (measured), so this ports **self-contained**: no `third_party/`
submodule and no downloaded weights.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/37_lejepa` (the lab's own LeJEPA: the encoder = a timm ViT backbone + a
projection MLP, the pure-torch SIGReg objective, the cross-view invariance loss,
the multi-view dataset, the trainer and the probe).

The lab wrapper trains under `DistributedDataParallel` with a bfloat16 autocast,
logs to TensorBoard, and trains an **online linear probe** on *detached* features
for monitoring. None of it touches the backbone — the probe reads detached
features, so it never back-propagates into it — so `train_step1_lejepa.py` drops
all of it: the loop is single-process fp32, the device is **resolved** rather than
assumed CUDA, and AMP / TensorBoard / tqdm / the online probe are dropped; the
elaborate collapse guards are reduced to a finiteness check. SIGReg's cross-rank
averaging is kept as single-process arithmetic (world size one), so the statistic
equals a one-rank distributed run. The weight-decay split (none on norms/biases)
and the AdamW + cosine-with-warmup schedule are kept; `build_backbone` is the
single construction path shared by the trainer and the linear-eval loader;
`img_size` and the ViT dims are **threaded** so a small hermetic CPU smoke runs a
tiny ViT at a lower resolution.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the bare **backbone** (`backbone.*`, the prefix stripped so it
loads straight into a plain timm model) — one num_features feature per image (768
for ViT-B/16). The projection MLP is training machinery and is excluded, and the
round trip (write it, load it back into a rebuilt backbone, compare the weights)
is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation); the
probe follows the lab's shared ARSSL protocol (features cached once, mean-centred
and L2-normalised, a single linear layer trained with SGD under a cosine
schedule), which makes the number comparable across the ported methods. (The
capture's own LeJEPA eval fits a LayerNorm+Linear head; using the shared
single-feature probe instead is a documented deviation, the same as every port.)

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny ViT at 32px, 2 views, a few
  fabricated images — runs through `python -m adapter` on a CPU (exercising the
  multi-view forward, the SIGReg slicing/quadrature and the invariance loss),
  passes `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1 encoder
  over a two-class ImageFolder, passes `contract-test`, writes the comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the LeJEPA recipe (ViT-B/16, 224px,
  4 views, λ=0.02, 100 epochs, batch 1024, AdamW, warmup 10 → cosine), a recipe,
  not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/37_lejepa-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML **plus `timm`** (and its transitive closure)
— LeJEPA's ViT is timm's. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same closure
as `26_simmim` / `22_mocov3`: identical timm resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/37_lejepa/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT contains a train/ subdirectory of images
    python bin/resolve-config.py --config methods/37_lejepa/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/imagenet --out resolved.json
    cd methods/37_lejepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/37_lejepa/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/37_lejepa && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
