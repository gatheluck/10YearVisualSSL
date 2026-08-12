# 11_cpc — step 1 (visual CPC 2018 pretext) + linear evaluation

van den Oord, Li & Vinyals, *Representation Learning with Contrastive Predictive
Coding*, 2018 ([arXiv:1807.03748](https://arxiv.org/abs/1807.03748)).

An image becomes a grid of overlapping patches. A patch encoder maps each patch
to a z-vector, a **PixelCNN-style masked-convolution context** autoregresses over
the grid, and a log-bilinear **InfoNCE** loss predicts future rows' z-vectors
from the context. Step 1 is that pretext.

## Scope — the paper-faithful `visual_cpc2018` path only

The capture ships **two** step-1 variants. The older local baseline
(`cpc_resnet`: a custom residual encoder with BatchNorm and a column-wise GRU) is
flagged by the capture's own `CPC_STEP1_PAPER_READY_BLOCK.md` as a **structural
protocol mismatch** that "must not be submitted as a paper-ready Step 1 job" (its
ImageNet linear-eval reaches only 6.4% top-1 vs the paper's ~48.7%). It is
**excluded**. This port brings across the **corrected `visual_cpc2018`** path: a
ResNet-v2-101-style no-BN patch encoder, a PixelCNN masked-conv context, and
InfoNCE over future rows of the patch grid. The captured step 2 (a ViT variant)
is excluded, as in every port.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/11_cpc` `visual_cpc2018` files (the lab's own model, patch dataset,
trainer and probe, torch/torchvision only) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP and logs to
TensorBoard; none is needed for a single-process run, so
`train_pretrain_cpc2018.py` owns a thin fp32 loop, the device is **resolved** rather
than assumed CUDA, TensorBoard is dropped, and InfoNCE negatives come from within
the batch (the cross-rank all-gather path is kept behind a flag, off by default).
The dataset's hard-coded 7×7 grid check is relaxed to any grid ≥ 2×2 so a small
hermetic CPU smoke can run; the paper's 7×7 geometry is still what the shipped
config asks for.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **patch encoder** (`encoder.*`) — conv1 + three
pre-activation bottleneck stages (3, 4, 23 blocks, no BatchNorm) with a
projection to `z_dim`. The PixelCNN context (`context.*`) and the InfoNCE
predictors (`predictors.*`) are pretext machinery and are excluded. The round
trip (write it, load it back into a rebuilt model, compare the weights) is
tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes the
grid-averaged z (`avg_z`, `z_dim`-d) — the mean of the encoder's per-patch z over
the patch grid. Patch grids use the deterministic val pipeline; the probe follows
the lab's ARSSL protocol (features cached once, mean-centred and L2-normalised, a
single linear layer trained with SGD under a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a narrow encoder, a 2×2 patch grid,
  one prediction step, a few fabricated images — runs through `python -m adapter`
  on a CPU, passes `contract-test`, and the encoder round-trip and a determinism
  check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the paper-target recipe (1024-d,
  7×7 grid, 5 prediction steps, 200 epochs), a recipe, not a completed run.
- **Not ported:** the deprecated local baseline (`cpc_resnet`).
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/11_cpc-step1-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule and no extra. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
closure as `image_gpt`: identical floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/11_cpc/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/11_cpc/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/11_cpc && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/11_cpc/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/11_cpc && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
