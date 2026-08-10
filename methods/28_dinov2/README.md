# 28_dinov2 — linear evaluation only (frozen DINOv2 backbone)

Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023
([arXiv:2304.07193](https://arxiv.org/abs/2304.07193)).

An **eval-only** port — a `linear_eval` stage and **no step 1** — the DINOv2
sibling of `36_franca`. In the capture, DINOv2's "Step 1" is a frozen-backbone
probe: the official pretrained **ViT-g/14** (LVD-142M) is downloaded and a linear
probe is fit on its frozen **CLS token**, because the from-scratch SSL data
(LVD-142M) is not public. That from-scratch pretraining is the excluded step, as
in every port.

## What is probed, and why it's comparable

The representation is DINOv2's pretrained CLS token
(`forward_features(x)["x_norm_clstoken"]`), frozen — a genuine SSL representation,
so the number is comparable across the ported methods (the "pretrained-backbone
reuse" row, like `36_franca`; unlike `var`, which probes a fixed tokeniser). The
probe follows the shared ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule).

## The backbone: pinned code + hash-pinned weights

The model is the pinned upstream `facebookresearch/dinov2` under
`third_party/dinov2`, **imported not copied**, pinned **directly** (no fork). It
is built with the **xformers path disabled** (`XFORMERS_DISABLED=1`) so the forward
is **torch-only** and reproducible on a CPU or any GPU — the giant's SwiGLU falls
back to a torch implementation whose `w12`/`w3` weight keys match the official
fused checkpoint, so the weights load strict (0 missing, 0 unexpected). The model
is built at the checkpoint's native **518px** (a 37×37 position-embedding grid) and
the input is interpolated to the eval resolution.

The weights are a **hash-pinned download** (`provenance.json` → `backbone_artifact`):
the official `dinov2_vitg14_pretrain.pth`, pinned by sha256. **CI never
downloads** — the hermetic smoke builds a random `dinov2_vits14` at a tiny
resolution, so nothing is fetched and only the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a **random** ViT-S/14
  (`ckpt` empty → `pretrained=False`) at 28px over a two-class ImageFolder, runs
  through `python -m adapter`, passes `contract-test`, writes the comparable
  `linear_probe` accuracies (meaningless for a random backbone — only the pipeline
  is verified), writes **no** `encoder.pt`, and records the pinned upstream commit
  in the manifest.
- **Not exercised here:** a real linear-probe number, which needs the official
  4.5 GB `dinov2_vitg14_pretrain.pth` (fetch it with `bin/fetch-weights.py`).
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/28_dinov2-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the eval-only torch-only stack (same closure
as `36_franca`; the pinned upstream adds `tqdm`, imported through `PYTHONPATH` and
never installed, so it is not in the lock). No `timm`, no `xformers` (disabled).

    git submodule update --init third_party/dinov2   # populate the pinned model
    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/28_dinov2/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # fetch the hash-pinned official backbone (a real run; CI never does this)
    python bin/fetch-weights.py --provenance methods/28_dinov2/provenance.json \
        --section backbone_artifact --out /path/to/weights

    # linear eval: DATA_ROOT has train/ and val/; DINOV2_CKPT is the .pth above
    python bin/resolve-config.py methods/28_dinov2/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set DINOV2_CKPT=/path/to/weights/dinov2_vitg14_pretrain.pth > eval.json
    cd methods/28_dinov2 && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The port
writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason` and the pinned `upstream` commit.
