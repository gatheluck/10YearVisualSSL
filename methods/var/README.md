# var — step 1 (next-scale autoregressive pretraining)

Tian, Jiang, Yuan, Peng & Wang, *Visual Autoregressive Modeling: Scalable Image
Generation via Next-Scale Prediction*, NeurIPS 2024
([arXiv:2404.02905](https://arxiv.org/abs/2404.02905)).

VAR reframes autoregressive image generation as **next-scale prediction**: a
VQVAE tokenises an image into a pyramid of code maps at increasing resolutions,
and a transformer predicts each scale from the coarser ones, class-conditioned.
Step 1 is that pretraining. The representation the rest of the project wants is
the transformer; the output head is generation machinery.

## Why this method, and what is new here

**var is the second port on the `third_party/` submodule mechanism, and the
first `submodule+adapter` one** — the more common case in the inventory (24
adapter to 7 patch). Unlike mar, whose model forced a device patch in a fork,
VAR's model runs on a CPU or a GPU **unmodified**: it has no hardcoded device,
and the only `torch.cuda.amp.autocast` on its forward path is `enabled=False` (a
no-op on a CPU). So the upstream is pinned **directly**
(`github.com/FoundationVision/VAR`, MIT) — no fork — and imported from
`third_party/var`, never copied. The run manifest records
`upstream = {repo, commit}`.

The upstream training script is a DDP trainer wired to `dist.py` and a bfloat16
AMP context; none of that is needed for a single-process, device-resolved run,
so `train_step1_var.py` owns a thin loop over the model's own forward. Flash and
fused attention are forced off, so a run does not depend on whether
flash-attention happens to be installed.

## `encoder.pt`, and why there is no linear evaluation yet

`encoder.pt` is the **representation side** of VAR — the token, class,
positional and level embeddings, and the transformer blocks. The generative head
(`head`, `head_nm`) is excluded, so `encoder.pt` means the same "the
representation network" it means in every other port, and the round trip — write
it, load it back into a rebuilt model, compare the weights — is tested.

Which representation a downstream probe should read from a *generative* model is
a deliberately deferred question (CONTRACT section 7), the same one mar raises.
So this port ships no `linear_eval` stage; its numbers are pretext only.

## What has and has not been exercised

- **Exercised:** a hermetic step-1 smoke that builds a **tiny random VQVAE** (no
  download) over a few fabricated images runs through `python -m adapter` on a
  CPU and passes `contract-test`, plus the encoder round-trip. Real training
  reads an ImageFolder and needs the pretrained VQVAE tokeniser (`VQVAE_CKPT`).
- **Not a full run:** `configs/step1.yaml` is the upstream VAR-d16 recipe, not a
  completed training run.
- **GPU:** the device resolution is verified on an A100; see the device mutation
  spec (`mutations/var-step1-device.json`).

## Environment

The pinned upstream needs `huggingface_hub` beyond this project's shared torch
stack, so var has its own locks — `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0), a closure over `requirements.txt`
plus that dependency (`requirements.lock.in`). The model is imported from the
submodule through `PYTHONPATH`, never installed, so it is not in the locks and
does not trip `verify-environment`.

    git submodule update --init third_party/var           # populate the upstream
    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/var/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # DATA_ROOT is an ImageFolder of images; VQVAE_CKPT is the pretrained tokeniser
    python bin/resolve-config.py methods/var/configs/step1.yaml \
        --set DATA_ROOT=/path/to/images --set VQVAE_CKPT=/path/to/vae.pth > resolved.json
    cd methods/var && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`, with
`encoder.pt` and `metrics.json` beside it.
