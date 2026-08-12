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

## `encoder.pt` (step 1)

`encoder.pt` is the **representation side** of VAR — the token, class,
positional and level embeddings, and the transformer blocks. The generative head
(`head`, `head_nm`) is excluded, so `encoder.pt` means the same "the
representation network" it means in every other port, and the round trip — write
it, load it back into a rebuilt model, compare the weights — is tested.

## Linear evaluation — a probe on the VQVAE tokeniser, not on `encoder.pt`

CONTRACT section 7 left open which representation a downstream probe reads from a
*generative* model. This port answers it for VAR by **following the lab's own
ARSSL harness** (`methods_step3/ARSSL/src/features/extract.py`): its VAR linear
probe reads the **VQVAE tokeniser** — the tokeniser encoder's continuous feature
map, global-average-pooled to `Cvae` dims — and *not* the VAR transformer that
step 1 trains.

Two consequences, stated plainly so the number is not misread:

- **`linear_eval` reads no `encoder.pt`.** It rebuilds the VQVAE from the config
  and loads the tokeniser weights, so the manifest records `encoder_absent_reason`
  rather than an encoder.
- **The accuracy describes the fixed, pretrained tokeniser, not VAR's learned
  representation.** It is *not* comparable to the other methods' `linear_probe`
  numbers as a measure of *SSL pretraining*, even though it uses the same
  contract slot; it measures the off-the-shelf tokeniser. `docs/EVAL_DOWNLOAD.md`
  records this and the measurement it rests on.

A real number needs the pretrained tokeniser, which is a download pinned by
sha256 in `provenance.json` and fetched with `bin/fetch-weights.py`. The hermetic
smoke builds a **random** VQVAE instead (as step 1 does), so it exercises the
pipeline only; its accuracy is meaningless.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke that builds a **tiny random VQVAE**
  (no download) over a few fabricated images runs through `python -m adapter` on
  a CPU and passes `contract-test`, plus the encoder round-trip. Real training
  reads an ImageFolder and needs the pretrained VQVAE tokeniser (`VQVAE_CKPT`).
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a random-VQVAE
  feature over a two-class ImageFolder, passes `contract-test`, writes the four
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`. The
  accuracy is meaningless without the real tokeniser (above).
- **Not a full run:** `configs/pretrain.yaml` and `configs/linear_eval.yaml` are the
  upstream / ARSSL recipes, not completed runs.
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
    python bin/resolve-config.py --config methods/var/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --set VQVAE_CKPT=/path/to/vae.pth --out resolved.json
    cd methods/var && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`, with
`encoder.pt` and `metrics.json` beside it.

### Linear evaluation

First fetch the pinned, hash-checked VQVAE tokeniser, then run the probe. The
probe reads `DATA_ROOT/train` and `DATA_ROOT/val` (an ImageFolder each).

    # download + verify the tokeniser named in provenance.json
    python bin/fetch-weights.py --provenance methods/var/provenance.json --out .weights/var
    python bin/resolve-config.py --config methods/var/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet --set VQVAE_CKPT=.weights/var/vae_ch160v4096z32.pth \ --out resolved.json
    cd methods/var && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

This stage writes `metrics.json` (the four comparable `linear_probe`
accuracies) and **no** `encoder.pt`; the manifest carries `encoder_absent_reason`.
Read what that number means in the section above before comparing it.
