# mar — step 1 (masked autoregressive pretraining)

Li, Tian, Li, Deng, Zhang, Feng, Cai and He, *Autoregressive Image Generation
without Vector Quantization*, NeurIPS 2024
([arXiv:2406.11838](https://arxiv.org/abs/2406.11838)).

MAR predicts image tokens in a random order in a VAE's continuous latent space,
without a discrete codebook: a masked autoencoder produces a conditioning vector
per token and a small diffusion head (DiffLoss) models the token's continuous
distribution. Step 1 is that pretraining. The representation the rest of the
project wants is the MAE encoder; the decoder and the diffusion head are training
machinery.

## Why this method, and what is new here

**mar is the first port whose model is a pinned git submodule rather than code
copied into the method** — the mechanism the whole remaining inventory needs
(DESIGN sections 1, 2.8, 5.31). The upstream is imported from `third_party/mar`,
never vendored, so the run manifest records `upstream = {repo, commit}` and the
contract can say exactly which code ran.

It is a `submodule+patch` method (DESIGN section 2.8), not adapter-only. The
upstream training forward built two tensors with a hard-coded `.cuda()`
(`sample_orders`, `forward_mae_encoder`), so `MAR.forward` could not run on a
machine with no GPU. The fix is a **two-line, device-preserving patch** carried
in a pinned fork (`gatheluck/mar`, forked from `LTH14/mar`): `.cuda()` becomes
the device of the model's parameters and of the input, which is identical on a
GPU and lets the same forward run on a CPU. `provenance.json` records the fork
commit, its base commit, and the patch.

The upstream training engine (`engine_mar.py`) is **not** used: it imports
`torch_fidelity` and `cv2` on import and calls `torch.cuda.synchronize()`,
so it neither imports nor runs without a GPU. `train_pretrain_mar.py` owns the
cached-latent path of its `train_one_epoch` with the DDP / AMP / EMA / FID
machinery removed, so one training step runs on a CPU or a GPU unchanged.

## `encoder.pt`, and why there is no linear evaluation

`encoder.pt` is the **MAE-encoder side** of MAR (the token projection, the
encoder blocks and norm, the class embedding and the learned buffers
`forward_mae_encoder` reads). The decoder and the diffusion head are excluded, so
`encoder.pt` means the same "the representation network" it means in every other
port, and the round trip — write it, load it back into a rebuilt `mar_base`,
compare the weights — is tested.

**MAR's `linear_eval` is deferred because the lab's evaluation cannot be
faithfully reproduced from what was captured** — this is a measured finding, not
a design preference. The lab's ARSSL harness
(`methods_step3/ARSSL/src/features/extract.py` and `src/run_eval.py`) evaluates
MAR through `from models_mar import mar_base` and `model.forward_encoder(images,
mask_ratio=0.0)`. Neither exists in the pinned upstream (`c6d53f7`): its module
is `models.mar`, and its encoder entry point is
`forward_mae_encoder(x, mask, class_embedding)` over VAE latents, not
`forward_encoder` over raw images. The lab's own mar checkout — the one with
`models_mar` and that method — is not in the Capture snapshot (the inventory
records it as a `0B`, `dirty-without-patch` gitlink), the MAR checkpoint is
HuggingFace-gated, and the harness carries documented silent-fallback bugs
(`DEF-01`, `DEF-02`). So the exact representation the lab probed is not
recoverable here. Rather than invent one and label it "MAR's", this port ships no
`linear_eval` stage; `docs/EVAL_DOWNLOAD.md` records the evidence, and CONTRACT
section 7 is where a chosen representation would be decided.

## What has and has not been exercised

- **Exercised:** a hermetic pretrain smoke on **fabricated cached latents** — no
  VAE, no download — runs through `python -m adapter` on a CPU and passes
  `contract-test`, and the encoder round-trip. Real training uses cached VAE
  latents (the upstream `CachedFolder` `.npz` `moments` format); the ~335 MB VAE
  is needed only to *produce* those latents, not to train on them.
- **Not a full run:** `configs/pretrain.yaml` is the upstream `mar_base` recipe, not
  a completed training run.
- **GPU:** the device resolution and the patched forward are verified on an
  A100; see the device mutation spec (`mutations/mar-pretrain-device.json`).

## Environment

The pinned upstream needs `timm`, `scipy` and `tqdm` beyond this project's
shared torch stack, so mar has its own locks — `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0), a closure over `requirements.txt`
plus the upstream's own dependencies (`requirements.lock.in`). The model itself
is imported from the submodule through `PYTHONPATH`, never installed, so it is
not in the locks and does not trip `verify-environment`.

    git submodule update --init third_party/mar          # populate the upstream
    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/mar/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # DATA_ROOT is the cached-latents root (class subdirs of .npz moments)
    python bin/resolve-config.py --config methods/mar/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/cached --out resolved.json
    cd methods/mar && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`, with
`encoder.pt` and `metrics.json` beside it.
