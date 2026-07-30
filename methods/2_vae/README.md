# 2_vae — step 1

Kingma & Welling, *Auto-Encoding Variational Bayes*, 2013.

An encoder maps an image to a distribution over a latent code, a decoder maps
a sample back, and the model is trained to reconstruct while keeping the
latent close to a unit Gaussian. The encoder is what the rest of the project
wants.

## Why this method was second

Measured across all 37 methods, not chosen by taste. **It is the only one that
uses MNIST** — the other 36 are ImageNet-only. At 28×28 with batch 100 and
Adam at 1e-3, it trains to completion on a CPU, so this is the first port
whose tests run a **real training run** rather than two steps on noise.

The intended second pilot, `VideoGen`(LTX-2), needs CUDA > 12.7 and a 22B
checkpoint: it cannot start without a GPU. It is deferred, not dropped.

## What was new here

**The output path lived inside the config.** The captured config carries an
absolute path on the cluster as `output.checkpoint_dir`, and the original
writes its checkpoints and TensorBoard events there. The contract says an
adapter writes only under `--out`, and a machine's path in a published config
is reproducible nowhere else — so the shipped config has **no output path at
all** and the adapter supplies one under `--out`. A config that tries to set
it is **refused**, not quietly overridden: overriding would leave a config
claiming a location that was not used.

**The data is not an `ImageFolder`.** The loader picks
`torchvision.datasets.MNIST` by looking for an `MNIST/` directory, and creates
it with `download=False`. The tests fabricate the IDX files rather than reach
the network.

**`encoder.pt` is one half of a generative model.** It carries `encoder.*`
plus the projections to the latent mean and log variance (`fc_mu`,
`fc_logvar`) — without those the encoder cannot produce a code. The decoder
is left out, so `encoder.pt` means the same thing here as in the first method.

## Settings, and where each comes from

From the paper's MNIST experiments, as the capture recorded them:

| Setting | Value |
|---|---|
| image size | 28×28, unresized |
| augmentation | none |
| optimizer | Adam, constant 1e-3, no weight decay |
| beta | 1.0 (standard VAE) |
| latent dim | 20 (the paper tested 2, 10, 20, 50) |
| batch size | 100 |

## Running it

```bash
python3 bin/launch.py --config methods/2_vae/configs/step1_mnist.yaml \
    --method 2_vae --set DATA_ROOT=/path/to/mnist
```

`DATA_ROOT` is a directory containing `MNIST/` in the layout torchvision
expects. Nothing is downloaded.

## The environment

Three files, as for every method: `requirements.txt` (which packages),
`requirements.lock.txt` (the full closure, pinned and hashed for linux
x86_64, linux aarch64 and macOS arm64), and `Dockerfile`.

```bash
pip install --require-hashes \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r methods/2_vae/requirements.lock.txt -r requirements-tools.lock.txt
```

**This closure is larger than the first method's, and tensorboard is why.**
The captured trainer writes event files through `torch.utils.tensorboard`,
which brings absl-py, grpcio, Markdown, protobuf, Werkzeug and two others with
it — measured, 14 packages become 22. Keeping the original faithful was judged
worth that; the alternative was making its logging conditional, which is a
behaviour change made to keep a lock file small.

Two consequences worth stating:

- `Pillow` is **not** declared here. torchvision needs it so the lock contains
  it, but nothing in this method imports it. It was copied over from the
  captured requirements and the guard caught it
- **TensorBoard event files embed a wall-clock timestamp**, so they differ
  between runs. The reproducibility claim is about the artifacts that
  constitute the result — `encoder.pt` and `metrics.json` — and the tests
  compare those. Every file is still listed in the manifest with its hash

## What changed during the port

`models/vae_cnn.py` and `data/vae_dataset.py` **came across untouched**, with
their digests pinned in `provenance.json`. Otherwise:

1. `main()` split into `parse_args()` and `run(args, config=None)`. The config
   may arrive as a value so the adapter can supply the output path rather than
   rewrite a file
2. `make_deterministic()` added, as in the first method
3. `run()` returns the final loss, so a caller need not re-derive it from the
   log
4. The seeds for Python's `random` and NumPy are set. **Not currently
   load-bearing**: measured, nothing in this method's path draws from either,
   and the loader shuffles with torch's generator. They are set defensively
   and because the first method genuinely needed them
5. `__init__.py`, `models/__init__.py` and `data/__init__.py` were rewritten;
   they re-exported `VAE_ViT`, which belongs to step 2 and is not here

## What is not here yet

Step 2 (ViT) and the linear evaluation.
