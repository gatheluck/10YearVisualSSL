# 02_vae — step 1

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
python3 bin/launch.py --config methods/02_vae/configs/pretrain_mnist.yaml \
    --method 02_vae --set DATA_ROOT=/path/to/mnist
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
    -r methods/02_vae/requirements.lock.txt -r requirements-tools.lock.txt
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

## Linear evaluation

`linear_eval` fits a single linear layer on the frozen VAE encoder's latent mean
`mu` (`VAE_CNN.get_features` -- the conv encoder, flattened, then `fc_mu`), one
`latent_dim`-d vector per image. It reads the `encoder.pt` a pretrain run wrote
and produces a classifier, not an encoder (the manifest carries an
`encoder_absent_reason`).

Faithful to the capture's VAE eval (`methods/2_vae/evaluate_linear.py`), not the
shared ARSSL probe: inputs are kept in `[0,1]` (no ImageNet mean/std, matching
the VAE's reconstruction training) and `mu` is fed to the linear layer without
mean-centre / L2-normalise. SGD (momentum) under a cosine schedule,
cross-entropy, top-1 and top-5. The loader is **dataset-agnostic**, as the
capture's is: `torchvision.datasets.MNIST` for the shipped MNIST pretrain (10
classes), or an `ImageFolder` (`train/`, `val/`) if an ImageNet-style DATA_ROOT
is given, with the class count inferred from the dataset.

The shipped pretrain is MNIST, so the paired probe is MNIST (10-class). The
capture's own numbers came from probing on ImageNet-1k -- including the stated
MNIST->ImageNet cross-domain transfer -- so the metric name is comparable
(`*_linear_probe_top1/5_accuracy`) but its scale depends on the dataset; see
docs/EVALUATION.md.

## What is not here yet

Step 2 (the unified ViT-B/16 backbone).
