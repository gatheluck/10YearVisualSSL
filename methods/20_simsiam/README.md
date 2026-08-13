# 20_simsiam — step 1 and linear evaluation

Chen & He, *Exploring Simple Siamese Representation Learning*, 2020
([arXiv:2011.10566](https://arxiv.org/abs/2011.10566)).

Two augmented views of one image go through a shared encoder; a predictor maps
one side and a stop-gradient blocks the other, and the loss is the negative
cosine similarity, symmetrised. There are no negative pairs and no momentum
encoder — the stop-gradient alone is what stops the representation collapsing.
The backbone is what the rest of the project wants.

## Why this method was third

Measured across the six remaining candidates that carry official-style files,
not chosen by taste. The measurements are in the Capture repository's design
notes; the short version:

- its trainer is the **smallest** of the six (288 lines) and uses **no
  automatic mixed precision at all**, so there is less that behaves
  differently off a GPU
- `setup_dist()` already **returns early when `LOCAL_RANK` is unset**, so the
  single-process path existed and `nccl` is never reached
- its dataset **subclasses `torchvision.datasets.ImageFolder`**, so the
  synthetic tree the first port's tests use works here unchanged
- it has an official-style **Python** linear evaluation, the same shape as the
  first port — so the second stage is a known quantity

Ruled out with evidence: `35_vjepa` has no pretrain trainer at all and downloads
a checkpoint; `04_context_encoder` is 1161 lines of GAN with two models and two
optimisers; `27_ibot` is the heaviest; `17_swav`'s official artefacts are only
a shell script and a config.

## The linear evaluation

The second stage freezes the encoder step 1 produced and fits a linear
classifier on real labels. **These are the numbers this project exists to
compare**, and this is the first method other than the first port to produce
them — so it is the first real test of whether the contract's metric
vocabulary holds across methods.

It reports **three** accuracies, not four: a best top-1 and a final top-1 and
top-5. The first port's evaluation also reports a best top-5; this original
does not, and inventing one would be a number nothing measured.

The handoff needed work. The captured loader rebuilds the whole SimSiam model
from a training checkpoint with `strict=True`, while the contract's artifact
is `encoder.pt` — the backbone alone. Rather than teach the evaluation a
second way to recognise a file, the adapter builds the encoder with
`load_encoder` and hands it in, so one place knows how an encoder is loaded.
`model_type='vit'` now refuses by name, because step 2 was not brought across.

## What was new here

**Which module is the encoder.** SimSiam trains three — a ResNet-50 backbone,
a projector, and a predictor — and only one of them is the representation.
That is not a judgement call: `SimSiamResNet.get_encoder()` returns
`self.backbone`, and the original's own `evaluate_linear_official.py` builds
its frozen feature extractor from exactly that. Read from the source rather
than decided here.

**A metric with nowhere to go.** The trainer reports `z_std`, the standard
deviation of the L2-normalised embeddings — SimSiam's collapse monitor, near
`1/sqrt(dim)` when healthy and near zero when collapsed. It is a real
measurement and it belongs to neither family in the contract's vocabulary, so
its translation table maps it to `None`: kept under the original's own name in
`metrics_raw`, kept out of the comparable block. Inventing a contract name
would offer it for comparison against methods that have no such quantity. This
is the first port to use that path.

## What changed during the port

Recorded in full in `provenance.json`; `models/simsiam_resnet.py` and
`data/simsiam_dataset.py` came across untouched and are pinned by hash.

- **The device is resolved instead of assumed.** The captured trainer called
  `.cuda(local_rank)` on the model and on every batch, so it could not start
  without a GPU. `resolve_device()` picks one, and asking for `cuda` where
  there is none is an error rather than a quiet fall back to the CPU — the two
  are not the same run
- **`main()` is split into `build_parser()` and `run(args, config)`**, and
  `run` returns the epoch loss, the collapse monitor and the epoch count. The
  captured version computed all three and discarded them, so an adapter had
  nothing to record
- **The run is seeded.** Every augmentation is a stock torchvision transform,
  and those draw from torch's generator — measured, not assumed, which is why
  the captured `data/simsiam_dataset.py` needed no change. `random` is seeded
  as well, so a transform added later cannot break reproducibility in silence
- **Step 2 was not brought across.** `train_step2_vit.py` and
  `models/simsiam_vit.py` have no official-style variant in the capture, which
  is the same reason step 2 was left out of the first port

## The configuration

`configs/pretrain.yaml` holds the recipe the captured runs used. Four keys from
the captured config are **deliberately absent** — `arch`, `optimizer`,
`lr_schedule` and `warmup_epochs`. The trainer never reads them; the
architecture, SGD, the cosine decay and the absence of warmup are fixed in its
code. A key that is ignored is a setting claiming an effect it never had, so
the adapter refuses them.

Neither the data path nor the output path is in the file. The capture named an
absolute path on the cluster for both, which is reproducible nowhere else.

## Running it

```bash
python3 bin/launch.py --config methods/20_simsiam/configs/pretrain.yaml --method 20_simsiam --set DATA_ROOT=/path/to/imagenet
```

Or the steps by hand, as the repository README describes for the first method.
The adapter writes `encoder.pt`, `metrics.json` and `run_manifest.json` under
`--out`, with the original's checkpoints, its copy of its config and its
TensorBoard events under `work/`.

## What has not been exercised

- **The full recipe has never been run here.** 100 epochs of ResNet-50 on
  ImageNet-1k needs the GPUs it was written for. The tests run a real training
  step on a handful of synthetic images at 32 pixels
- **No GPU and no multi-process run.** The distributed path is the captured
  one and is untouched, but nothing here has executed it
- **The container definition has never been built** on this machine; it is
  checked by reading, like the other two
- The numbers in `configs/pretrain.yaml` are the recipe, not results. No accuracy
  from this port has been measured against anything
