# 04_context_encoder — step 1 and linear evaluation

Pathak, Krähenbühl, Donahue, Darrell and Efros, *Context Encoders: Feature
Learning by Inpainting*, 2016
([arXiv:1604.07379](https://arxiv.org/abs/1604.07379)).

A square hole is cut from the centre of an image and a convolutional
encoder-decoder is trained to fill it. The reconstruction loss is joined by an
adversarial loss from a discriminator that sees only the real or generated hole,
which sharpens the inpainting. The representation the rest of the project wants
is the encoder plus its 4096-d bottleneck; the decoder and the discriminator are
training machinery.

## Why this method, and what "step 1" means here

The last of the six official-style candidates measured in the Capture
repository's design notes (DESIGN 5.41), and the only one still portable as a
self-contained method. It is the one GAN of the six — two networks, two
optimisers — so it exercises a training shape the other ports do not.

**A labelling caveat worth stating.** In the capture, `model_type='alexnet'` is
Step 1 and `model_type='vit'` is Step 2. This port brings across **both**: the
native AlexNet Step 1 (`configs/pretrain.yaml`, no arch) and the unified
**ViT-B/16 Step 2** (`configs/pretrain_vit.yaml`, `arch: vit`). The Step-2 path
keeps the same centre-hole inpainting task on a ViT-B/16 encoder + a transformer
decoder, always adversarial (two AdamW optimisers, generator + centre-hole
discriminator), AdamW + warmup/cosine to `min_lr`, AMP on CUDA, gradient
clipping; checkpoints at 100/200/300, each probed by the same frozen-backbone
`linear_eval` (the mean patch-token feature, `embed_dim`). The capture's
cluster-only machinery (DDP, bfloat16, the 812-line `utils.py` resume contract) is
dropped, as in every port; the ViT path needs `timm` (imported lazily on
`arch: vit`), and the native AlexNet path is byte-for-byte unchanged.

## The linear evaluation

The second stage freezes the encoder step 1 produced, extracts its 4096-d
bottleneck features once, and fits a single linear classifier on ImageNet
labels. **These are the numbers this project exists to compare**, and this port
adds them for a sixth method.

It reports all four comparable accuracies (best and final top-1, best and final
top-5). The handoff is the shape the earlier two-stage ports use: the adapter
builds the encoder with `load_encoder` from `encoder.pt` and hands it in, rather
than letting the evaluation rebuild the whole model from a training checkpoint.
The probe is arch-aware: the AlexNet path reads the 4096-d bottleneck
(`model(x) -> (_, features)`), the ViT path the mean patch-token feature
(`model.get_features(x)`, `embed_dim`). The stand-alone CLI path can only rebuild
the fixed AlexNet (the ViT needs its config dimensions; the official Caffe
features stay step 2), so it refuses a bare `model_type='vit'` by name.

## What was new here

**A GAN, ported as step 1.** The generator is the encoder-decoder; when
adversarial training is on, a centre-hole `Discriminator` is trained alongside
it with its own Adam optimiser (BCE-with-logits on real vs generated holes). The
reconstruction and adversarial losses are real measurements that belong to no
family in the contract's vocabulary, so their translation table maps them to
`None`: kept under their own names in `metrics_raw`, kept out of the comparable
block.

**`encoder.pt` is the encoder and the bottleneck.** The original's own linear
evaluation reads the representation as `model(x) -> (_, features)` — the
bottleneck output — so `encoder.pt` holds both the conv encoder (`encoder.*`)
and the bottleneck (`fc.*`); the decoder (`decoder_fc`, `decoder`) and the
discriminator are left out.

## What changed during the port

Recorded in full in `provenance.json`. **Nothing came across byte-for-byte:**
the capture interleaves step 1 and step 2 in single files, so each ported file
was rewritten to extract a clean step 1, and each file's captured digest is
recorded in `provenance.json` rather than pinned.

- **The device is resolved instead of assumed.** The captured trainer and
  evaluation sent the model and every batch to CUDA with `.cuda(args.gpu)`, so
  neither could start without a GPU. `resolve_device()` picks one; asking for
  `cuda` where there is none is refused rather than served a CPU in silence
- **`main()` is split into `build_parser()`/`run(...)`** in both the trainer and
  the evaluation, returning the metrics the captured versions discarded
- **Single process.** The captured DDP and the 812-line `utils.py` resume
  contract were not brought across; the native pretrain loop is plain fp32, and
  the ViT Step-2 loop uses AMP only on CUDA (fp32 on CPU) so both run unchanged
  on a CPU or a GPU
- **The unified ViT-B/16 Step 2 was added** (`models/vit_context_encoder.py`,
  `train_pretrain_vit.py`, `configs/pretrain_vit.yaml`). The encoder is built as
  a `VisionTransformer` (configurable, so a tiny model runs a CPU smoke; the
  shipped config is ViT-B/16) under `self.encoder`, with a transformer decoder
  predicting the hole patches; the shared `Discriminator` (sized to the hole) is
  reused. The official Caffe feature extractor and the capture's bfloat16 remain
  out. `timm` is imported only on the `arch: vit` path

## The configuration

`configs/pretrain.yaml` holds the recipe the captured run used. `data_root` is the
parent of `train/`/`val/`, not the `train/` directory — the loader joins the
split itself. The output path is not a config key; the contract fixes it at
`--out`.

## Running it

```bash
python3 bin/launch.py --config methods/04_context_encoder/configs/pretrain.yaml --method 04_context_encoder --set DATA_ROOT=/path/to/imagenet
```

Then the linear evaluation, on the `encoder.pt` the first stage wrote:

```bash
python3 bin/launch.py --config methods/04_context_encoder/configs/linear_eval.yaml --method 04_context_encoder --set DATA_ROOT=/path/to/imagenet --set ENCODER=runs/<pretrain-run>/out/encoder.pt
```

## What has and has not been exercised

- **A real training step ran on the GPU.** On an NVIDIA A100 (driver CUDA 13.0)
  step 1 completes a training step (generator + discriminator) and writes a
  loadable `encoder.pt` — the test
  `test_a_real_run_on_cuda_produces_a_loadable_encoder`, on a handful of
  synthetic images
- **The linear evaluation runs, but on the CPU.** Its smoke test uses
  `device: cpu` (through the same `resolve_device` path), and a separate test
  checks it refuses `cuda` when no GPU is visible
- **The full recipe has never been run.** 300 epochs of AlexNet inpainting on
  ImageNet needs the GPUs it was written for
- **The ViT Step-2 path ran a hermetic CPU smoke.** A tiny ViT (16-d embed, one
  block, 32px image, 16px hole) trains two epochs with `save_at_epochs: [1, 2]`
  through `python -m adapter`, writes `encoder.pt` and both
  `encoder_epoch{1,2}.pt` milestones (each an adversarial generator+discriminator
  step), and a milestone probe passes `contract-test`. The full 300-epoch
  ViT-B/16 recipe (bfloat16, DDP) has not been run here
- **No multi-process run.** The captured DDP path was not brought across; AMP is
  used on the ViT Step-2 path on CUDA only
- **The container definition has never been built** on this machine
- The numbers in the configs are the recipe, not results; no accuracy from this
  port has been measured against anything
