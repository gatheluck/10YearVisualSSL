# 17_swav — step 1 (SwAV ResNet-50 pretext) + linear evaluation

Caron, Misra, Mairal, Goyal, Bojanowski and Joulin, *Unsupervised Learning of
Visual Features by Contrasting Cluster Assignments*, 2020
([arXiv:2006.09882](https://arxiv.org/abs/2006.09882)).

Rather than comparing two views directly, SwAV assigns each view to a set of
learned prototypes and makes one view predict the other's assignment. A
Sinkhorn-Knopp normalisation keeps the prototypes evenly used, which is what
stops the representation collapsing. The encoder is what the rest of the
project wants; the prototypes are training machinery.

## Scope — the ResNet-50 path and the unified ViT-B/16 Step 2

This port covers the paper-faithful **ResNet-50** step 1 (`configs/pretrain.yaml`,
LARC-SGD, multi-crop, the online Sinkhorn-Knopp assignment) **and** the capture's
unified **ViT-B/16 Step 2** (`configs/pretrain_vit.yaml`, `arch: vit`): the same
ViT-B/16 backbone every method shares, plus a projection head
(`Linear(768, 2048) -> BN -> ReLU -> Linear(2048, 128)`, L2-normalised) and 3000
learnable **prototypes**, trained from scratch with the multi-crop
swapped-assignment objective — 2 global (224) + 6 local (96) crops, online
Sinkhorn-Knopp for the assignments, cross-view swapped prediction. Optimiser
AdamW (betas 0.9, 0.999) with a 10-epoch warmup then cosine decay to `min_lr`;
checkpoints at 100/200/300 epochs, each probed by the same frozen-backbone
`linear_eval` (the CLS embedding, `embed_dim`-d). The ViT path needs `timm`
(imported lazily); the native ResNet-50 path is byte-for-byte unchanged.

## The linear evaluation

The second stage freezes the encoder step 1 produced and fits a linear
classifier on real labels. **These are the numbers this project exists to
compare**, and this port adds them for a fifth method.

It reports **three** accuracies: a best top-1 and a final top-1 and top-5. The
handoff is the same shape the other two-stage ports use — the adapter builds
the encoder with `load_encoder` from `encoder.pt` (the backbone alone) and
hands it in, rather than letting the evaluation rebuild the whole model from a
training checkpoint. `load_encoder` builds a ResNet-50 by default and, for a
config carrying `arch: vit`, the ViT-B/16 via `build_vit_swav`, so the same probe
reads either backbone.

## What was new here

**A configuration made of lists.** Multi-crop is described by four parallel
lists — the crop sizes, how many of each, and the scale bounds — and the
loader asserts they are the same length. That assertion arrives as a bare
`AssertionError` from inside the dataset, saying nothing about which setting
was wrong, so the adapter refuses mismatched lists by name instead.

**Settings that are optional in the original.** The trainer reads a dozen keys
with `cfg.get(...)` and a default behind them. Left out of the contract's
config, a run would be described without its Sinkhorn epsilon, its warmup or
its prototype freezing — filled in by whatever that version of the code
happened to default to. They are all declared, because the resolved config has
to say what ran.

## What changed during the port

Recorded in full in `provenance.json`. `models/resnet_swav.py`,
`distributed_sinkhorn.py` and `checkpoint_utils.py` came across untouched and
are pinned by hash.

- **the loader could not run on one process at all.** `get_swav_dataloader`
  built a `DistributedSampler` unconditionally, which needs an initialised
  process group; the run raised *"Default process group has not been
  initialized"*. The sampler is now conditional, the way every other ported
  method's loader already had it
- the device is resolved instead of assumed; the captured trainer called
  `.cuda(local_rank)` on the model and on every crop
- `main()` split into `build_parser()` and `run(args, config)`, which returns
  the epoch loss and the epoch count. The captured version computed them and
  discarded them
- `models/__init__.py` advertised the step 2 ViT through a lazy `__getattr__`
  over a module that was not brought across — a name the package could not
  import. It is now a real lazy accessor: `build_vit_swav` builds the ported
  `models/vit_swav.py` (needs `timm`), so the promise is kept for the unified
  ViT-B/16 Step-2 path and the native ResNet-50 path never imports it
- **the linear evaluation was added.** `evaluate_linear.py` came across with its
  device resolved, its `main()` split into `build_parser()` and
  `run(args, encoder, in_dim)` returning its metrics, and its encoder handed in
  rather than rebuilt from a training checkpoint. `in_dim` is arch-aware: 2048
  for the ResNet-50 backbone, the ViT's `embed_dim` for an `arch: vit` encoder
- **the unified ViT-B/16 Step 2 was added** (`models/vit_swav.py`,
  `train_pretrain_vit_swav.py`, `configs/pretrain_vit.yaml`). A timm
  `VisionTransformer` (dynamic image size, so it takes the 224 global and 96
  local crops) carries the projection head + prototypes and mirrors
  `ResNetSwAV`'s list-of-crops `forward`, so the multi-crop `train_epoch`,
  `swav_loss` and Sinkhorn are reused unchanged; the trainer differs only in the
  ViT backbone and **AdamW** (vs the native LARC-SGD), and writes milestone
  checkpoints at `save_at_epochs` (100/200/300)

## Sinkhorn on one process

Every collective in `distributed_sinkhorn.py` is guarded by
`dist.is_initialized()`, which is why this method runs here at all. That guard
is pinned by a test: a version that reduced unconditionally would fail on a
single CPU, and the failure would look like a porting error rather than a
missing process group.

## What has and has not been exercised

- **A real training step ran on the GPU.** On an NVIDIA A100 (driver CUDA 13.0)
  step 1 completes a training step and writes a loadable `encoder.pt` — the test
  `test_a_real_run_on_cuda_produces_a_loadable_encoder`, on four synthetic
  images with two crop sizes of 32 and 16 pixels
- **The linear evaluation runs, but on the CPU.** Its smoke test uses
  `device: cpu` (through the same `resolve_device` path step 1 uses), and a
  separate test checks it refuses `cuda` when no GPU is visible. The probe has
  not been exercised on a GPU here
- **The ViT Step-2 path ran a hermetic CPU smoke.** A tiny ViT (16-d embed, one
  block) pretrains two epochs with `save_at_epochs: [1, 2]` through
  `python -m adapter`, writes `encoder.pt` and both `encoder_epoch{1,2}.pt`
  milestones, and a milestone probe passes `contract-test` — the test
  `test_pretrain_milestones_then_probe_passes_contract`. The full 300-epoch
  ViT-B/16 recipe (3000 prototypes, eight crops) has not been run here
- **The full recipe has never been run.** 100 epochs of ResNet-50 on ImageNet
  with 3000 prototypes and eight crops needs the GPUs it was written for
- **No multi-process run**, so the distributed Sinkhorn path — the reductions
  themselves — has never executed. Only the single-process branch has
- **The container definition has never been built** on this machine
- The numbers in the configs are the recipe, not results; no accuracy from this
  port has been measured against anything
