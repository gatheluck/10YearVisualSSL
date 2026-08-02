# 17_swav — step 1 and linear evaluation

Caron, Misra, Mairal, Goyal, Bojanowski and Joulin, *Unsupervised Learning of
Visual Features by Contrasting Cluster Assignments*, 2020
([arXiv:2006.09882](https://arxiv.org/abs/2006.09882)).

Rather than comparing two views directly, SwAV assigns each view to a set of
learned prototypes and makes one view predict the other's assignment. A
Sinkhorn-Knopp normalisation keeps the prototypes evenly used, which is what
stops the representation collapsing. The encoder is what the rest of the
project wants; the prototypes are training machinery.

## The linear evaluation

The second stage freezes the encoder step 1 produced and fits a linear
classifier on real labels. **These are the numbers this project exists to
compare**, and this port adds them for a fifth method.

It reports **three** accuracies: a best top-1 and a final top-1 and top-5. The
handoff is the same shape the other two-stage ports use — the adapter builds
the encoder with `load_encoder` from `encoder.pt` (the backbone alone) and
hands it in, rather than letting the evaluation rebuild the whole model from a
training checkpoint. `model_type='vit'` is refused by name: step 2 was not
brought across, so `build_vit_swav` is absent.

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
- `models/__init__.py` advertised the step 2 ViT through a lazy `__getattr__`.
  That module was not brought across, so the package was promising a name it
  could not import
- **the linear evaluation was added.** `evaluate_linear.py` came across with its
  device resolved, its `main()` split into `build_parser()` and
  `run(args, encoder, in_dim)` returning its metrics, and its encoder handed in
  rather than rebuilt from a training checkpoint. `model_type='vit'` is refused
  by name; the branch importing `build_vit_swav` (step 2, absent) is unreachable

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
- **The full recipe has never been run.** 100 epochs of ResNet-50 on ImageNet
  with 3000 prototypes and eight crops needs the GPUs it was written for
- **No multi-process run**, so the distributed Sinkhorn path — the reductions
  themselves — has never executed. Only the single-process branch has
- **The container definition has never been built** on this machine
- The numbers in the configs are the recipe, not results; no accuracy from this
  port has been measured against anything
