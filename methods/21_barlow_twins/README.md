# 21_barlow_twins — step 1 and linear evaluation

Zbontar, Jing, Misra, LeCun and Deny, *Barlow Twins: Self-Supervised Learning
via Redundancy Reduction*, 2021
([arXiv:2103.03230](https://arxiv.org/abs/2103.03230)).

Two distorted views of one image go through the same encoder, and the loss
drives the cross-correlation matrix of the two embeddings towards the
identity — the diagonal to one, everything off it to zero. No negative pairs
and no stop-gradient: redundancy reduction is what stops the representation
collapsing. The backbone is what the rest of the project wants.

## Why this method was fourth

The runner-up of the six candidates measured in the Capture repository's
design notes. It shares the template of the port before it, which is the
point: the two differ in exactly two ways, and both are things the earlier
port did not have to solve.

## The linear evaluation

The second stage freezes the encoder step 1 produced and fits a linear
classifier on real labels. **These are the numbers this project exists to
compare**, and this port adds them for a fourth method.

It reports **three** accuracies, not four: a best top-1 and a final top-1 and
top-5. This original does not record a best top-5, and inventing one would be a
number nothing measured.

The handoff is the same shape the earlier two-stage ports used. The captured
evaluation rebuilds the whole model from a training checkpoint with
`strict=True` and takes its backbone; the contract's artifact is `encoder.pt`,
the backbone alone. Rather than teach the evaluation a second way to recognise
a file, the adapter builds the encoder with `load_encoder` and hands it in.
`model_type='vit'` is refused by name — step 2 was not brought across, so
`build_barlow_vit` is absent, and the captured evaluation's top-level import of
it was removed so the module imports at all.

## What was new here

**Mixed precision, and refusing to downgrade it quietly.** The captured
trainer offers three settings — `fp32`, `bf16` and `amp_fp16` — and writes
`device_type="cuda"` into both its autocast and its `GradScaler`. On a CPU,
fp32 and bf16 exist and fp16 does not. Running fp32 when fp16 was asked for
would report a run at a precision it never used, so **the pair is refused by
name**, exactly as asking for an absent GPU is. `bf16` is accepted and
actually exercised on a CPU by the tests.

**Python's own `random`.** Here the solarisation and the blur call
`random.random()` directly rather than going through a torchvision transform,
so `make_deterministic` seeds `random` and not only torch.

A first version of this port went further: it claimed loader workers needed
seeding too, added a `seed_worker`, and changed the captured loader to accept
one. **That claim was wrong, and a surviving mutation exposed it.** Torch's
worker loop seeds `random` itself — `random.seed` appears in `_worker_loop`,
and two runs with no `worker_init_fn` draw identical values. Both were
measured after the fact. The change was reverted and the loader is pinned by
hash. The test that runs one config twice with two workers stays, because
reproducibility with workers is a real property; it simply is not this port's
doing.

## What changed during the port

Recorded in full in `provenance.json`; `models/barlow_resnet.py` and
`optim/lars.py` came across untouched and are pinned by hash.

- the device is resolved instead of assumed; the captured trainer called
  `.cuda(local_rank)` on the model and every batch
- `autocast_context` and `GradScaler` take the resolved device rather than the
  literal `"cuda"`
- `main()` split into `build_parser()` and `run(args, config)`, which returns
  the epoch loss and the epoch count. The captured version computed them and
  discarded them
- **the linear evaluation was added.** `evaluate_linear.py` came across with
  its device resolved, its `main()` split into `build_parser()` and
  `run(args, encoder, in_dim)` returning its metrics, and its encoder handed in
  rather than rebuilt from a training checkpoint. Its top-level import of
  `build_barlow_vit` was removed and `model_type='vit'` refused by name, since
  step 2 is absent
- step 2 (the ViT) was not brought across: the capture has no official-style
  variant of it, the same reason it was left out of the earlier ports

## The configuration

`configs/step1.yaml` holds the recipe the captured runs used. Note that Barlow
Twins takes **two** learning rates, as LARS does in the paper — one for the
weights and one for the biases and normalisation parameters. A first draft of
this port declared a single `base_lr`, which the trainer never reads; the run
failed on it, which is what declaring only what a stage reads is for.

`end_epoch`, the captured pilot switch that stops a run short, is deliberately
not a setting: the config says how many epochs ran, and a second way to change
that would sit outside the hash.

## What has and has not been exercised

- **A real training step and the linear probe ran on the GPU.** On an NVIDIA
  A100 (driver CUDA 13.0) step 1 completes a training step, writes a loadable
  `encoder.pt`, and the linear evaluation runs to a number — the tests
  `test_a_real_run_on_cuda_produces_a_loadable_encoder` and the linear-eval
  smoke test, on a handful of synthetic images at 32 pixels. The CPU path runs
  the same chain
- **The full recipe has never been run.** 100 epochs of ResNet-50 on
  ImageNet-1k needs the GPUs it was written for
- **`amp_fp16` has never executed.** It is accepted by the config check when a
  GPU is asked for and refused on a CPU; neither is the same as having run it,
  and the GPU smoke test uses fp32
- **No multi-process run.** The distributed path is the captured one, untouched
- **The container definition has never been built** on this machine
- The numbers in the configs are the recipe, not results; no accuracy from this
  port has been measured against anything
