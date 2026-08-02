# 27_ibot — step 1 and linear evaluation

Zhou, Wei, Wang, Shen, Xie, Yuille and Kong, *iBOT: Image BERT Pre-Training
with Online Tokenizer*, 2021 ([arXiv:2111.07832](https://arxiv.org/abs/2111.07832)).

A student and a teacher ViT share a projection head. Each image is shown as two
global crops (with some patches masked) and several local crops; every masked
patch token and the [CLS] token are trained to predict the teacher's
assignment, and the teacher is an exponential moving average of the student. A
centering-and-sharpening step on the teacher's outputs is what stops the
representation collapsing. The encoder the rest of the project wants is the
backbone; the heads and the centering buffers are training machinery.

## Why this method was sixth

Chosen by the same measurement that ordered the earlier official-style ports
(Capture repository, DESIGN 5.40/5.41): the six candidates carrying
`*_official*` files. iBOT was the **heaviest** of the six — a ViT-S/B trainer
of 517 lines — so it came last. That was a compute cost, not a portability
problem, and it is the one constraint the now-available GPU removes. It shares
the template of the three ports before it: `setup_dist()` returns early when
`LOCAL_RANK` is unset (so the single-process path exists and `nccl` is never
reached), it uses **no automatic mixed precision**, and its multi-crop loader
already builds a `DistributedSampler` only when distributed — so, unlike the
SwAV loader, it runs on one process unchanged. It carries an official-style
Python step-1 trainer and an official-style linear evaluation, so both stages
port faithfully.

## The linear evaluation

The second stage freezes the encoder step 1 produced and fits a linear
classifier on real labels. **These are the numbers this project exists to
compare.**

It reports **all four** comparable accuracies: best and final top-1, and best
and final top-5. iBOT's own evaluation records a best top-5 as well as the
rest, so unlike the SimSiam port (which reports three) every comparable slot in
the contract vocabulary is filled.

The handoff needed the same care the SimSiam port did. The captured evaluation
rebuilds the whole iBOT model from a training checkpoint with `strict=True` and
takes its teacher; the contract's artifact is `encoder.pt`, the teacher
backbone alone. Rather than teach the evaluation a second way to recognise a
file, the adapter builds the encoder with `load_encoder` and hands it in, so
one place knows how an encoder is loaded. The feature dimension is left to the
evaluation, which derives it from the architecture and the feature protocol.

## What was new here

**Which backbone is the encoder — and the source is ambiguous.** iBOT trains a
student and a teacher. The model exposes `get_encoder()` returning the
**student**, but every official linear-evaluation script probes
`--checkpoint_key teacher`, and the paper's reported numbers are the teacher's
(the EMA). Faithfulness to the *reported result* wins: `encoder.pt` holds the
teacher ViT. It is selected by the `teacher.` prefix, which does not match
`teacher_head.`, so the head does not come along, and the prefix is stripped so
the file is a plain ViT `state_dict`. This decision is recorded in
`provenance.json` and enforced by the adapter, not left to a config flag a run
could set the other way.

**Two metrics with nowhere to go.** The trainer reports the CLS and patch
components of its loss alongside the total. Both are real measurements and
belong to neither family in the contract vocabulary, so their translation table
maps them to `None`: kept under their own names in `metrics_raw`, kept out of
the comparable block. Inventing contract names would offer them for comparison
against methods that have no such quantities.

**A configuration of many optional settings.** iBOT reads most of its
hyperparameters with a default behind them — the temperature schedule, the
masking ratio, the teacher momentum. Left out of the contract config, a run
would be described without them, filled by whatever that version of the code
defaulted to. Every key the trainer reads is declared, so the resolved config
says what ran. The multi-crop scale ranges are refused by name if they are not
`[low, high]` pairs, rather than failing deep inside the loader.

## What changed during the port

Recorded in full in `provenance.json`; `models/ibot.py`,
`models/vision_transformer.py` and `data/multicrop.py` came across untouched
and are pinned by hash.

- **The device is resolved instead of assumed.** The captured trainer called
  `.cuda(local_rank)` on the model, the loss and every crop and mask, so it
  could not start without a GPU. `resolve_device()` picks one, and asking for
  `cuda` where there is none is an error rather than a quiet fall back to the
  CPU — the two are not the same run. The device guard is mutation-tested
  (`mutations/27_ibot-step1-device.json`)
- **`main()` is split into `build_parser()` and `run(args, config)`**, and
  `run` returns the epoch loss, its CLS and patch components and the epoch
  count. The captured version computed them and discarded them
- **The run is seeded through `make_deterministic`.** iBOT's multi-crop and
  block masking draw from Python's `random` and from torch; both are seeded, so
  the same config twice gives the same `encoder.pt` — checked by test
- **The backbone is chosen from `model.arch`** rather than hard-coded, so the
  resolved config says which architecture ran. Only `vit_small` is accepted;
  `vit_base` belongs to step 2
- **Step 2 was not brought across.** `train_step2_vit.py`,
  `configs/step2_vit_b.yaml`, the step-2 shell script and
  `tests/test_step2_protocol.py` are step 2 (ViT-B), which the contract does
  not adopt (its stages are `step1` and `linear_eval`)

## The configuration

`configs/step1.yaml` holds the recipe the captured runs used. Two keys from the
captured config are **deliberately absent** — `data.val_path` and
`training.optimizer`. The step-1 trainer never reads them (the optimizer is
AdamW by construction), and a key that is ignored is a setting claiming an
effect it never had. The two pilot-only truncation knobs (`stop_after_epochs`,
`max_steps_per_epoch`) are not part of the contract config either, so a run is
never silently cut short.

Neither the data path nor the output path is in the file. The capture named an
absolute path on the cluster for both, which is reproducible nowhere else.

## Running it

```bash
python3 bin/launch.py --config methods/27_ibot/configs/step1.yaml --method 27_ibot --set DATA_ROOT=/path/to/imagenet
```

Then the linear evaluation, on the `encoder.pt` the first stage wrote:

```bash
python3 bin/launch.py --config methods/27_ibot/configs/linear_eval.yaml --method 27_ibot --set DATA_ROOT=/path/to/imagenet --set ENCODER=runs/<step1-run>/out/encoder.pt
```

The adapter writes `encoder.pt` (step 1), `metrics.json` and
`run_manifest.json` under `--out`, with the original's checkpoints, its config
copy and its TensorBoard events under `work/`.

## What has and has not been exercised

- **A real training step ran on the GPU.** On an NVIDIA A100 (driver CUDA 13.0)
  the step-1 trainer completes a training step, writes a loadable `encoder.pt`,
  and the linear evaluation runs to a number — the tests
  `test_a_real_run_on_cuda_produces_a_loadable_encoder` and the linear-eval
  smoke test, on a handful of synthetic images at 32 pixels. The CPU path runs
  the same chain
- **The full recipe has never been run.** 800 epochs of ViT-S/16 on
  ImageNet-1k is hundreds of GPU-hours; nothing here has executed one
- **No multi-process run.** The distributed path is the captured one and is
  untouched, but nothing here has executed it
- **The container definition has never been built** on this machine; it is
  checked by reading, like the others
- The numbers in `configs/step1.yaml` and `configs/linear_eval.yaml` are the
  recipe, not results. No accuracy from this port has been measured against
  anything
