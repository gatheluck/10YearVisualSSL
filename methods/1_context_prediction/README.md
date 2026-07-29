# 1_context_prediction — step 1

Doersch, Gupta and Efros, *Unsupervised Visual Representation Learning by
Context Prediction*, ICCV 2015.

A siamese AlexNet is shown a centre patch and one of its eight neighbours and
has to say which of the eight it is. Solving that requires recognising parts
and their spatial arrangement, which is where the representation comes from.

## Which track this is, and why

The captured tree held two step-1 implementations. **This is the
official-style one.** Its sibling states plainly why they are separate:

> This is intentionally separate from `train_step1_alexnet.py` because the
> legacy file is not paper-compatible: model, preprocessing, and sampling all
> differ from the released deepcontext implementation.

For a ten-year comparison the paper-compatible one is the baseline, so the
legacy track was not brought across. It is still in the Capture repository if
it is ever wanted.

## Settings, and where each comes from

The training settings follow the **released deepcontext implementation**
(`https://github.com/cdoersch/deepcontext`), recorded in the `protocol` block
that every checkpoint carries:

| Setting | Value | Source |
|---|---|---|
| patch size | 96 px | paper Section 3, and the release |
| gap between patches | 48 px | paper Section 3 ("approximately half the patch width") |
| jitter | ±7 px | paper Section 3 |
| image resize | random target area in 150K–450K pixels | release |
| optimizer | SGD, **lr 1e-5 fixed, momentum 0, weight decay 0** | release |
| loss | cross-entropy, `reduction="sum"` | matches Caffe `SoftmaxWithLoss` with normalization NONE |
| DDP | loss multiplied by world size before backward | undoes DDP's gradient averaging, so the sum reduction survives |
| colour | keep one random channel, replace the other two with noise | release; defeats the chromatic-aberration shortcut |
| patch normalisation | RMS-normalised, then scaled by 50 | release |

**The paper text and the released code disagree about the optimizer**, and
this track follows the code. The paper says "high momentum values (e.g.
.999)"; the release uses momentum 0 at a fixed 1e-5. The legacy track followed
the paper text, which is one of the ways the two differ. Notes in the Capture
repository (`CRITICAL_PAPER_CORRECTIONS.md`, `PAPER_EXACT_SPECS.md`) argue for
the paper reading and describe the legacy files; **they do not describe this
track**, which is why they were not copied here.

Not verified here: which of the two reproduces the published numbers. That
needs a full run on ImageNet.

## What changed during the port

The two files that carry the science — `models/alexnet_context_official.py`
and `data/context_dataset_official.py` — **came across untouched**, and
`tests/test_method_1_context_prediction.py` pins their digests against
`provenance.json`. Everything else that changed is listed there, and here:

1. **`main()` split into `build_parser()` and `run(args)`.** A pure
   extraction, so the adapter can build the same arguments without going
   through a shell. Every original flag is still accepted, because the
   cluster's job scripts call this file directly
2. **The device is selectable.** The original hard-coded `cuda`, so it could
   not run anywhere else. `auto` uses cuda when it is there; asking for `cuda`
   without one is refused rather than quietly downgraded to the CPU
3. **The training data stream is now seeded.** See below
4. `__init__.py`, `models/__init__.py` and `data/__init__.py` were rewritten:
   they re-exported the legacy modules, which are not here
5. `run()` returns the final evaluation, and evaluates once at the end. The
   original evaluated only every `eval_every_steps`, so a run whose length was
   not a multiple of that finished with nothing to report

### The seeding fix

**Two runs of the same config produced different weights.** Measured on CPU
with a synthetic dataset, before the fix.

The dataset draws every patch position, jitter, colour-channel choice and
pixelation decision from Python's `random` and from `np.random`. The captured
code seeded neither on the training path:

- `torch.manual_seed(seed)` was called, but that does not touch either
- `seed_worker`, which does seed them, was passed to the **validation** loader
  only — the training loader was built without it
- with `num_workers=0` it does not run at all

`run()` now seeds `random` and `np.random` alongside torch, and passes
`seed_worker` to the training loader as well. **No distribution changes** —
only whether the same draw can be repeated.

Not verified: whether, before the fix, `num_workers>0` on Linux left every
worker with the same inherited `random` state and therefore duplicated
sampling. That depends on fork semantics and was not measured.

## Running it

Resolve the config first, so the run is identified by a hash:

```bash
python3 bin/resolve-config.py \
    --config methods/1_context_prediction/configs/step1.yaml \
    --out runs/ctxpred/resolved.json \
    --set DATA_ROOT=/path/to/ILSVRC2012
```

Then run the adapter from this directory:

```bash
cd methods/1_context_prediction
PYTHONPATH=../.. python3 -m adapter \
    --config ../../runs/ctxpred/resolved.json \
    --out ../../runs/ctxpred/out
```

Then check it against the contract:

```bash
python3 bin/contract-test.py --out runs/ctxpred/out \
    --config runs/ctxpred/resolved.json --exit-status $?
```

`--out` receives `encoder.pt` (the encoder's `state_dict`, without the pretext
classifier), `metrics.json`, `run_manifest.json`, and a `work/` directory
holding the original's own checkpoints, `run_config.json` and
`progress.jsonl`. Nothing is written outside `--out`.

The multi-GPU path is unchanged from the capture: the file still takes the
original flags, so `torchrun ... train_step1_alexnet_official.py --data_path
... --save_dir ...` works as before.

## Requirements

`requirements.txt`, as captured. The adapter and the contract tools need
nothing installed; the training itself needs torch and torchvision.

## What is not here yet

- step 2 (ViT) and the linear evaluation
- `resume`. The original supports it; this adapter refuses the key rather
  than accept it and ignore it
