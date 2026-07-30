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

## The environment

The adapter and the contract tools need nothing installed. **The training
needs three packages**: `torch`, `numpy` and `Pillow`. Nothing else is
imported anywhere in this directory, and
`tests/test_method_requirements.py` checks that in both directions — an
import that is not declared fails, and a declared package that nothing
imports fails too.

The file that came across from the capture also listed `timm`, `PyYAML`,
`tensorboard` and `tqdm`. Those belong to the legacy track and are never
imported here; they were removed, and the check now prevents that class of
mistake for every method.

| File | What it is for |
|---|---|
| `requirements.txt` | which packages. Floors, not an environment |
| `requirements.lock.txt` | exact versions, so a recorded run can be rebuilt |

### Any Linux, a laptop, or a cloud VM (CPU)

Verified: this is the path the port was checked on, and the environment it
produces is byte-for-byte the lock file.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r methods/1_context_prediction/requirements.lock.txt \
    -r requirements-tools.lock.txt
```

Nothing about it is distribution-specific: a `python3-venv` package and pip
are all it assumes. On Ubuntu that is `apt install python3-venv`.

Use the interpreter named in `.python-version` at the repository root. The
lock holds cp312 wheels and will not install on another minor version.

To confirm the environment is exactly the lock and not merely compatible with
it:

```bash
python3 bin/verify-environment.py --lock methods/1_context_prediction/requirements.lock.txt --lock requirements-tools.lock.txt
```

**Pass every lock the install used.** This used to be a `diff` against
`pip freeze`, and naming only one of the two files made `PyYAML` look like an
unexplained extra — a correct environment reported as wrong. The tool takes
the files as arguments so the mistake is not available.

### Rebuilding the lock

When a version has to move, regenerate rather than hand-edit — a hash typed by
hand is a hash nobody checked:

```bash
pip freeze | sort > /tmp/closure.txt
pip download -d /tmp/w --no-deps -r /tmp/closure.txt
pip download -d /tmp/w --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 --python-version 3.12 --implementation cp --abi cp312 -r /tmp/closure.txt
```

then hash each wheel with `shasum -a 256` and list every distinct digest under
its package. `tests/test_method_requirements.py` fails if any entry loses its
hash, if the lock stops being a closure, or if a version is not exact.

### On the cluster

The captured `setup_conda_env.sh` built a conda environment named
`py3.10_context_prediction` with Python 3.10 and `pytorch-cuda=12.1`. It did
not come across, because it hard-codes cluster paths and a
`scripts/activate_runtime.sh` that only exists there. The venv path above
works on a login node as well, since it needs nothing outside pip.

### What reproducibility means here

**Guaranteed: the same environment and the same config give the same bits.**
`tests/test_method_1_context_prediction.py` measures it — two runs, compared
by the hash of `encoder.pt`. Holding it took work: the training data stream
was unseeded (see above), and torch was free to pick kernels by timing until
`make_deterministic()` was added.

**Not guaranteed, and not achievable: agreement across different hardware.**
Floating-point addition is not associative, so a different instruction set,
BLAS or cuDNN reorders the arithmetic and the last bits move. No amount of
pinning changes that.

**Therefore what must always hold: a difference is explainable.**
`run_manifest.json` records every installed package and its version, a
`packages_sha256` over the lot, and the system and machine. Two runs that
disagree can be compared and the reason found.

That record is checkable, not merely stored. To ask whether a *finished* run
used the locked environment — long after the machine is gone:

```bash
python3 bin/verify-environment.py --lock methods/1_context_prediction/requirements.lock.txt --lock requirements-tools.lock.txt --manifest runs/ctxpred/out/run_manifest.json
```

Set deliberately, in `make_deterministic()`:
`torch.use_deterministic_algorithms(True, warn_only=True)`,
`cudnn.deterministic = True`, `cudnn.benchmark = False`, and
`CUBLAS_WORKSPACE_CONFIG` from the adapter's entry point, before any CUDA
context exists. `warn_only=True` is a choice: an operation with no
deterministic kernel then warns in the run's own output instead of aborting,
which keeps the method usable while still saying that a step was not
reproducible.

**Not verified**: none of the CUDA-specific settings has been exercised. This
machine has no GPU.

### **What cannot be reproduced, and why**

**The exact versions the captured cluster runs used are not recoverable.**

- the environment directories (`envs/`) were classified as artifacts and were
  never captured — deliberately, they are gigabytes of build output
- `setup_conda_env.sh` pins nothing: `conda install pytorch torchvision
  torchaudio pytorch-cuda=12.1` resolves to whatever was current that day
- so the record fixes Python 3.10 and CUDA 12.1, and nothing further

`requirements.lock.txt` therefore pins **the versions this port was verified
against**, which is an honest lock but a different one. Closing the gap needs
the versions read off the cluster; that has not been done.

## What is not here yet

- step 2 (ViT) and the linear evaluation
- `resume`. The original supports it; this adapter refuses the key rather
  than accept it and ignore it
