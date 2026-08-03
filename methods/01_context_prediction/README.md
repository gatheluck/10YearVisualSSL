# 01_context_prediction — step 1 and linear evaluation

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
`tests/test_method_01_context_prediction.py` pins their digests against
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

One command does the whole thing — resolve, run, verify against the contract,
and record what was asked:

```bash
python3 bin/launch.py --config methods/01_context_prediction/configs/step1.yaml \
    --method 01_context_prediction --set DATA_ROOT=/path/to/ILSVRC2012
```

Then the linear evaluation, reading the encoder the first stage produced:

```bash
python3 bin/launch.py --config methods/01_context_prediction/configs/linear_eval.yaml \
    --method 01_context_prediction --set DATA_ROOT=/path/to/ILSVRC2012 \
    --set ENCODER=runs/01_context_prediction-<sha>/out/encoder.pt
```

The steps below are what that does, and are worth knowing when something goes
wrong.

### The same thing by hand

Resolve the config first, so the run is identified by a hash:

```bash
python3 bin/resolve-config.py \
    --config methods/01_context_prediction/configs/step1.yaml \
    --out runs/ctxpred/resolved.json \
    --set DATA_ROOT=/path/to/ILSVRC2012
```

Then run the adapter from this directory:

```bash
cd methods/01_context_prediction
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

## Stage 2: linear evaluation

The standard frozen-features protocol: the encoder is frozen, features are
extracted once, and a single linear layer is trained on top.

**This is the first stage that consumes another stage's output.** Step 1 emits
`encoder.pt` because the contract says so; if nothing can read it, the
contract is decorative. This reads it.

```bash
python3 bin/resolve-config.py \
    --config methods/01_context_prediction/configs/linear_eval.yaml \
    --out runs/lineval/resolved.json \
    --set DATA_ROOT=/path/to/ILSVRC2012 \
    --set ENCODER=runs/ctxpred/out/encoder.pt
cd methods/01_context_prediction
PYTHONPATH=../.. python3 -m adapter \
    --config ../../runs/lineval/resolved.json --out ../../runs/lineval/out
```

Two things this settled:

**The stage comes from the config, not a flag.** The contract fixes the
adapter's arguments at exactly two and says anything else affecting the result
belongs in the config (CONTRACT section 2). A `--stage` flag would have been
an input `config_sha256` does not cover. So every config now declares
`stage:`, and each stage declares exactly the keys it reads — a setting the
stage never looks at cannot sit in a config claiming to have had an effect.

**It produces no encoder, and says so.** It evaluates one and produces a
classifier. CONTRACT section 3 permits that only with a recorded reason; that
mechanism existed since the contract was written and had never been reached
until now. `run_manifest.json` carries `encoder_absent_reason`.

The evaluator accepts either input and tells them apart by their keys, never
by guessing:

| Input | Where it comes from |
|---|---|
| `encoder.pt` | the contract: the encoder alone, unprefixed |
| a training checkpoint | the cluster's own runs: `{"state_dict": <whole model>}` |

Anything else is refused. Loading the wrong one would evaluate an
untrained encoder and report a number that looks like a result.

It needed no device work — the original already chose CPU when no GPU was
present. It did need seeding, for the same reason step 1 did.

`torchvision` entered the method's dependencies here (`ImageFolder` and the
standard transforms), which `tests/test_method_requirements.py` caught as an
undeclared import before anything ran.

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
    -r methods/01_context_prediction/requirements.lock.txt \
    -r requirements-tools.lock.txt
```

Nothing about it is distribution-specific: a `python3-venv` package and pip
are all it assumes. On Ubuntu that is `apt install python3-venv`.

Use the interpreter named in `.python-version` at the repository root. The
lock holds cp312 wheels and will not install on another minor version.

To confirm the environment is exactly the lock and not merely compatible with
it:

```bash
python3 bin/verify-environment.py --lock methods/01_context_prediction/requirements.lock.txt --lock requirements-tools.lock.txt
```

**Pass every lock the install used.** This used to be a `diff` against
`pip freeze`, and naming only one of the two files made `PyYAML` look like an
unexplained extra — a correct environment reported as wrong. The tool takes
the files as arguments so the mistake is not available.

### As a container

`Dockerfile` in this directory builds the same locked environment as an image.
It is the same lock, installed the same way, so nothing here duplicates the
venv path — the container is a second way to reach one environment, not a
second environment.

```bash
docker build -f methods/01_context_prediction/Dockerfile -t ctxpred .
```

Build from the **repository root**: the tooling lock and `bin/` live there.
Podman takes the same command.

Two properties worth knowing:

- **The base image is pinned by digest**, not by tag. `python:3.12.13-slim-bookworm`
  will mean a different image next month, and an image built from a tag cannot
  be rebuilt. The digest was read from the registry on 2026-07-30 and covers
  linux/amd64 and linux/arm64 among others
- **The build proves its own environment.** `verify-environment.py` runs as a
  build step, so an image whose contents disagree with the lock **fails to
  build** rather than existing and misreporting what is in it

Running it:

```bash
docker run --rm -v "$PWD/runs:/runs" -w /work/methods/01_context_prediction ctxpred python3 -m adapter --config /runs/resolved.json --out /runs/out
```

On a cluster, where Docker generally needs privileges nobody has, Apptainer
reads the same image:

```bash
apptainer build ctxpred.sif docker-daemon://ctxpred:latest
apptainer exec --bind "$PWD/runs:/runs" ctxpred.sif sh -c 'cd /work/methods/01_context_prediction && python3 -m adapter --config /runs/resolved.json --out /runs/out'
```

There is deliberately no `ENTRYPOINT`: Docker and Apptainer disagree about how
one is invoked, and a plain image is driven the same way by both.

#### What was measured, and what was not

The image **has now been built and run**, on 2026-07-30, for both
architectures Docker offers on this machine:

| | linux/arm64 | linux/amd64 |
|---|---|---|
| build, including the self-verification step | passes | passes |
| identity inside | Python 3.12.13, torch 2.13.0+cpu | Python 3.12.13, torch 2.13.0+cpu |
| `resolve-config` → adapter → `contract-test` | exit 0 | exit 0 |
| the run's recorded environment vs the locks | matches | matches |
| the same config twice | **byte-identical** | **byte-identical** |

**And across the two, deliberately not identical:**

```
arm64  encoder.pt bd90b98ef87dea0265bb…   val_loss 2.4203052520751953
amd64  encoder.pt cbf177068c3dcd6583f5…   val_loss 2.7717556953430176
```

That is the guarantee working, not failing. Floating-point addition is not
associative, so a different instruction set reorders the arithmetic. What must
hold is that the difference is explainable, and it is: the manifests record
`aarch64` and `x86_64`.

Still not verified: **Apptainer**, and **any GPU**. Neither is available here,
so the `apptainer` commands above come from its documentation rather than from
a run, and the CUDA determinism settings remain unexercised.

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
`tests/test_method_01_context_prediction.py` measures it — two runs, compared
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
python3 bin/verify-environment.py --lock methods/01_context_prediction/requirements.lock.txt --lock requirements-tools.lock.txt --manifest runs/ctxpred/out/run_manifest.json
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
