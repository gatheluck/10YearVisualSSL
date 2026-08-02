# Running the ported methods on a GPU

Last updated: 2026-08-02

Until now every method was ported, tested and verified **on a CPU only**. The
locks shipped were the CPU wheels, and `torch.cuda` was never reached on real
hardware -- the device-selection tests mocked `torch.cuda.is_available`, and
nothing ran a kernel on a GPU. This document records how a GPU is brought in
without giving up any of the reproducibility the CPU path already had, and the
one behaviour every method must hold whether or not a GPU is present.

**This separation is held by machinery, not by prose.** Where a rule below can
be checked by a test, the test is named against it. Read this before adding a
method or a GPU lock, the same way `docs/PLATFORMS.md` is read before touching
the platform layer.

---

## 1. One environment per method, never a shared one

A single virtual environment shared across methods was considered and
rejected. It is wrong here for reasons the repository already encodes:

- `adapterlib/__init__.py` opens with it: *"Ten years of methods means ten
  years of incompatible environments, so every adapter is its own process."*
  Thirty-plus methods will not agree on one set of versions; the design is
  built so they never have to.
- Each method already carries its **own** `requirements.lock.txt` and its own
  `Dockerfile`, which builds a fresh `venv` and installs only that method's
  lock. The CI `locked` job (`.github/workflows/tests.yml`) does the same, once
  per method, and **discovers methods by globbing rather than listing** so the
  matrix cannot go stale as methods are added.
- `bin/verify-environment.py` treats **any** package the locks do not name as a
  failure -- *"installed but no lock describes it, so this environment cannot be
  rebuilt from the locks given"*. A shared superset environment therefore fails
  verification for every method whose lock is a strict subset of it. Measured:
  `methods/1_context_prediction/requirements.lock.txt` names neither
  `tensorboard` nor `PyYAML`, so a shared venv that has them (because another
  method needs them) makes method 1 unverifiable.

The GPU path follows the same rule: **one GPU lock and one GPU venv per
method.** The convenience of installing torch once does not outweigh making the
environments unverifiable and unable to scale.

---

## 2. The GPU lock is the CPU lock's closure, rebuilt against the CUDA index

Each method that has `requirements.lock.txt` (CPU) also gets
`requirements.lock.cu130.txt` (GPU). The CPU lock stays the source of truth for
which **releases** the port was verified against; the GPU lock is the same
releases, resolved against the CUDA index so that:

1. `torch` and `torchvision` resolve to their CUDA wheels rather than the
   `+cpu` wheels, and their `--hash` lines are the CUDA wheels' hashes.
2. the wheels a GPU build pulls in and a CPU build does not -- `nvidia-*`,
   `cuda-*` and `triton` -- are added, each pinned and hashed. For `2_vae` this
   is eighteen extra distributions (measured: 22 packages become 41). Without
   them the install fails under `--require-hashes` and, even if it did not,
   `verify-environment` would reject them as undescribed.

Everything else -- `numpy`, `pillow`, the `tensorboard` stack -- comes from
PyPI with the **same** versions and the same hashes as the CPU lock. The two
locks describe the same closure of releases; they differ only in the build
variant of the two CUDA packages and in the CUDA runtime wheels that variant
requires.

### Why `+cu130` still matches a lock that says `2.13.0`

`bin/verify-environment.py` already anticipated GPU builds (`split_local`, and
the `build variant` branch of `compare`). A lock pins the **release**
(`torch==2.13.0`); the installed wheel's build tag (`+cpu`, `+cu130`) is
recorded as a build variant of that release, not counted as a difference. In
practice torch's *distribution metadata* is the bare `2.13.0` anyway -- the
`+cu130` tag appears only in `torch.__version__` at runtime -- so the pin
matches exactly and `verify-environment` reports the environment matches the
locks. This is why the GPU lock pins `torch==2.13.0`, identical to the CPU lock,
rather than `torch==2.13.0+cu130`.

### Regenerating a GPU lock

The releases are held identical to the CPU lock by passing that lock's `==`
pins as a constraint:

    # 1. constraints = the CPU lock's name==version lines, hashes stripped
    # 2. compile the method's floors against them, on the CUDA index:
    uv pip compile methods/<m>/requirements.txt \
        --constraint <cpu-lock-pins> \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
        --generate-hashes

The `cu130` index is chosen to match this project's target hardware: the wheel
it serves is `torch 2.13.0+cu130` (`torch.version.cuda == 13.0`), and the
verification below ran on an NVIDIA A100 whose driver reports CUDA 13.0. The
`cu126` index happens to serve a byte-identical `+cu130` wheel (same hashes);
`cu130` is named because the name should say what the artifact is.

### Installing and verifying a GPU environment

    uv pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        -r methods/<m>/requirements.lock.cu130.txt \
        -r requirements-tools.lock.txt
    python bin/verify-environment.py \
        --lock methods/<m>/requirements.lock.cu130.txt \
        --lock requirements-tools.lock.txt   # must exit 0

The per-method GPU venv lives at `.venvs/<m>/` at the repository root (ignored
by `.gitignore`), **not** inside `methods/<m>/`. This matters: a venv placed in
the method directory is read by the method-scanning tests --
`test_method_requirements` walks `methods/<m>` for `*.py` and would ast-parse
the whole of the vendored torch, turning a 16-second suite into a 456-second
one and reporting torch's own imports as undeclared requirements (measured,
2026-08-02). The Dockerfile (`/opt/venv`) and CI (repo-root `.venv`) already
keep the environment out of the source tree for the same reason; `.venvs/<m>/`
does the same while letting many methods' environments coexist.

The venv is large -- about 4.6 GB for `2_vae`, because the CUDA runtime wheels
are large -- so it is built on demand, not kept for every method at once.

---

## 3. A machine with no GPU must still pass the whole suite

Development will continue on machines with no GPU. The suite must stay green
there, exactly as it already stays green with no `torch` installed at all
(every method's tests are guarded by `skipUnless(HAVE_TORCH, ...)`).

The rule for GPU work is the same, one level in:

- A test that **needs a real GPU** -- one that runs a kernel on `cuda` -- is
  guarded by `skipUnless(torch.cuda.is_available(), ...)`. On a GPU-less
  machine it skips; it never fails.
- A test that checks **device *selection*** does not need a GPU and must not be
  guarded by one. It mocks `torch.cuda.is_available` both ways and asserts the
  decision, so the refuse/honour logic is covered even where no GPU exists.

A skipped GPU test is reported as skipped, never as passed. Installing a GPU
lock is opt-in; it is never a precondition for the suite to pass.

---

## 4. The device invariant every method must honour

A config carries a `device` (`auto`, `cuda` or `cpu`). Selecting the device is
**decided from that value, not assumed from what hardware happens to be
visible**:

- `cpu` runs on the CPU **even when a GPU is present**. Asking for the CPU and
  silently getting a GPU is a different run than the one requested.
- `cuda` is honoured only when a GPU is visible, and **refused loudly**
  otherwise. Falling back from `cuda` to `cpu` without a word turns a
  misconfigured GPU job into a run that looks fine and is a thousand times
  slower -- and misreports which hardware produced the result.
- `auto` takes a GPU when one is visible and a CPU otherwise.

This is the logic of `resolve_device(spec, local_rank)` in
`methods/1_context_prediction` and `methods/20_simsiam`. It is one rule; a new
method reuses this shape rather than reimplementing it.

**Validating `device` is not the same as honouring it.** `methods/2_vae` was
ported on the CPU track: its adapter validated `config["device"]` against
`auto/cuda/cpu`, but `to_args()` never put the value on the trainer's arguments
and the trainer selected the device from `torch.cuda.is_available()` alone. On a
CPU-only machine this was invisible -- the answer was always `cpu`, requested or
not. On a GPU machine it is a real defect: `device: cpu` runs on the GPU, and
`device: cuda` on a GPU-less machine would run on the CPU rather than refusing.
This was found the first time the port ran on real GPU hardware, and fixed by
giving `2_vae` the same `resolve_device` the other methods already had. The
guard against it is a test that asks for `cpu` where a GPU exists and one that
asks for `cuda` where none does; neither needs a GPU to run.

---

## 5. Reading order for GPU work

    cat docs/PLATFORMS.md   # where a job runs; the platform layer
    cat docs/GPU.md         # this file; the GPU environment and the device invariant
    # then the method you are touching:
    cat methods/<m>/README.md
    cat methods/<m>/requirements.lock.cu130.txt   # if it has one yet
