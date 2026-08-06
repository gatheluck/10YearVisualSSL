# 10 Year Visual SSL

> ## 🚧 Work in progress — not a finished release
>
> This repository is under **active development** and is published early **on
> purpose**: public repositories get free CI on GitHub's standard runners, and
> the per-method test matrix had already exhausted the private-repo budget. It
> is **not** a stable or audited release.
>
> - Methods are ported **one at a time**; the layout, configs, APIs and results
>   **may change without notice**.
> - The formal audit and the planned move to the **`cvpaperchallenge`**
>   organisation have **not** happened yet.
> - **Nothing here has executed a full-scale training run.** Each method's
>   `README` states exactly what was and was not exercised.
>
> Please treat everything as provisional.

Ten years of visual-domain self-supervised learning (SSL) methods, ported to
**run in ordinary environments rather than on one specific supercomputer**.
Support for ABCI is separated into a loosely coupled module; the core does not
assume it.

## What this optimises for

1. **Reproducibility.** A result that cannot be reproduced is not a result.
   Every run records the configuration that actually ran, the artifacts it
   produced, and their hashes, and `bin/contract-test.py` decides by machine
   whether that record is complete and self-consistent
2. **Loose coupling to any compute facility.** The default is your own
   machine, and that path is complete on its own

## Status

| Component | Status |
|---|---|
| `bin/resolve-config.py` | **implemented and tested.** Produces the canonical resolved config and its `config_sha256` |
| `bin/contract-test.py` | **implemented and tested.** Decides by machine that a port is finished |
| `platforms/` | **implemented and tested.** Platform separation; `local` is self-contained |
| `methods/` | **thirteen methods ported and tested** (eleven with a linear evaluation; `36_franca` is the first eval-only port, with no step 1). The per-method table is below under [Methods](#methods) |
| `bin/launch.py` | **implemented and tested.** One command: resolve, submit, verify, record |
| `adapterlib/` | **implemented and tested.** The one place a `run_manifest.json` is written |
| `LICENSE` | **MIT** (Copyright (c) 2026 LIMIT.Lab) |

Three adapters exist and the chain runs end to end on a CPU: a configuration
becomes the exact bytes a run is identified by, an adapter produces
`encoder.pt`, `metrics.json` and `run_manifest.json`, and `contract-test`
decides by machine whether the port is finished. Every command below was run
to produce the output shown.

**What is not yet reproducible is a full-scale training run.** The recipes in
the shipped configs are the ones the captured cluster runs used — hundreds of
GPU-hours on ImageNet — and nothing here has executed one. Each method's
README says exactly what was and was not exercised.

## Methods

Every method ported so far lives under `methods/`, one directory each, with its
own locked environment and adapter. **Step 1** is the self-supervised
pretraining; **linear eval** is the frozen-feature linear probe that produces
the downstream numbers this project exists to compare. Directory names are
zero-padded so they sort in numeric order.

| Directory | Method | Stages | Notes |
|---|---|---|---|
| `01_context_prediction` | Context Prediction — Doersch, Gupta & Efros, ICCV 2015 | step 1 + linear eval | the first pilot; verified on a CPU end to end |
| `02_vae` | VAE — Kingma & Welling, 2013 | step 1 | pretext-only; the one method on MNIST, so it trains to completion on a CPU |
| `04_context_encoder` | Context Encoder — Pathak et al., 2016 | step 1 + linear eval | the one GAN (two models, two optimisers); `encoder.pt` is the conv encoder + bottleneck |
| `05_jigsaw_puzzle` | Jigsaw Puzzles — Noroozi & Favaro, ECCV 2016 | step 1 + linear eval | a self-contained re-implementation (the lab's own CFN AlexNet, no submodule); predicts which permutation reordered the 3×3 tiles; `encoder.pt` is the shared CFN encoder |
| `17_swav` | SwAV — Caron et al., 2020 | step 1 + linear eval | its loader could not run on one process; the sampler is now conditional |
| `20_simsiam` | SimSiam — Chen & He, 2020 | step 1 + linear eval | the second method to produce comparable downstream numbers |
| `21_barlow_twins` | Barlow Twins — Zbontar et al., 2021 | step 1 + linear eval | refuses fp16 on a CPU rather than downgrading quietly |
| `25_mae` | MAE — He et al., CVPR 2022 | step 1 + linear eval | a self-contained re-implementation (the lab's own MAE, no submodule); masked-autoencoder pretraining; `linear_eval` probes the trained encoder (avg-pooled patch tokens) |
| `27_ibot` | iBOT — Zhou et al., 2021 | step 1 + linear eval | first exercised on an A100 as written; `encoder.pt` is the teacher ViT |
| `36_franca` | Franca — arXiv:2507.14137 | linear eval only | the first **eval-only** port (no step 1): probes the frozen pretrained Franca ViT-B/14 CLS token, a genuine comparable representation. Backbone is a hash-pinned download; from-scratch SSL pretraining is the excluded step 2 (CONTRACT §7, docs/EVAL_DOWNLOAD.md) |
| `image_gpt` | iGPT — Chen et al., ICML 2020 | step 1 + linear eval | a self-contained re-implementation (no submodule) ported from the lab's inline model; pretrains a GPT on colour-cluster tokens; `linear_eval` probes the trained model, a genuine comparable number |
| `mar` | MAR — Li et al., NeurIPS 2024 | step 1 | the first `submodule+patch` port: the model is the pinned `third_party/mar` fork, imported not copied; `linear_eval` deferred — its captured eval path is unrecoverable (CONTRACT §7, docs/EVAL_DOWNLOAD.md) |
| `var` | VAR — Tian et al., NeurIPS 2024 | step 1 + linear eval | the first `submodule+adapter` port: `third_party/var` pinned directly (no fork). Next-scale autoregressive generation; `linear_eval` probes the pretrained VQVAE **tokeniser** (a hash-pinned download), which measures the fixed tokeniser rather than VAR's learned representation (CONTRACT §7, docs/EVAL_DOWNLOAD.md) |

Ten produce **comparable** `linear_probe` accuracy on a genuinely learned
representation. `02_vae` is pretext-only and `mar` has no linear eval; `var`'s
`linear_eval` probes a fixed pretrained tokeniser rather than its own learned
representation, so its number is not comparable in the same sense
(`docs/EVAL_DOWNLOAD.md`). `methods/_reference/` is not a method under study but
the known-good adapter the contract tests run against.
Deferred, not dropped: `VideoGen` (LTX-2), which needs CUDA > 12.7 and a 22B
checkpoint.

## Repository layout

`exists` is in the tree and under test. `planned` is not written yet, and is
shown so the shape is visible before it is built.

```
10YearVisualSSL/
├── bin/                            command-line tools                exists
│   ├── resolve-config.py             authoring config -> canonical resolved JSON
│   ├── launch.py                     resolve, submit, verify, record
│   ├── verify-environment.py         is this the locked environment?
│   ├── run-ci-locally.py            run the workflow here, by reading it
│   ├── mutate.py                     break the code, check the tests notice
│   ├── build-lock.py                 render a resolved set into a CPU lock
│   ├── fetch-weights.py              download a pinned, hash-checked backbone
│   └── contract-test.py              decides by machine that a port is finished
├── adapterlib/                     the one place a run_manifest.json is written
│   └── __init__.py                                                    exists
├── methods/                        one directory per method
│   ├── _reference/                   known-good adapter; trains nothing exists
│   │   └── adapter/
│   │       ├── __init__.py             the body: what this method does
│   │       └── __main__.py             python -m adapter --config ... --out ...
│   ├── 01_context_prediction/         first pilot, step 1              exists
│   │   ├── Dockerfile                  the locked environment as an image
│   │   ├── adapter/                    translates the config, calls the original
│   │   ├── train_step1_alexnet_official.py   the original loop, extracted
│   │   ├── models/ data/               untouched; digests pinned by tests
│   │   ├── configs/step1.yaml          the settings the capture used
│   │   ├── configs/linear_eval.yaml    stage 2: frozen-features evaluation
│   │   ├── requirements.txt            which packages, checked against imports
│   │   ├── requirements.lock.txt       exact versions, to rebuild a run
│   │   ├── provenance.json             what came across, and what changed
│   │   └── README.md                   the science, and the port's deviations
│   ├── 02_vae/                        second method, step 1            exists
│   │   ├── adapter/  configs/  models/  data/
│   │   ├── Dockerfile  requirements.lock.txt  provenance.json
│   │   └── README.md                   MNIST; trains to completion on CPU
│   └── VideoGen/                     deferred: needs a GPU            planned
│       ├── adapter/
│       └── configs/
├── third_party/                    authors' code, untouched          exists
│   ├── mar/                          pinned submodule (fork), used by methods/mar
│   ├── var/                          pinned submodule, used by methods/var
│   └── franca/                       pinned submodule, used by methods/36_franca
├── mutations/                      mutation specs, with their measured results
├── platforms/                      where a job runs. Loosely coupled  exists
│   ├── base.py                       the shared interface, free of platform terms
│   ├── local/backend.py              this machine. The default, self-contained
│   └── abci/backend.py               optional

├── configs/                        shared bases that methods include  planned
├── runs/                           run outputs. Not tracked           exists
│   └── <method>-<config sha>/        named after the config, not the clock
│       ├── launch.json               what was asked, and how it turned out
│       ├── resolved.json             the exact config that ran
│       └── out/
│           ├── encoder.pt
│           ├── metrics.json
│           └── run_manifest.json
├── docs/
│   ├── PLATFORMS.md                  the platform separation          exists
│   ├── EVAL_DOWNLOAD.md              generative-method probes + weights
│   ├── PORTING_ROADMAP.md            the 37 Step 1&2 methods, order, status
│   └── GPU.md                        GPU env + the device invariant   exists
├── tests/                          one file per unit, plus the chain  exists
│   ├── test_resolve_config.py
│   ├── test_adapterlib.py
│   ├── test_contract_test.py
│   ├── test_end_to_end.py            resolve -> adapt -> verify, for real
│   ├── test_method_requirements.py   declarations match the imports
│   ├── test_launch.py                resolve -> submit -> verify -> record
│   ├── test_ci.py                    the workflow runs what it claims
│   ├── test_mutate.py                the mutation tool cannot lie
│   ├── test_method_17_swav.py         the fifth port
│   ├── test_method_21_barlow_twins.py the fourth port
│   ├── test_metric_vocabulary.py     one vocabulary, and what may be compared
│   ├── test_encoder_convention.py    every port loads back what it wrote
│   ├── test_method_20_simsiam.py     the third port
│   ├── test_no_hard_coded_methods.py shared machinery discovers methods
│   ├── test_repository_scan.py       one scan, and it works without git
│   ├── _repo_files.py                which files belong to the repository
│   ├── test_language.py              everything here is in English
│   └── test_repo_hygiene.py          nothing generated is tracked
├── .github/workflows/tests.yml     CI: the suite on linux x86_64      exists
├── CLAUDE.md                       the working rules                  exists
├── README.md
└── LICENSE                         MIT, for our code only             exists
```

The design of record is not here — see *Where the design lives* below.

**`methods/_reference` is not an example that might rot.** It is exercised by
`tests/test_end_to_end.py` through the real chain, so a later adapter can copy
something known to pass.

### Where third-party code goes

**`third_party/<name>/`, one pinned submodule per upstream repository, and
nothing of ours inside it.**

The authors' tree is never edited. Everything we write about a method lives in
`methods/<name>/adapter/`, which imports from the submodule:

```
third_party/deepcontext/          <- the authors' repository, at a pinned commit
methods/01_context_prediction/
    adapter/__init__.py           <- ours: imports deepcontext, writes the outputs
```

**One shared directory, not one per method,** because the originals already
share: measured across the 31 recorded upstream repositories, `cosmos`
(`444d86120a57`) and `vggt-omega` (`39a0cb8af885`) each appear under two
different methods **at the same commit**. Per-method placement would make each
of those two submodules, two clones, and two pins that can drift apart without
anything noticing.

It also settles a naming question the originals leave open. They spell the
same idea four ways — `third_party/` (15), `external/` (7), `repo/` (5),
`repos/` (4). One spelling here.

Where a method needs the upstream code *changed*, the change is a branch on
our fork and the submodule pins a commit on that branch, so this repository
still contains no modified copy of anyone's work. Of the 31 recorded
repositories, 24 are expected to need only an adapter and 7 a patch; the
per-repository detail is in `docs/INVENTORY.md` on the Capture side.

## Requirements

Python 3.10 or newer. **The core needs nothing installed**; it is standard
library only, so it also runs on a login node that forbids extra packages.

```bash
python3 --version
```

Writing configs in YAML is optional and needs one package. JSON configs need
nothing, and the resolved artifact is JSON either way.

```bash
python3 -m pip install pyyaml     # only if you want to author in YAML
```

`./tests/run-tests.sh` prints whether PyYAML is present, so a skipped test is
never mistaken for a passing one.

**Each method declares its own training dependencies**, in its
`requirements.txt` (which packages) and `requirements.lock.txt` (**the full
closure**, every version exact and every wheel hashed). The interpreter is
pinned in `.python-version`. `tests/test_method_requirements.py` checks a
method's declarations against what it actually imports, in both directions,
refuses a lock that is not a closure, and refuses an entry without a hash. The
core never imports any of them.

## Reproducibility: the resolved config

**`config_sha256` is the hinge.** `run_manifest.json` claims that one
configuration produced one result, and that claim is worth something only if
the same configuration always hashes the same way. `bin/resolve-config.py`
produces that canonical form.

Write the authoring configs — `include` lets a method reuse a shared base.
**These keys are illustrative**; each method defines its own, and
`methods/01_context_prediction/configs/step1.yaml` is a real one:

```bash
mkdir -p configs && printf '{"seed":0,"optimizer":{"name":"sgd","lr":0.1,"momentum":0.9}}\n' > configs/base.json
printf '{"include":["base.json"],"optimizer":{"lr":0.03},"data_root":"${DATA_ROOT}"}\n' > configs/example.json
```

Resolve. Values come from `--set`, never from the environment:

```bash
python3 bin/resolve-config.py --config configs/example.json --out runs/demo/resolved.json --set DATA_ROOT=/mnt/data
```

```
  wrote runs/demo/resolved.json
  config_sha256 0639d99a22108b2548335912300c2905e1b05767feab17091a78ad4f0c47d813
```

The resolved file is one line, keys sorted, `include` gone, `${DATA_ROOT}`
gone, and `optimizer.lr` overridden while `momentum` survives the merge:

```json
{"data_root":"/mnt/data","optimizer":{"lr":0.03,"momentum":0.9,"name":"sgd"},"seed":0}
```

Check the hash with anything you like — it is a plain sha256 of those bytes:

```bash
shasum -a 256 runs/demo/resolved.json
```

To get the hash without writing anything:

```bash
python3 bin/resolve-config.py --config configs/example.json --print-hash --set DATA_ROOT=/mnt/data
```

### What it refuses, and why

**The environment is never read.** A config that silently absorbs the machine
it was resolved on is not reproducible, so an unset variable stops the run and
nothing is written:

```bash
python3 bin/resolve-config.py --config configs/example.json --out /tmp/x.json; echo "EXIT=$?"
```

```
  *** data_root: DATA_ROOT is not set. The environment is never consulted -- pass it with --set DATA_ROOT=<value>
EXIT=1
```

It also refuses, by name: an `include` that is missing, cyclic, or reaches
outside the config root; a duplicate key (one value would be dropped without a
word); `NaN` and `Infinity`; and anything JSON cannot carry. Nothing is
skipped quietly.

**Why JSON and not YAML for the resolved file:** JSON has a canonical form
reachable from the standard library, and YAML has neither. `gpus: 8` and
`gpus: 8.0`, quoted and unquoted strings, differing key order — all change the
bytes without changing the settings, which would make the hash meaningless.

## Running the tests

Every push runs them on linux x86_64 as well —
[`.github/workflows/tests.yml`](.github/workflows/tests.yml). The pre-commit
hook is machinery, but machinery on one machine: it needs configuring per
clone, `--no-verify` skips it, and it only ever exercises the platform the
committer happens to have. CI closes all three, and it is where the linux
x86_64 claim is actually checked — everything end-to-end before it ran on
macOS arm64.

| Job | When | What |
|---|---|---|
| `core` | every push | the suite with **nothing installed** |
| `locked` | every push | install from the lock hash-checked, verify the environment, run everything |
| `container` | pull requests | build the image and run the contract chain inside it |

The workflow is itself under test (`tests/test_ci.py`): **a CI job that cannot
fail is worse than no CI**, so `continue-on-error`, `|| true`, `set +e` and
`if: always()` are refused, and the actions are pinned to commit SHAs rather
than tags.

**Decide by exit status.** Grepping the output for a success string misses
failures.

```bash
./tests/run-tests.sh; echo "EXIT=$?"
```

Once per clone, so that the pre-commit hook is active:

```bash
git config core.hooksPath .githooks
```

## Training a method

The whole chain, with the one method that is ported. Each step is checked by
`tests/test_method_01_context_prediction.py`, which runs exactly this sequence
on synthetic images.

**1. Build the environment.** Only the training needs packages; the tools
below need nothing installed.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r methods/01_context_prediction/requirements.lock.txt -r requirements-tools.lock.txt
```

Swap the index for a CUDA one (`.../whl/cu121`) to get a GPU build at the same
versions. See the method's README for what that does and does not guarantee.

`requirements-tools.lock.txt` is PyYAML and nothing else, needed only for step
2 to read a YAML authoring config. Omit it if your config is JSON. It is not a
method dependency, and no method declares it.

**2. Resolve the config**, so the run is identified by a hash rather than by a
file somebody may edit afterwards:

```bash
python3 bin/resolve-config.py --config methods/01_context_prediction/configs/step1.yaml --out runs/ctxpred/resolved.json --set DATA_ROOT=/path/to/ILSVRC2012
```

**3. Run the adapter.** It runs from the method's directory, and reaches this
repository through `PYTHONPATH`:

```bash
cd methods/01_context_prediction && PYTHONPATH=../.. python3 -m adapter --config ../../runs/ctxpred/resolved.json --out ../../runs/ctxpred/out; echo "EXIT=$?"
```

**4. Check the result against the contract**, passing the adapter's exit
status so that both signals have to agree:

```bash
python3 bin/contract-test.py --out runs/ctxpred/out --config runs/ctxpred/resolved.json --exit-status <the status from step 3>
```

`runs/ctxpred/out` then holds `encoder.pt`, `metrics.json`,
`run_manifest.json`, and a `work/` directory with the training run's own
checkpoints and logs. Nothing is written outside it.

The method's own README covers what it implements, which settings it uses and
where each came from, and what changed during the port:
[methods/01_context_prediction/README.md](methods/01_context_prediction/README.md).

## One command for a whole run

The steps above are what the launcher does, in order, so that nobody has to
remember the order:

```bash
python3 bin/launch.py --config methods/01_context_prediction/configs/step1.yaml --method 01_context_prediction --set DATA_ROOT=/path/to/ILSVRC2012
```

It resolves the config, submits the job through the chosen platform, verifies
the outputs against the contract, and records the invocation. Three decisions
worth knowing:

- **The run directory is named after the configuration**, not the clock:
  `runs/<method>-<config_sha256[:12]>/`. Two runs of one experiment then
  collide, which is information rather than a nuisance — you meant to change
  something and did not. `--again` repeats one deliberately
- **`--gpus` and `--hours` are launcher arguments, not config keys.** How long
  a scheduler is asked to allow does not change the result, and folding it
  into the config would make two identical experiments hash differently. What
  *does* affect the result, `WORLD_SIZE`, is recorded by the run itself
- **A submitted job is not a finished one.** Where the platform can report how
  it went, the launcher verifies immediately; where it can only queue the work
  it says `submitted` and stops, rather than checking an output directory
  nothing has written yet. `--verify-only <run-dir>` finishes the job later
- **The distribution variables are set, not inherited.** `WORLD_SIZE`, `RANK`
  and `LOCAL_RANK` are stated explicitly, because `adapterlib` records
  `WORLD_SIZE` in the manifest and a value left over in your shell would
  otherwise be written into the results. Multi-process fan-out is **not
  implemented**: `--processes` above 1 is refused rather than approximated,
  and `--gpus` is a resource request that does not imply a process count

Alongside the outputs it writes `launch.json`: the authoring config, the
substitutions, the platform, the resources and the verdict. The manifest says
what the run *did*; this says what was *asked of it*.

```
runs/_reference-dd6145f9edb6/
├── launch.json          what was asked, and how it turned out
├── resolved.json        the exact config that ran
└── out/                 encoder.pt, metrics.json, run_manifest.json
```

## Running CI here

CI runs on GitHub. When it cannot — the account hit a billing wall on
2026-07-31 — the same jobs can be run on this machine, provided Docker is
available:

```bash
python3 bin/run-ci-locally.py --event pull_request
```

**It reads `.github/workflows/tests.yml` and executes what it finds.** It
contains none of the workflow's commands, and `tests/test_run_ci_locally.py`
asserts that: a script that restated them would be a second implementation of
the workflow, and the two would agree until the day one was edited — at which
point "CI passed locally" would be false with nothing able to notice.

Each matrix job gets its own export of `HEAD`, so a run says something about
the committed tree rather than about whatever is open in an editor.

What it cannot reproduce, it prints:

- the image is not GitHub's runner image
- `linux/amd64` is emulated here. Results have matched, but it is not the same
  silicon, and this repository already says agreement across different
  hardware is not guaranteed
- `uses:` steps are actions, not shell; their effect is provided differently
  and noted rather than executed

`--dry-run` prints the plan, with the matrix resolved, without running it.

## Checking an adapter's output against the contract

**`contract-test` is how "the port is finished" is decided by a machine**
rather than by opinion.

```bash
python3 bin/contract-test.py --out <dir> --config runs/demo/resolved.json --exit-status <n>
```

**Success requires two signals to agree:** exit status 0 *and* `status: "ok"`
in `run_manifest.json`. Neither is trusted alone — on the Capture side a gate
once returned exit 0 while reporting detected secrets.

The tool also refuses any file in `--out` that the manifest does not list. An
output nobody knows about is a hole in reproducibility.

## What may be compared with what

`metrics.json` carries two blocks:

```json
{"schema_version": 2,
 "metrics": {"final_linear_probe_top1_accuracy": 42.3},
 "metrics_raw": {"final_top1_acc": 42.3}}
```

`metrics` uses a **fixed vocabulary**; `metrics_raw` keeps the names the
original gave its own numbers, so nothing is lost in translation. Both are
required, and `contract-test` reads the vocabulary from the same place the
adapters write it, so the two cannot drift apart.

The distinction the vocabulary exists for is `pretext` against
`linear_probe`:

| Prefix | What it measures | Comparable across methods |
|---|---|---|
| `pretext` | the method's **own** training objective or task | **No** |
| `linear_probe` | downstream classification from a linear probe, against real labels | **Yes** |
| `epochs_completed`, `steps_completed` | counters | in kind only |

The first port's `val_acc1` is eight-way patch-position accuracy — its own
pretext task. A linear probe's top-1 is real classification accuracy. Putting
those in one column produces a comparison table that is wrong and looks right,
so **the stage decides which family a port may use**, and reaching across is
refused rather than discouraged. Accuracies are percentages, 0 to 100, which
was measured from both sources rather than assumed.

## What reproducibility means here

Worth stating plainly, because the honest answer is narrower than "it is
reproducible".

**Guaranteed — and measured.** The same environment and the same config give
the same bits. `bin/resolve-config.py` makes a config identify itself by hash;
each method's lock file pins the whole dependency closure with hashes and the
interpreter is pinned; the ported training path asks torch for deterministic
kernels; and the tests compare two runs by the hash of the artifact they
produce.

**Not guaranteed, and not achievable by any means.** Agreement across
different hardware. Floating-point addition is not associative, so a different
CPU, GPU, BLAS or cuDNN reorders the arithmetic and the low bits move. No
amount of pinning changes this, and a project claiming otherwise is
mismeasuring.

**Therefore, what must always hold: a difference is explainable.** Every
`run_manifest.json` records the complete set of installed packages with their
versions, a `packages_sha256` over the lot, and the system and machine. When
two runs disagree, the manifests say why.

And the record is checked, not merely kept. `bin/verify-environment.py`
answers both halves with one comparison — *is this environment the locked
one*, and *did that run use it*:

```bash
python3 bin/verify-environment.py --lock methods/01_context_prediction/requirements.lock.txt --lock requirements-tools.lock.txt [--manifest runs/<id>/out/run_manifest.json]
```

Any difference fails, including a package no lock mentions: something
installed that the lock does not describe means the environment cannot be
rebuilt from it. The one exemption is `pip`, which `python -m venv` seeds
before any lock is read — and it is *reported* as ignored, with the reason,
rather than passed over.

**Measured, on 2026-07-30.** The container was built and run on both
architectures Docker offers here, and the chain completed on each:

```
linux/arm64   same config twice -> encoder.pt bd90b98ef87dea0265bb…  (identical)
linux/amd64   same config twice -> encoder.pt cbf177068c3dcd6583f5…  (identical)
```

The two architectures **disagree with each other**, which is the guarantee
working rather than failing — and the manifests record `aarch64` and `x86_64`,
so the disagreement is explained rather than mysterious.

**Not verified.** Apptainer, and any GPU: neither is available on the machine
this was written on, so the `apptainer` commands in the method's README come
from its documentation rather than from a run, and the CUDA determinism
settings are unexercised.

## Execution platforms

**Running on any particular compute facility is optional.** The default is
your own machine (`platforms/local`), and that path is complete on its own.
Platform support lives in loosely coupled modules, and the core assumes none
of them.

The separation is **held by machinery** —
`tests/test_platform_isolation.py`. See [docs/PLATFORMS.md](docs/PLATFORMS.md).

## License

**The code in this repository is MIT** (`LICENSE`,
Copyright (c) 2026 LIMIT.Lab).

MIT covers **only the code we wrote**.

| Subject | Treatment |
|---|---|
| our code | MIT |
| authors' published code | **never copied.** Referenced as a pinned submodule; each repository's own license applies unchanged |
| anything judged a derivative work | **not included here** (the 4 entries marked `derivative` in `official-manifest.txt` on the Capture side) |

Licenses and treatment for all 31 author repositories are in
`docs/INVENTORY.md` on the Capture side. **Twelve of them are
non-commercial.** Referencing by submodule means no redistribution occurs,
but **whether they may be used is a separate judgement.**

## Where the design lives

**The design of record is the Capture repository**
(`gatheluck/10YearVisualSSLCapturePrivate`, private forever), so that there is
one origin even as the implementation moves here.

| Document | Contents |
|---|---|
| `docs/DESIGN.md` | the philosophy and the reasoning |
| `docs/CONTRACT.md` | **the adapter contract** |
| `docs/INVENTORY.md` | inventory of the 31 author repositories and recommended treatment |

## Guards against repeated mistakes

Seven kinds of mistake recurred often enough here to be worth mechanising
rather than remembering. They are listed with their counts in
[CLAUDE.md](CLAUDE.md); the mechanisms are:

| Mechanism | What it prevents |
|---|---|
| `bin/mutate.py` | assertions that cannot fail, and mutation reports that lie. An absent or ambiguous anchor is an error, bytecode is never reused, and it refuses to mutate anything outside its own copy |
| `tests/test_no_hard_coded_methods.py` | shared machinery naming one method. A list looks right until the second method arrives |
| `tests/test_repo_hygiene.py` | the README layout drifting from the tree — in **both** directions |
| `tests/test_ci.py` | a CI job that cannot fail |
| `tests/test_language.py`, `tests/test_platform_isolation.py` | prose and platform vocabulary leaking where they do not belong |
| `tests/test_repository_scan.py` | the same rule implemented twice. Two copies of "which files belong to this repository" agreed everywhere git existed and diverged inside the container image. It pins one implementation, and proves the scan works with git actually removed from `PATH` |

Each was written after the same mistake had been made two or three times. The
counts are in the commit history and are not flattering; they are recorded
because a mistake nobody counted is a mistake that recurs.

The last row was added after the fourth occurrence, in the commit that
introduced the row above it -- which is the honest measure of how little a
written rule achieves on its own. What the guards do not cover, the container
job does: it runs the suite in an image with no `.git`, no `.github` and no
git binary, which is the closest thing here to what a reader downloads.

## Development

Strict TDD. The rules are in [CLAUDE.md](CLAUDE.md). Everything in this
repository is written in English, enforced by `tests/test_language.py`.
