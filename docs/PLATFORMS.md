# Execution platform separation

Last updated: 2026-07-29

**Running on any particular compute facility is optional.** The core assumes
no platform. Support for a platform lives in a loosely coupled module and is
reached only through that module.

**This separation is held by machinery, not by documentation.**
`tests/test_platform_isolation.py` fails if it is broken. We have seen, with a
concrete example, that a policy written down is not a policy that holds
(Capture repository, `docs/DESIGN.md` §5.26).

---

## 1. Structure

```
platforms/
  base.py            the shared interface. **No platform vocabulary here**
  __init__.py        resolves by name, dynamically. **No table**
  local/backend.py   runs on this machine. The default, and self-contained
  <name>/backend.py  anything else, all optional
```

The core obtains a backend through `platforms.load_backend(name)`.
**The name comes from the caller.** No platform name is written in the code:
the moment one is, the core knows about that platform.

```python
from platforms import load_backend, JobSpec

backend = load_backend(args.platform)      # defaults to "local"
backend.Backend().submit(JobSpec(
    name="ctxpred-pretrain", command=[...], env_name="py3.10_context_prediction",
    gpus=8, hours=24))
```

---

## 2. The interface

`JobSpec` states **what is needed, in ordinary words**.

| Field | Meaning |
|---|---|
| `name` | job name |
| `command` | the command to run |
| `env_name` | the conda environment to use |
| `gpus` / `hours` | **the amount needed.** Not a resource type or a queue name |
| `workdir` / `env` | working directory and environment variables |

**Translating that into a resource type is each backend's job.** The core
holds no translation table. Sharing one would mean the core knows that
platform's vocabulary.

`Backend` has exactly two methods.

| Method | Contract |
|---|---|
| `is_available()` | **Check, do not assume** — e.g. whether the submit command exists |
| `submit(spec)` | Run or enqueue, and return a `JobResult` |

`JobResult.exit_status` is **`None` when the outcome is not yet known.**
A backend that only enqueues must not return `0`. **0 means "it succeeded",
and an unknown outcome must never be passed off as a success.**

---

## 3. What the machinery holds

| Check | What breaking it would cause |
|---|---|
| Platform-specific vocabulary stays inside `platforms/<name>/` | the core becomes tied to that platform |
| `platforms/local/` always exists | an extra platform becomes mandatory and nothing runs locally |
| No platform-specific vocabulary in the interface | the interface is polluted and the separation is nominal |
| Nothing outside `platforms/` imports a specific platform | importing it *is* knowing about it |
| Resolution is **discovery, not a table** | a newly added platform is not found |
| Both backends implement the same interface | they cannot be swapped |

The last two are checked **by behaviour**: a dummy backend is dropped in to
see whether it is discovered, and `issubclass` is evaluated for real. Deciding
by string matching misfires — it flagged a usage example inside a docstring as
a hard-coded name (it actually did).

---

## 4. Adding a platform

1. Write `Backend(base.Backend)` in `platforms/<name>/backend.py`
2. Have `is_available()` **actually check** whether it can be used
3. Keep the need-to-resource translation **inside that module only**
4. Make `./tests/run-tests.sh` pass

**No registration step.** Put the file there and `available_backends()`
finds it.

---

## 5. Left undecided

- How an asynchronous backend waits for completion after enqueuing
  (polling or notification)
- Who assigns `WORLD_SIZE` across several nodes
- How logs are collected

**These are decided after the two pilot methods are through.** Deciding now
would produce a design that has never met a real job.

---

## 6. Running a test job on ABCI

The core stays platform-agnostic; this section is prose, so it may name the
platform. Everything machine-specific — the **group id** and any **environment
activation** — is injected at run time and never committed.

**Prerequisites (once per method):** check out the submodules
(`git submodule update --init`) and build the method's environment
(`.venvs/<method>/`, per `docs/GPU.md`).

**Data layout — one rule for every method.** `DATA_ROOT` is the dataset
**root**: a directory that contains a `train/` subdirectory (and `val/` for
linear evaluation), each holding the usual per-class image folders
(`train/<class>/*.JPEG`). Pretraining reads `train/`; linear evaluation reads
`train/` and `val/`. You pass the same `DATA_ROOT=<imagenet>` for step 1 and for
linear evaluation — never the `train` directory itself. This is resolved in one
place (`adapterlib.dataset_split_dir`) and held uniform by
`tests/test_data_root_convention.py`. Two methods are inherent exceptions and
say so in their own config: **`02_vae`** trains on MNIST (downloaded to
`DATA_ROOT`), and **`mar`** reads pre-encoded cached VAE latents rather than an
image folder.

**Submit a short run** (one GPU, one epoch), with the group id in the
environment so it never reaches the repository:

```
ABCI_GROUP=<your-group> python3 bin/launch.py \
  --config methods/<method>/configs/pretrain.yaml --method <method> \
  --platform abci --gpus 1 --hours 1 \
  --set DATA_ROOT=<imagenet-on-abci> \
  --override train.epochs=1 \
  --python "$PWD/.venvs/<method>/bin/python"
```

- `--override train.epochs=1` shortens the run (any existing setting works, e.g.
  `--override train.max_steps=50`); it lands in `config_sha256`, so a short run is
  recorded as the distinct run it is.
- `--python` names the interpreter the job runs. The job `cd`s into the method
  directory, so pass an **absolute** path — pointing it straight at the method's
  venv interpreter needs no activation, since that interpreter already holds the
  method's packages.
- `--setup` lines run before the command; use them only if the site needs a
  `module load` (or other activation). They are injected here, not stored in the
  repo, so nothing machine-specific is committed.
- `--gpus` maps to a resource type inside `platforms/abci/` only; `--hours` is the
  walltime. Both stay out of `config_sha256` (they do not change the result).

**One place holds everything: the run directory `runs/<method>-<hash>/`.** It is
named after the config hash and is self-contained, so it is the single thing to
inspect or hand over:

```
runs/<method>-<hash>/
  job.log                the job's full stdout+stderr in one file
  out/run_manifest.json  status (ok / failed) and the error message
  out/metrics.json       the numbers
  launch.json            job_id, what was asked, and the log path
  resolved.json          the exact config that ran
```

`job.log` is pinned there for **every** backend: on ABCI via `#PBS -o` (with
`-j oe` merging stderr), and locally by redirecting the command's output. It
opens with the environment diagnostics — hostname, `nvidia-smi`, the interpreter
and its `torch` / CUDA visibility, and `git submodule status` — then the run's
output, and on failure the trapped line, command and exit code. So one file
usually shows the cause: wrong interpreter, no GPU visible, a submodule not
checked out, or the adapter's own error. The launcher prints the run directory
path when it finishes.

**Checking the outcome.** Submission only enqueues, so the launcher records
`exit_status` as unknown and does not guess. After the job finishes, verify the
outputs against the contract:

```
python3 bin/launch.py --verify-only runs/<method>-<hash>
```

which is exit status 0 **and** `status: ok` in the manifest — the same bar
`contract-test` applies everywhere.

The §5 items (PBS-completion polling, `WORLD_SIZE` across nodes, log collection)
are still open and are settled once the two pilots have run.
