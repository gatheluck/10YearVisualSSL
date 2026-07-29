# 10 Year Visual SSL

Ten years of visual-domain self-supervised learning (SSL) methods, ported to
**run in ordinary environments rather than on one specific supercomputer**.
Support for ABCI is separated into a loosely coupled module; the core does not
assume it.

**Currently private. It will be published after an audit.**

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
| adapters | not started. Pilots are `1_context_prediction` and `VideoGen` (LTX-2) |
| launcher | not started |
| `LICENSE` | **MIT** (Copyright (c) 2026 LIMIT.Lab) |

No adapter exists yet, so a full training run cannot be reproduced today. The
part of the chain that does exist — turning a configuration into the exact
bytes a run is identified by — is complete, and every command below was run to
produce the output shown.

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

## Reproducibility: the resolved config

**`config_sha256` is the hinge.** `run_manifest.json` claims that one
configuration produced one result, and that claim is worth something only if
the same configuration always hashes the same way. `bin/resolve-config.py`
produces that canonical form.

Write the authoring configs — `include` lets a method reuse a shared base:

```bash
mkdir -p configs && printf '{"seed":0,"optimizer":{"name":"sgd","lr":0.1,"momentum":0.9}}\n' > configs/base.json
printf '{"include":["base.json"],"method":"1_context_prediction","optimizer":{"lr":0.03},"data_root":"${DATA_ROOT}"}\n' > configs/ctxpred.json
```

Resolve. Values come from `--set`, never from the environment:

```bash
python3 bin/resolve-config.py --config configs/ctxpred.json --out runs/demo/resolved.json --set DATA_ROOT=/mnt/data
```

```
  wrote runs/demo/resolved.json
  config_sha256 0639d99a22108b2548335912300c2905e1b05767feab17091a78ad4f0c47d813
```

The resolved file is one line, keys sorted, `include` gone, `${DATA_ROOT}`
gone, and `optimizer.lr` overridden while `momentum` survives the merge:

```json
{"data_root":"/mnt/data","method":"1_context_prediction","optimizer":{"lr":0.03,"momentum":0.9,"name":"sgd"},"seed":0}
```

Check the hash with anything you like — it is a plain sha256 of those bytes:

```bash
shasum -a 256 runs/demo/resolved.json
```

To get the hash without writing anything:

```bash
python3 bin/resolve-config.py --config configs/ctxpred.json --print-hash --set DATA_ROOT=/mnt/data
```

### What it refuses, and why

**The environment is never read.** A config that silently absorbs the machine
it was resolved on is not reproducible, so an unset variable stops the run and
nothing is written:

```bash
python3 bin/resolve-config.py --config configs/ctxpred.json --out /tmp/x.json; echo "EXIT=$?"
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

**Decide by exit status.** Grepping the output for a success string misses
failures.

```bash
./tests/run-tests.sh; echo "EXIT=$?"
```

Once per clone, so that the pre-commit hook is active:

```bash
git config core.hooksPath .githooks
```

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

## Development

Strict TDD. The rules are in [CLAUDE.md](CLAUDE.md). Everything in this
repository is written in English, enforced by `tests/test_language.py`.
