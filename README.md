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
| `bin/contract-test.py` | **implemented and tested.** Decides by machine that a port is finished |
| `platforms/` | **implemented and tested.** Platform separation; `local` is self-contained |
| adapters | not started. Pilots are `1_context_prediction` and `VideoGen` (LTX-2) |
| launcher | not started |
| `LICENSE` | **MIT** (Copyright (c) 2026 LIMIT.Lab) |

Because no adapter exists yet, there is no end-to-end reproduction procedure
to run. The steps below are what exists and can be executed today; the
reproduction procedure is written here once the first pilot lands.

## Requirements

Python 3.10 or newer, standard library only. Nothing to install.

```bash
python3 --version
```

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
python3 bin/contract-test.py --out <dir> --config <resolved.yaml> --exit-status <n>
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
