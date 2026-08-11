#!/usr/bin/env python3
"""Run one experiment: resolve, submit, verify, record.

    launch.py --config <authoring.yaml|json> --method <name>
              [--platform local] [--set KEY=VALUE ...]
              [--gpus N] [--hours H] [--processes 1]
              [--runs-dir runs] [--again]
    launch.py --verify-only <run-dir>

The pieces existed and nothing joined them. A scheduler backend could submit
a job and **nothing called it**; `resolve-config` and `contract-test` were run
by hand, in the right order, by whoever remembered the order.

No platform is named in this file, including in prose. The isolation guard
caught an earlier draft that named one in this very paragraph -- correctly:
this is core code, and the place to discuss a particular platform is the
documentation, not here.

Three decisions this makes, none of them plumbing:

**Resources are not part of the config.** `--gpus` and `--hours` are arguments
here, deliberately outside `config_sha256`: how long a scheduler is asked to
allow does not change the result, and folding it in would make two identical
experiments hash differently. What *does* affect the result -- `WORLD_SIZE` --
is recorded by the run itself, in its manifest.

**A submitted job is not a finished one.** Where the backend reports how it
went, this verifies straight away. Where it cannot -- a scheduler that has
only queued the work, which reports `exit_status = None` -- it says so and
stops, rather than checking an output directory nothing has written yet.

**The invocation is recorded.** `run_manifest.json` says what the run did; it
cannot say what was asked of it. `launch.json` holds the authoring config, the
substitutions, the platform and the resources, so a run directory explains
itself without the shell history that produced it.

The run directory is named after the configuration, not the clock. Two runs of
one experiment then collide, which is information rather than a nuisance: you
meant to change something and did not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platforms import JobSpec, available_backends, load_backend  # noqa: E402

SCHEMA_VERSION = 1
LAUNCHER_VERSION = 1
RECORD = "launch.json"
RESOLVED = "resolved.json"
OUT = "out"
DEFAULT_PLATFORM = "local"


class LaunchError(Exception):
    """A refusal, always naming what was refused."""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True,
                          text=True, **kw)


def should_verify(exit_status: int | None) -> bool:
    """Whether there is anything to check yet.

    `None` means the backend only queued the work. Checking then would report
    contract violations for a job that has not started.
    """
    return exit_status is not None


def summarise(exit_status: int | None, contract_ok: bool | None) -> dict:
    """The outcome, from the two signals that have to agree.

    Success needs both: a zero exit status and outputs that satisfy the
    contract. Neither alone -- the same rule `contract-test` applies, applied
    once more where the two are finally in the same place.
    """
    if exit_status is None:
        outcome = "submitted"
    elif exit_status == 0 and contract_ok:
        outcome = "ok"
    else:
        outcome = "failed"
    return {"exit_status": exit_status, "contract_ok": contract_ok,
            "outcome": outcome}


def config_digest(resolved: Path) -> str:
    import hashlib
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def resolve(config: Path, into: Path, sets: dict,
            overrides: list[str] | None = None) -> Path:
    """Produce the canonical resolved config, or refuse."""
    into.mkdir(parents=True, exist_ok=True)
    out = into / RESOLVED
    cmd = [sys.executable, ROOT / "bin" / "resolve-config.py",
           "--config", config, "--out", out]
    for k, v in sets.items():
        cmd += ["--set", f"{k}={v}"]
    for item in overrides or []:
        cmd += ["--override", item]
    r = _run(cmd)
    if r.returncode != 0:
        raise LaunchError(f"the config could not be resolved:\n{r.stderr.strip()}")
    return out


def job_environment(gpus: int, processes: int = 1) -> dict:
    """What the job runs with. **Stated, never inherited.**

    The contract makes the launcher responsible for the distribution
    variables. Leaving them unset does not mean "one process" -- it means
    whatever the surrounding shell happens to hold, and `adapterlib` records
    WORLD_SIZE in the manifest. A value left over from something else would
    have been written into the results as a fact about the run.

    Multi-process fan-out is not implemented. Rather than quietly running one
    process where several were asked for, that is refused; see the caller.
    """
    return {
        "PYTHONPATH": str(ROOT),
        "WORLD_SIZE": str(processes),
        "RANK": "0",
        "LOCAL_RANK": "0",
    }


def method_dir(name: str) -> Path:
    d = ROOT / "methods" / name
    if not (d / "adapter").is_dir():
        known = sorted(p.name for p in (ROOT / "methods").iterdir()
                       if (p / "adapter").is_dir())
        raise LaunchError(
            f"no method {name!r} with an adapter. Known: {', '.join(known)}")
    return d


def backend_for(name: str):
    try:
        return load_backend(name)
    except Exception as exc:
        raise LaunchError(
            f"{exc}. Known platforms: {', '.join(sorted(available_backends()))}"
        ) from None


def verify(run: Path, exit_status: int) -> tuple[bool, str]:
    """Check the outputs against the contract."""
    r = _run([sys.executable, ROOT / "bin" / "contract-test.py",
              "--out", run / OUT, "--config", run / RESOLVED,
              "--exit-status", exit_status])
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def launch(config: Path, method: str, runs_dir: Path, platform: str,
           sets: dict, gpus: int, hours: int, again: bool,
           processes: int = 1, python: str | None = None,
           setup: list[str] | None = None,
           overrides: list[str] | None = None) -> tuple[int, dict]:
    if processes != 1:
        raise LaunchError(
            f"--processes {processes}: multi-process fan-out is not "
            "implemented. Running one process where several were asked for "
            "would produce a result that looks like the one requested, so "
            "this is refused rather than approximated")
    md = method_dir(method)
    mod = backend_for(platform)

    # Resolved once into a staging area, because the run directory is named
    # after the digest and the digest is not known until it is resolved.
    # Cleared whatever happens: a failed resolution that leaves a directory
    # behind makes the next run look like a repeat of something.
    import shutil
    staging = runs_dir / ".staging"
    try:
        resolved = resolve(config, staging, sets, overrides)
        digest = config_digest(resolved)
        payload = resolved.read_bytes()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    run = runs_dir / f"{method}-{digest[:12]}"

    if run.exists() and not again:
        raise LaunchError(
            f"{run} already exists. The directory is named after the "
            "configuration, so this is the same experiment: pass --again to "
            "repeat it deliberately, or change the config")
    run.mkdir(parents=True, exist_ok=True)
    (run / RESOLVED).write_bytes(payload)

    spec = JobSpec(
        name=f"{method}-{digest[:12]}",
        command=[python or sys.executable, "-m", "adapter",
                 "--config", str((run / RESOLVED).resolve()),
                 "--out", str((run / OUT).resolve())],
        env_name=method, gpus=gpus, hours=hours, workdir=str(md),
        env=job_environment(gpus=gpus, processes=processes),
        setup=list(setup or []),
    )
    result = mod.Backend().submit(spec)

    contract_ok, report = None, ""
    if should_verify(result.exit_status):
        contract_ok, report = verify(run, result.exit_status)

    record = {
        "schema_version": SCHEMA_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "authoring_config": str(config),
        "config_sha256": digest,
        "method": method,
        "platform": platform,
        "gpus": gpus,
        "hours": hours,
        "set": dict(sets),
        "override": list(overrides or []),
        "python": python or sys.executable,
        "setup": list(setup or []),
        "job_id": result.job_id,
        **summarise(result.exit_status, contract_ok),
    }
    (run / RECORD).write_text(json.dumps(record, indent=2, sort_keys=True)
                              + "\n", encoding="utf-8")
    if report:
        print(report)
    return (0 if record["outcome"] == "ok" else 1), record


def verify_only(run: Path) -> tuple[int, dict]:
    run = Path(run)
    rec_path = run / RECORD
    if not rec_path.is_file():
        raise LaunchError(f"{run} has no {RECORD}; it was not launched here")
    record = json.loads(rec_path.read_text(encoding="utf-8"))
    status = record.get("exit_status")
    if not should_verify(status):
        raise LaunchError(
            f"{run} was only submitted (exit status unknown), so there is "
            "nothing to verify yet")
    contract_ok, report = verify(run, status)
    record.update(summarise(status, contract_ok))
    rec_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if report:
        print(report)
    return (0 if record["outcome"] == "ok" else 1), record


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise LaunchError(f"--set {item!r} is not KEY=VALUE")
        k, _, v = item.partition("=")
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path)
    ap.add_argument("--method")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--override", action="append", default=[],
                    metavar="DOTTED.KEY=VALUE",
                    help="change an existing config setting (e.g. "
                         "train.epochs=1 for a short run); lands in the hash")
    ap.add_argument("--python", default=None,
                    help="interpreter for the job's command (default: this "
                         "one). Point it at the method's venv for a cluster run")
    ap.add_argument("--setup", action="append", default=[], metavar="LINE",
                    help="a shell line to run before the command, e.g. to "
                         "activate an environment; may be repeated. Injected "
                         "at run time so nothing machine-specific is committed")
    ap.add_argument("--gpus", type=int, default=0,
                    help="a resource request for the scheduler. It does not "
                         "imply how many processes run")
    ap.add_argument("--processes", type=int, default=1,
                    help="processes to fan out to. Only 1 is implemented")
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--again", action="store_true",
                    help="repeat a configuration that has already been run")
    ap.add_argument("--verify-only", type=Path, metavar="RUN_DIR")
    a = ap.parse_args()

    try:
        if a.verify_only:
            rc, record = verify_only(a.verify_only)
        else:
            if not (a.config and a.method):
                raise LaunchError("--config and --method are both required")
            rc, record = launch(a.config, a.method, a.runs_dir, a.platform,
                                parse_set(a.set), a.gpus, a.hours, a.again,
                                a.processes, python=a.python, setup=a.setup,
                                overrides=a.override)
    except LaunchError as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
    print(f"  {record['outcome']}: {record['method']} "
          f"{record['config_sha256'][:12]} on {record['platform']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
