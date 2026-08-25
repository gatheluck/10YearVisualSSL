#!/usr/bin/env python3
"""Run the short real-run grid: every declared method x stage, one experiment each.

    matrix-run.py --data SHAPE=PATH [--data SHAPE=PATH ...]
                  [--method NAME ...] [--platform NAME]
                  [--runs-dir DIR] [--out matrix.json] [--python PATH]

`launch.py` runs one experiment; nothing ran the *grid* of them and decided, by
machine, that every method's weights and evaluations land where they should.
This does, and it is deliberately thin -- it reuses `launch.py` for each cell
(one implementation of resolve -> submit -> verify -> record, invoked, never
copied) and adds only three things over a hand-run loop.

**Discovered, never listed.** A method opts in by shipping a
`real_run_smoke.json` in its own directory; this finds them by looking, so a new
method with a spec joins the grid with no edit here. No method is named in this
file (tests/test_no_hard_coded_methods.py).

**Artifacts thread through a method's own stages.** A stage may ask for
`@encoder`, meaning the `encoder.pt` an earlier stage of the same method
produced, or `@produces:<file>` for any other file an earlier stage declared
(e.g. image_gpt's `clusters.npy`, which its linear_eval must quantise with);
the grid resolves each to that file on disk, and a request for one no earlier
stage produced is a hard, named failure rather than a silently empty `--set`.
`@data` is resolved from the `--data SHAPE=PATH` mapping for the spec's
`data_shape` -- a shape with no mapping is a hard, reported failure, never a
skipped cell that reads as a pass.

**The grid's verdict is one artifact.** `matrix.json` records every cell's run
directory and outcome, and the grid succeeds only when every cell does. One
failed cell fails the whole grid and its exit status, so "run everything for a
short epoch and check it all landed" is a machine judgment, not an impression.
A failed cell carries why (the tail of its output, and the path to its job log),
so nothing fails in silence.

The same driver serves the hermetic check (a synthetic-but-real-shaped data
root, the default backend) and a real cluster run (real data roots, a scheduler
backend, a GPU): only `--data` and `--platform` change. No platform is named
here; the backend is whatever `launch.py` resolves.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCH = ROOT / "bin" / "launch.py"
METHODS = ROOT / "methods"
SPEC_NAME = "real_run_smoke.json"
DEFAULT_PLATFORM = "local"
SCHEMA_VERSION = 1


class MatrixError(Exception):
    """A refusal, always naming what was refused."""


def discover_specs(methods_dir: Path = METHODS) -> list[tuple[str, dict]]:
    """Every method that declares a real-run smoke, found by looking."""
    found = []
    if not methods_dir.is_dir():
        return found
    for spec_path in sorted(methods_dir.glob(f"*/{SPEC_NAME}")):
        found.append((spec_path.parent.name,
                      json.loads(spec_path.read_text(encoding="utf-8"))))
    return found


def _new_run_dir(runs: Path, method: str, before: set) -> Path | None:
    """The single run directory launch.py just created, or None if it made none
    (or somehow more than one -- either is a failure the caller reports)."""
    after = set(runs.glob(f"{method}-*")) if runs.is_dir() else set()
    new = after - before
    return new.pop() if len(new) == 1 else None


def _produced_path(method: str, stage: dict, fname: str, produced) -> str:
    """The on-disk path of a file an earlier stage of this method produced.
    A request for one no earlier stage produced is a hard, named failure --
    never a silently empty `--set` that would read as a pass."""
    if not produced or fname not in produced:
        raise MatrixError(
            f"{method}/{stage['name']}: {fname} requested before any stage "
            "produced it")
    return str(Path(produced[fname]).resolve())


def stage_command(method: str, stage: dict, runs: Path, platform: str,
                  data_root, produced, python) -> list[str]:
    cmd = [sys.executable, str(LAUNCH), "--method", method,
           "--platform", platform, "--runs-dir", str(runs),
           "--config", str(METHODS / method / stage["config"])]
    if python:
        cmd += ["--python", str(python)]
    for key, val in stage.get("sets", {}).items():
        if val == "@data":
            val = str(data_root)
        elif val == "@encoder":
            val = _produced_path(method, stage, "encoder.pt", produced)
        elif isinstance(val, str) and val.startswith("@produces:"):
            val = _produced_path(method, stage, val[len("@produces:"):],
                                 produced)
        cmd += ["--set", f"{key}={val}"]
    for key, val in stage.get("overrides", {}).items():
        cmd += ["--override", f"{key}={val}"]
    return cmd


def run_stage(method: str, stage: dict, runs, platform: str, data_root,
              produced=None, python=None) -> dict:
    """Run one stage through launch.py; return a cell record. Never raises for a
    failed run -- a failure is a cell with outcome != "ok", carrying its
    diagnostics, so the grid can report it rather than abort."""
    runs = Path(runs)
    before = set(runs.glob(f"{method}-*")) if runs.is_dir() else set()
    cmd = stage_command(method, stage, runs, platform, data_root, produced,
                        python)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    run_dir = _new_run_dir(runs, method, before)
    outcome = "failed"
    if proc.returncode == 0 and run_dir is not None:
        try:
            record = json.loads(
                (run_dir / "launch.json").read_text(encoding="utf-8"))
            outcome = record.get("outcome", "failed")
        except (OSError, ValueError):
            outcome = "failed"
    cell = {
        "method": method,
        "stage": stage["name"],
        "run_dir": run_dir,
        "returncode": proc.returncode,
        "outcome": outcome,
        "produces": list(stage.get("produces", [])),
        "produces_metric": stage.get("produces_metric"),
    }
    if outcome != "ok":
        cell["error"] = (proc.stdout[-2000:] + proc.stderr[-2000:]).strip()
        if run_dir is not None:
            cell["job_log"] = str((run_dir / "job.log").resolve())
    return cell


def run_method(method: str, spec: dict, data_roots: dict, runs, platform: str,
               python=None) -> list[dict]:
    """Run one method's stages in order, threading every file a stage produces
    to a later stage that asks for it (`@encoder` for encoder.pt, `@produces:
    <file>` for any other, e.g. clusters.npy)."""
    shape = spec.get("data_shape")
    if shape not in data_roots:
        return [{
            "method": method, "stage": None, "run_dir": None,
            "returncode": None, "outcome": "failed", "produces": [],
            "produces_metric": None,
            "error": f"no --data mapping for data_shape {shape!r}; "
                     f"have {sorted(data_roots)}",
        }]
    data_root = data_roots[shape]
    cells = []
    produced = {}
    for stage in spec.get("stages", []):
        try:
            cell = run_stage(method, stage, runs, platform, data_root,
                             produced, python)
        except MatrixError as exc:
            cell = {"method": method, "stage": stage.get("name"),
                    "run_dir": None, "returncode": None, "outcome": "failed",
                    "produces": [], "produces_metric": None, "error": str(exc)}
        cells.append(cell)
        if cell["outcome"] != "ok":
            break  # a later stage that needs this one's artifacts cannot run
        for fname in stage.get("produces", []):
            if cell["run_dir"] is not None:
                produced[fname] = cell["run_dir"] / "out" / fname
    return cells


def run_grid(specs: list, data_roots: dict, runs, platform: str,
             python=None) -> dict:
    """The whole grid, and its one verdict: ok only when every cell is ok."""
    cells = []
    for method, spec in specs:
        cells.extend(run_method(method, spec, data_roots, runs, platform,
                                 python))
    ok = bool(cells) and all(c["outcome"] == "ok" for c in cells)
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "status": "ok" if ok else "failed",
        "cells": cells,
    }


def parse_data(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise MatrixError(f"--data {item!r} is not SHAPE=PATH")
        key, _, val = item.partition("=")
        out[key] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", action="append", default=[], metavar="SHAPE=PATH",
                    help="where a data_shape's data lives; repeat per shape")
    ap.add_argument("--method", action="append", default=[], metavar="NAME",
                    help="restrict the grid to these methods (default: all "
                         "that declare a spec)")
    ap.add_argument("--platform", default=DEFAULT_PLATFORM)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, metavar="matrix.json",
                    help="write the grid's machine verdict here")
    ap.add_argument("--python", default=None,
                    help="interpreter for each job's command (passed on to "
                         "launch.py); point it at a method venv for a cluster")
    a = ap.parse_args()

    try:
        specs = discover_specs()
        if a.method:
            wanted = set(a.method)
            specs = [(m, s) for m, s in specs if m in wanted]
            missing = wanted - {m for m, _ in specs}
            if missing:
                raise MatrixError(
                    f"no {SPEC_NAME} for: {', '.join(sorted(missing))}")
        if not specs:
            raise MatrixError(
                f"no method declares {SPEC_NAME}; the grid would be empty")
        data_roots = parse_data(a.data)
    except MatrixError as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2

    report = run_grid(specs, data_roots, a.runs_dir, a.platform, a.python)
    if a.out:
        a.out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                    default=str) + "\n", encoding="utf-8")
    for cell in report["cells"]:
        where = cell["run_dir"] or "(no run directory)"
        print(f"  {cell['outcome']:<8} {cell['method']}/{cell['stage']}  "
              f"{where}")
        if cell["outcome"] != "ok" and cell.get("error"):
            tail = cell["error"].splitlines()
            if tail:
                print(f"           {tail[-1][:160]}")
    done = sum(c["outcome"] == "ok" for c in report["cells"])
    print(f"  grid: {report['status']} ({done}/{len(report['cells'])} cells ok)")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
