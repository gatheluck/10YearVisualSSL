#!/usr/bin/env python3
"""Audit a produced grid: did every declared weight and evaluation land?

    matrix-audit.py --matrix matrix.json [--out audit.json]

`matrix-run.py` produces the grid and self-reports a verdict in matrix.json.
This is the independent judge of what it produced. It answers "did everything
land where it should?" from two sources only -- the outputs on disk and each
method's own `real_run_smoke.json` -- and never from the matrix's own claim of
success. A driver that records a cell as `ok` while its `encoder.pt` is missing
does not get past this: the audit reads the file, not the boast.

What it checks, per cell the matrix names, deriving the expectations from the
method's spec (not from the cell):

- the run directory exists, with `out/run_manifest.json` reporting `status: ok`;
- every file the spec says the stage `produces` is present under `out/`;
- the `produces_metric`, if any, is present in `out/metrics.json`;
- `launch.json` records `outcome: ok` and `contract_ok: true`.

And across a method's cells: the stages present must be an in-order prefix of
what the spec declares, and a method whose last cell is `ok` must have run every
stage -- so a silently dropped final stage cannot read as a complete success.

The audit fails, loudly and with the reason, if any check fails, if a cell is
not `ok`, if the matrix's own status disagrees with what the disk shows, or if
the matrix names no cells. Nothing is skipped in silence: every problem is in
the report and on the exit status. No platform is named here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1


class AuditError(Exception):
    """A refusal, always naming what was refused."""


def _driver():
    """The one implementation of spec discovery lives in matrix-run.py; load it
    rather than keep a second copy of "which methods declare a spec"."""
    spec = importlib.util.spec_from_file_location(
        "matrix_run_for_audit", ROOT / "bin" / "matrix-run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_cell(cell: dict, stage_spec: dict) -> list[str]:
    """Every problem with one ok cell, checked against the spec on disk. An
    empty list means it landed."""
    tag = f"{cell.get('method')}/{cell.get('stage')}"
    problems = []
    run_dir = cell.get("run_dir")
    if not run_dir:
        return [f"{tag}: the matrix records no run directory"]
    run = Path(run_dir)
    if not run.is_dir():
        return [f"{tag}: run directory {run} does not exist"]

    out = run / "out"
    manifest = out / "run_manifest.json"
    if not manifest.is_file():
        problems.append(f"{tag}: no out/run_manifest.json")
    else:
        try:
            if _load_json(manifest).get("status") != "ok":
                problems.append(f"{tag}: run_manifest status is not ok")
        except ValueError:
            problems.append(f"{tag}: out/run_manifest.json is not valid JSON")

    for fname in stage_spec.get("produces", []):
        if not (out / fname).is_file():
            problems.append(f"{tag}: missing produced file {fname}")

    metric = stage_spec.get("produces_metric")
    if metric:
        metrics = out / "metrics.json"
        if not metrics.is_file():
            problems.append(f"{tag}: no out/metrics.json for metric {metric}")
        else:
            try:
                if metric not in _load_json(metrics).get("metrics", {}):
                    problems.append(f"{tag}: metric {metric} not in metrics.json")
            except ValueError:
                problems.append(f"{tag}: out/metrics.json is not valid JSON")

    record = run / "launch.json"
    if not record.is_file():
        problems.append(f"{tag}: no launch.json")
    else:
        try:
            rec = _load_json(record)
            if rec.get("outcome") != "ok":
                problems.append(f"{tag}: launch.json outcome is not ok")
            if not rec.get("contract_ok"):
                problems.append(f"{tag}: launch.json contract_ok is not true")
        except ValueError:
            problems.append(f"{tag}: launch.json is not valid JSON")
    return problems


def audit(matrix: dict, specs) -> dict:
    """The whole audit: per-cell checks against the specs, plus completeness of
    each method's stage chain. Returns a report; status ok only when nothing is
    wrong and the matrix's own status agrees."""
    by_name = {method: spec for method, spec in specs}
    cells = matrix.get("cells", [])
    top_problems = []
    if not cells:
        top_problems.append("the matrix names no cells; nothing was produced")

    cell_reports = []
    by_method: dict[str, list] = {}
    for cell in cells:
        by_method.setdefault(cell.get("method"), []).append(cell)

    for method, mcells in by_method.items():
        spec = by_name.get(method)
        if spec is None:
            top_problems.append(
                f"{method}: named in the matrix but declares no spec")
            continue
        declared = [s["name"] for s in spec.get("stages", [])]
        present = [c.get("stage") for c in mcells]
        if present != declared[:len(present)]:
            top_problems.append(
                f"{method}: stages {present} are not an in-order prefix of "
                f"the declared {declared}")
        if mcells and mcells[-1].get("outcome") == "ok" and present != declared:
            top_problems.append(
                f"{method}: last stage is ok but not all declared stages ran "
                f"(have {present}, declared {declared})")

    stage_by = {(m, s["name"]): s for m, spec in specs
                for s in spec.get("stages", [])}
    ok_cells = 0
    for cell in cells:
        key = (cell.get("method"), cell.get("stage"))
        stage_spec = stage_by.get(key, {})
        if cell.get("outcome") != "ok":
            cell_reports.append({"cell": f"{key[0]}/{key[1]}", "ok": False,
                                 "problems": [f"cell outcome is "
                                              f"{cell.get('outcome')!r}, not ok"]})
            continue
        problems = audit_cell(cell, stage_spec)
        cell_reports.append({"cell": f"{key[0]}/{key[1]}",
                             "ok": not problems, "problems": problems})
        if not problems:
            ok_cells += 1

    all_cells_ok = all(cr["ok"] for cr in cell_reports)
    derived_ok = bool(cells) and not top_problems and all_cells_ok
    claimed = matrix.get("status")
    if claimed != ("ok" if derived_ok else "failed"):
        top_problems.append(
            f"the matrix claims status {claimed!r} but the audit finds "
            f"{'ok' if derived_ok else 'failed'}")

    status_ok = derived_ok and not top_problems
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if status_ok else "failed",
        "checked": len(cells),
        "ok_cells": ok_cells,
        "problems": top_problems,
        "cells": cell_reports,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", required=True, type=Path,
                    help="the matrix.json a grid run produced")
    ap.add_argument("--out", type=Path, metavar="audit.json",
                    help="write the audit report here")
    a = ap.parse_args()

    try:
        matrix = _load_json(a.matrix)
        specs = _driver().discover_specs()
    except (OSError, ValueError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2

    report = audit(matrix, specs)
    if a.out:
        a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    for cr in report["cells"]:
        if not cr["ok"]:
            for problem in cr["problems"]:
                print(f"  {problem}")
    for problem in report["problems"]:
        print(f"  {problem}")
    print(f"  audit: {report['status']} "
          f"({report['ok_cells']}/{report['checked']} cells landed)")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
