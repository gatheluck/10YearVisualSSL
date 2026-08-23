#!/usr/bin/env python3
"""Cross-method output-layout contract: bin/matrix-audit.py judges, from the
outputs on disk and the specs alone, that every declared weight and evaluation
landed where it should -- docs/REAL_RUN_VERIFICATION.md step 4.

matrix-run.py *produces* the grid and self-reports a verdict; this AUDITS the
produced tree independently. It re-derives what each cell must have produced
from the method's own `real_run_smoke.json` (not from the matrix's self-report),
then checks the files on disk. So a driver that claims a cell is "ok" while its
`encoder.pt` is missing does not survive: the audit reads the file, not the
claim.

Most of this is hermetic and needs no training: a fabricated runs tree plus a
matrix.json exercise the auditor's logic in the base environment. Two controls,
both required -- a positive (a complete, well-formed tree passes) and a NEGATIVE
(delete one produced file and the audit must fail). mutations/matrix-audit.json
breaks the on-disk existence check, and the negative control is what kills it.

One integration test runs the real matrix-run -> matrix-audit chain under a
method venv, so the fabricated fixture cannot silently drift from the real
output layout; it skips where PyYAML or the spec's needs are absent.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import _real_run   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
HAVE_YAML = importlib.util.find_spec("yaml") is not None


def importable(mods) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def fabricate(specs, root: Path) -> Path:
    """A complete, well-formed runs tree and matrix.json for the given specs,
    every cell ok and every declared artifact present. Built by hand so the
    auditor's logic can be exercised without training; the integration test
    guards this against drift from the real layout."""
    runs = root / "runs"
    cells = []
    for method, spec in specs:
        for i, stage in enumerate(spec["stages"]):
            run_dir = runs / f"{method}-fake{i}"
            out = run_dir / "out"
            out.mkdir(parents=True, exist_ok=True)
            (out / "run_manifest.json").write_text(
                json.dumps({"status": "ok"}), encoding="utf-8")
            metric = stage.get("produces_metric")
            (out / "metrics.json").write_text(
                json.dumps({"metrics": {metric: 0.5} if metric else {}}),
                encoding="utf-8")
            for fname in stage.get("produces", []):
                (out / fname).write_bytes(b"")
            (run_dir / "launch.json").write_text(
                json.dumps({"outcome": "ok", "contract_ok": True}),
                encoding="utf-8")
            cells.append({
                "method": method, "stage": stage["name"],
                "run_dir": str(run_dir), "returncode": 0, "outcome": "ok",
                "produces": list(stage.get("produces", [])),
                "produces_metric": metric,
            })
    matrix = {"schema_version": 1, "platform": "local", "status": "ok",
              "cells": cells}
    matrix_path = root / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    return matrix_path


class MatrixAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="audit-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def specs_here(self):
        return _real_run.driver().discover_specs()

    def audit(self, matrix, out):
        return subprocess.run(
            [sys.executable, str(BIN / "matrix-audit.py"),
             "--matrix", str(matrix), "--out", str(out)],
            capture_output=True, text=True)

    def a_produced_file(self, specs):
        """The first (method, stage, filename) a spec declares it produces."""
        for method, spec in specs:
            for stage in spec["stages"]:
                for fname in stage.get("produces", []):
                    return method, stage["name"], fname
        return None

    # -- the claims --------------------------------------------------------

    def test_at_least_one_method_declares_a_spec(self):
        """With no spec there is nothing to audit and this would test nothing."""
        self.assertGreater(len(self.specs_here()), 0)

    def test_a_complete_tree_passes(self):
        specs = self.specs_here()
        self.assertGreater(len(specs), 0)
        matrix = fabricate(specs, self.tmp)
        out = self.tmp / "audit.json"
        r = self.audit(matrix, out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report["status"], "ok", report)

    def test_a_missing_produced_file_fails_the_audit(self):
        specs = self.specs_here()
        target = self.a_produced_file(specs)
        if target is None:
            self.skipTest("no spec declares a produced file")
        method, stage_name, fname = target
        matrix = fabricate(specs, self.tmp)
        # The driver still claims ok, but the file the spec requires is gone.
        report_in = json.loads(matrix.read_text())
        cell = next(c for c in report_in["cells"]
                    if c["method"] == method and c["stage"] == stage_name)
        (Path(cell["run_dir"]) / "out" / fname).unlink()
        out = self.tmp / "audit.json"
        r = self.audit(matrix, out)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report["status"], "failed", report)

    def test_a_cell_claimed_ok_that_actually_failed_is_caught(self):
        """The audit must not trust the matrix's own verdict."""
        specs = self.specs_here()
        matrix = fabricate(specs, self.tmp)
        report_in = json.loads(matrix.read_text())
        # Corrupt one manifest to a failed status while the cell claims ok.
        cell = report_in["cells"][0]
        (Path(cell["run_dir"]) / "out" / "run_manifest.json").write_text(
            json.dumps({"status": "error"}), encoding="utf-8")
        out = self.tmp / "audit.json"
        r = self.audit(matrix, out)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(json.loads(out.read_text())["status"], "failed")

    def test_the_real_chain_audits_clean(self):
        """matrix-run -> matrix-audit on real output, so the fabricated fixture
        above cannot drift from the layout the tools actually produce."""
        specs = self.specs_here()
        self.assertGreater(len(specs), 0)
        if not HAVE_YAML:
            self.skipTest("launch's resolve step needs PyYAML")
        for method, spec in specs:
            if not importable(spec.get("needs", [])):
                self.skipTest(f"{method} needs {spec.get('needs')}")
        data_args = []
        for shape in sorted({s["data_shape"] for _, s in specs}):
            root = _real_run.build_data(shape, self.tmp / "data" / shape)
            data_args += ["--data", f"{shape}={root}"]
        matrix = self.tmp / "matrix.json"
        run = subprocess.run(
            [sys.executable, str(BIN / "matrix-run.py"),
             "--runs-dir", str(self.tmp / "runs"), "--out", str(matrix),
             *data_args], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        out = self.tmp / "audit.json"
        r = self.audit(matrix, out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(json.loads(out.read_text())["status"], "ok")


if __name__ == "__main__":
    unittest.main()
