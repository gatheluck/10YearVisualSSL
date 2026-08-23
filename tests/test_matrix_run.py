#!/usr/bin/env python3
"""Real-run GRID: bin/matrix-run.py drives every declared method x stage and
decides, by one machine verdict, that every weight and evaluation landed.

tests/test_real_run_smoke.py proves one method's stages land through launch.py;
this proves the *grid* driver over them -- docs/REAL_RUN_VERIFICATION.md step 3.
It runs the real CLI (bin/matrix-run.py) as a subprocess over the discovered
specs, on the default backend, with tiny synthetic-but-real-shaped data, and
checks the emitted matrix.json.

Two controls, both required. The positive: good data -> every discovered cell
is present and ok, exit 0, status ok, artifacts on disk. The NEGATIVE: an empty
data root -> the first stage cannot load anything -> that cell fails -> the
whole grid fails, exit non-zero, status failed. The negative is what makes the
verdict falsifiable: a driver that always reports "ok" would pass the positive
alone. mutations/matrix-run.json breaks exactly that all-cells-agree rule, and
the negative control is what kills it.

It runs where launch's resolve step has PyYAML and every declared spec's `needs`
import in the active environment, and skips otherwise -- a skipped grid is never
mistaken for a pass.
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


class MatrixRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="matrix-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def cli(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(BIN / "matrix-run.py"), *map(str, args)],
            capture_output=True, text=True)

    def specs_here(self):
        return _real_run.driver().discover_specs()

    def shapes(self, specs):
        return sorted({spec["data_shape"] for _, spec in specs})

    def gate(self, specs) -> None:
        if not HAVE_YAML:
            self.skipTest("launch's resolve step needs PyYAML")
        for method, spec in specs:
            if not importable(spec.get("needs", [])):
                self.skipTest(f"{method} needs {spec.get('needs')}")

    # -- the claims --------------------------------------------------------

    def test_at_least_one_method_declares_a_spec(self):
        """With no spec the grid is empty and this file would test nothing."""
        self.assertGreater(len(self.specs_here()), 0)

    def test_the_grid_lands_every_cell(self):
        specs = self.specs_here()
        self.assertGreater(len(specs), 0)
        self.gate(specs)
        data_args = []
        for shape in self.shapes(specs):
            root = _real_run.build_data(shape, self.tmp / "data" / shape)
            data_args += ["--data", f"{shape}={root}"]
        out = self.tmp / "matrix.json"
        r = self.cli("--runs-dir", self.tmp / "runs", "--out", out, *data_args)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report["status"], "ok", report)
        # every discovered (method, stage) is a cell, and each one landed.
        want = {(m, st["name"]) for m, s in specs for st in s["stages"]}
        got = {(c["method"], c["stage"]) for c in report["cells"]}
        self.assertEqual(got, want, report)
        for cell in report["cells"]:
            self.assertEqual(cell["outcome"], "ok", cell)
            out_dir = Path(cell["run_dir"]) / "out"
            for fname in cell["produces"]:
                self.assertTrue((out_dir / fname).is_file(),
                                f"{cell['method']}/{cell['stage']}: {fname}")
            if cell["produces_metric"]:
                metrics = json.loads((out_dir / "metrics.json").read_text())
                self.assertIn(cell["produces_metric"], metrics["metrics"], cell)

    def test_a_failing_cell_fails_the_whole_grid(self):
        specs = self.specs_here()
        self.assertGreater(len(specs), 0)
        self.gate(specs)
        # A real data root per shape, but EMPTY: the class folders are absent,
        # so the first training stage cannot load anything and fails.
        data_args = []
        for shape in self.shapes(specs):
            empty = self.tmp / "empty" / shape
            empty.mkdir(parents=True, exist_ok=True)
            data_args += ["--data", f"{shape}={empty}"]
        out = self.tmp / "matrix.json"
        r = self.cli("--runs-dir", self.tmp / "runs", "--out", out, *data_args)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report["status"], "failed", report)
        self.assertTrue(any(c["outcome"] != "ok" for c in report["cells"]),
                        report)


if __name__ == "__main__":
    unittest.main()
