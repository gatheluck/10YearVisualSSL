#!/usr/bin/env python3
"""Real-run smoke: a REAL method's stages land their artifacts through launch.py.

Every other end-to-end test (`tests/test_end_to_end.py`) drives
`methods/_reference`, which trains nothing; the per-method smokes
(`tests/test_method_*.py`) call adapter functions in-process. This drives a real
method's own `pretrain` -> `linear_eval` THROUGH launch.py (resolve -> local
backend -> `python -m adapter` -> contract-test -> record) and checks, by
machine, that every weight/eval artifact landed at its expected path --
docs/REAL_RUN_VERIFICATION.md step 2.

**Discovered, not listed.** A method declares its own short real-run in
`methods/<m>/real_run_smoke.json` (its own directory, so no shared file names a
method -- tests/test_no_hard_coded_methods.py). The discovery and the launch.py
driving live in `bin/matrix-run.py` and are imported here, so there is one
implementation of "which methods declare a spec" and "run one stage": this smoke
and the grid driver (tests/test_matrix_run.py, step 3) cannot drift apart. This
checks a single method's stages in detail; the grid test checks the verdict over
all of them.

It runs where launch's resolve step has PyYAML and the spec's `needs` import in
the active environment, and skips otherwise, so a skipped run is never mistaken
for a pass.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import _real_run   # noqa: E402

HAVE_YAML = importlib.util.find_spec("yaml") is not None
PLATFORM = _real_run.driver().DEFAULT_PLATFORM


def importable(mods) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


class RealRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="realrun-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def check_stage(self, cell: dict) -> None:
        """Every machine signal the contract chain leaves behind must agree."""
        self.assertEqual(cell["outcome"], "ok", self.diag(cell))
        run = Path(cell["run_dir"])
        out = run / "out"
        for fname in cell["produces"]:
            self.assertTrue((out / fname).is_file(),
                            f"{cell['method']}/{cell['stage']}: {fname} not "
                            f"under {out}")
        metric = cell["produces_metric"]
        if metric:
            metrics = json.loads((out / "metrics.json").read_text())
            self.assertIn(metric, metrics["metrics"],
                          f"{cell['method']}/{cell['stage']}: {metric}")
        manifest = json.loads((out / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "ok", manifest)
        record = json.loads((run / "launch.json").read_text())
        self.assertEqual(record["outcome"], "ok", record)
        self.assertTrue(record["contract_ok"], record)

    def diag(self, cell: dict) -> str:
        """Everything needed to see why a stage failed, including the job log
        the backend wrote -- no silent failure (CLAUDE.md, DESIGN 2.4)."""
        parts = [f"{cell['method']}/{cell['stage']} rc={cell.get('returncode')}",
                 cell.get("error", "")]
        log = cell.get("job_log")
        if log and Path(log).is_file():
            parts.append(f"--- {log} ---\n"
                         + Path(log).read_text(errors="replace"))
        return "\n".join(parts)

    # -- the claims --------------------------------------------------------

    def test_at_least_one_method_declares_a_real_run_smoke(self):
        """With no spec this file would test nothing and silently pass."""
        self.assertGreater(len(_real_run.driver().discover_specs()), 0)

    def test_each_declared_real_run_lands_its_artifacts(self):
        drv = _real_run.driver()
        specs = drv.discover_specs()
        self.assertGreater(len(specs), 0)
        for method, spec in specs:
            with self.subTest(method=method):
                if not HAVE_YAML:
                    self.skipTest("launch's resolve step needs PyYAML")
                if not importable(spec.get("needs", [])):
                    self.skipTest(f"{method} needs {spec.get('needs')}")
                runs = self.tmp / method / "runs"
                data = _real_run.build_data(
                    spec["data_shape"], self.tmp / method / "data")
                encoder = None
                for stage in spec["stages"]:
                    cell = drv.run_stage(method, stage, runs, PLATFORM, data,
                                         encoder)
                    self.check_stage(cell)
                    for fname in stage.get("produces", []):
                        if fname == "encoder.pt":
                            encoder = Path(cell["run_dir"]) / "out" / "encoder.pt"


if __name__ == "__main__":
    unittest.main()
