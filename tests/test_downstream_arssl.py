#!/usr/bin/env python3
"""The ARSSL downstream harness (docs/STEP3_PORTING_PLAN.md, item A1).

A1 ports the *shape* of the capture's ARSSL evaluation harness
(`origin/snapshots:methods_step3/ARSSL/src/run_eval.py`): drive **one frozen
backbone** through a battery of downstream task probes and aggregate the per-task
numbers into a single result. The capture loads the backbone once and loops over
task keys (`in100 in1k coco ade20k nyuv2 ssv2`), running a per-task probe and
writing one combined JSON.

This port is a **thin driver only** (the plan's chosen scope): it reuses the
downstream task runners already in this repo (`downstream/{ade20k,coco,nyuv2,
ssv2}.py`) rather than re-implementing any task head -- one implementation,
invoked, never copied (CLAUDE.md; matches `bin/matrix-run.py` reusing
`launch.py`). It drives each task as a subprocess, verifies each with the
downstream contract, and aggregates into one ARSSL result whose verdict is `ok`
only when every selected task is `ok`. The ImageNet columns the capture also
carries are *not* re-homed here: ImageNet-1k stays the per-method `linear_eval`
by design (docs/DOWNSTREAM.md), and ImageNet-100 is not yet ported -- those are
wired in later plan items, not A1.

Because the driver is subprocess-based it is **pure standard library** (like
`bin/matrix-run.py`): torch/timm live only in the task-runner subprocesses. So
its discovery, config composition and aggregation are tested here in the base
env; only the end-to-end smoke needs torch + timm (the real task runner).

The task runners are **discovered, not listed** (CLAUDE.md: "Discover, never
list."): a `downstream/*.py` module is a task runner iff it declares, at module
level, a string `TASK` constant and both a `run` and a `main` function. This is
matched structurally by parsing the AST, never by a substring over the file, and
the discovery test carries a positive control (a module with the triple is
found) and a negative control (a module missing it is not).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The harness is pure stdlib, so it imports in the base env (no torch needed).
from downstream import arssl                                        # noqa: E402
from downstream import contract                                     # noqa: E402

# The end-to-end smoke drives the real ADE20K runner (the lightest task: no
# pycocotools/h5py/av), so it needs torch + timm exactly as the ADE20K smoke
# does. Its tiny-data fixture is reused, not re-implemented (one fixture).
try:
    import timm                                             # noqa: F401
    import torch                                            # noqa: F401
    from tests.test_downstream_ade20k import tiny_ade, SMOKE_PROBE
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(HAVE_TIMM, "ARSSL end-to-end smoke needs timm + torch")


class TestTaskRunnersAreDiscoveredNotListed(unittest.TestCase):
    """The driver finds task runners by their structure, with both controls."""

    def _write(self, pkg: Path, name: str, body: str) -> None:
        (pkg / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")

    def test_the_triple_is_the_detector_positive_and_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            # Positive control: a module with TASK + run + main is a runner.
            self._write(pkg, "good_task", '''
                TASK = "good_task_metric"
                def run(cfg, out, device_override=None):
                    return {}
                def main(argv=None):
                    return 0
            ''')
            # Negative controls, each missing exactly one leg of the triple, so a
            # detector that dropped any leg would wrongly include one of them.
            self._write(pkg, "no_task", '''
                def run(cfg, out, device_override=None):
                    return {}
                def main(argv=None):
                    return 0
            ''')
            self._write(pkg, "no_run", '''
                TASK = "no_run_metric"
                def main(argv=None):
                    return 0
            ''')
            self._write(pkg, "no_main", '''
                TASK = "no_main_metric"
                def run(cfg, out, device_override=None):
                    return {}
            ''')
            found = arssl.discover_tasks(pkg)
            self.assertEqual(set(found), {"good_task_metric"},
                             "only the module with the full TASK+run+main triple "
                             "is a task runner")
            self.assertEqual(found["good_task_metric"], "good_task")

    def test_a_substring_of_task_does_not_fool_the_detector(self):
        # A decoy a substring match would wrongly catch: the token TASK appears
        # only inside a string/other name, never as a module-level assignment.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            self._write(pkg, "decoy", '''
                NOT_TASK = "TASK = 'looks-like-one'"
                def run(cfg, out, device_override=None):
                    return {}
                def main(argv=None):
                    return 0
            ''')
            self.assertEqual(arssl.discover_tasks(pkg), {},
                             "a string containing TASK must not be read as the "
                             "module-level TASK constant")

    def test_the_real_downstream_runners_are_all_found(self):
        found = arssl.discover_tasks()
        # The four ported task runners each declare the triple; the two helper
        # modules (contract, spatial_backbones) and the driver itself do not.
        self.assertGreaterEqual(len(found), 4)
        self.assertNotIn(arssl.TASK, found,
                         "the driver must not discover itself as a task")
        for key, module in found.items():
            self.assertIsInstance(key, str)
            self.assertTrue(key)
            self.assertNotIn(module, ("contract", "spatial_backbones", "arssl"))


SHARED_BACKBONE = {"kind": "vit", "encoder": "", "arch": "vit_base_patch16_224",
                   "img_size": 32, "patch_size": 16, "embed_dim": 64, "depth": 2,
                   "num_heads": 2}


def arssl_config(**over) -> dict:
    """A minimal, valid ARSSL config over one discovered task (ADE20K), so the
    validation/composition tests use a real task key without naming a method."""
    key = next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")
    cfg = {"task": "arssl", "seed": 0, "device": "cpu",
           "backbone": dict(SHARED_BACKBONE),
           "tasks": {key: {"data_root": "/tmp/nowhere",
                           "probe": {"epochs": 1}}}}
    for k, v in over.items():
        cfg[k] = v
    return cfg


class TestConfigValidation(unittest.TestCase):
    def test_a_valid_config_is_accepted(self):
        arssl.validate_config(arssl_config())

    def test_an_unknown_top_key_is_refused_by_name(self):
        cfg = arssl_config()
        cfg["mystery"] = 1
        with self.assertRaises(arssl.ConfigError) as e:
            arssl.validate_config(cfg)
        self.assertIn("mystery", str(e.exception))

    def test_setting_the_output_is_refused(self):
        cfg = arssl_config()
        cfg["out"] = "/anywhere"
        with self.assertRaises(arssl.ConfigError) as e:
            arssl.validate_config(cfg)
        self.assertIn("--out", str(e.exception))

    def test_an_empty_task_battery_is_refused(self):
        with self.assertRaises(arssl.ConfigError) as e:
            arssl.validate_config(arssl_config(tasks={}))
        self.assertIn("empty", str(e.exception).lower())

    def test_an_undiscovered_task_is_refused_not_skipped(self):
        # Negative control: an unknown task is a hard, named refusal -- never a
        # silent skip that would read as "the battery ran".
        cfg = arssl_config(tasks={"not_a_real_task": {"data_root": "/x"}})
        with self.assertRaises(arssl.ConfigError) as e:
            arssl.validate_config(cfg)
        self.assertIn("not_a_real_task", str(e.exception))

    def test_a_per_task_config_may_not_shadow_a_shared_key(self):
        # The battery is ONE frozen backbone at one seed/device: a task cannot
        # carry its own, which would silently diverge from the others.
        key = next(iter(arssl_config()["tasks"]))
        cfg = arssl_config()
        cfg["tasks"][key]["backbone"] = {"kind": "vit"}
        with self.assertRaises(arssl.ConfigError) as e:
            arssl.validate_config(cfg)
        self.assertIn("backbone", str(e.exception))


class TestPerTaskConfigComposition(unittest.TestCase):
    def test_compose_overlays_the_shared_backbone_seed_and_device(self):
        cfg = arssl_config()
        key = next(iter(cfg["tasks"]))
        composed = arssl.compose_task_config(cfg, key)
        self.assertEqual(composed["task"], key)
        self.assertEqual(composed["seed"], cfg["seed"])
        self.assertEqual(composed["device"], cfg["device"])
        self.assertEqual(composed["backbone"], cfg["backbone"])
        self.assertEqual(composed["data_root"], "/tmp/nowhere")
        self.assertEqual(composed["probe"], {"epochs": 1})

    def test_compose_forwards_task_own_keys_agnostic_to_probe_or_detector(self):
        # A detection-shaped task carries `detector`, not `probe`; the driver
        # forwards whatever the task declares -- it does not know either name.
        cfg = arssl_config()
        key = next(iter(cfg["tasks"]))
        cfg["tasks"][key] = {"data_root": "/d", "detector": {"anchors": [1]}}
        composed = arssl.compose_task_config(cfg, key)
        self.assertEqual(composed["detector"], {"anchors": [1]})
        self.assertNotIn("probe", composed)


class TestTheAggregateVerdict(unittest.TestCase):
    def test_every_task_ok_is_ok_and_unions_the_comparable_metrics(self):
        cells = [
            {"task": "ade20k_segmentation", "status": "ok",
             "metrics": {"ade20k_miou": 1.0}},
            {"task": "nyuv2_depth", "status": "ok",
             "metrics": {"nyuv2_rmse": 2.0}},
        ]
        agg = arssl.aggregate(cells)
        self.assertEqual(agg["status"], "ok")
        self.assertEqual(agg["metrics"], {"ade20k_miou": 1.0, "nyuv2_rmse": 2.0})
        self.assertEqual(set(agg["tasks"]),
                         {"ade20k_segmentation", "nyuv2_depth"})

    def test_one_failed_task_fails_the_whole_battery(self):
        cells = [
            {"task": "ade20k_segmentation", "status": "ok",
             "metrics": {"ade20k_miou": 1.0}},
            {"task": "nyuv2_depth", "status": "failed", "metrics": None,
             "error": "boom"},
        ]
        agg = arssl.aggregate(cells)
        self.assertEqual(agg["status"], "failed")

    def test_an_empty_battery_is_not_ok(self):
        # Mirrors matrix-run: `ok` requires at least one cell, all ok. "Nothing
        # ran" must never read as success.
        self.assertEqual(arssl.aggregate([])["status"], "failed")

    def test_two_tasks_claiming_one_metric_is_refused(self):
        cells = [
            {"task": "a", "status": "ok", "metrics": {"ade20k_miou": 1.0}},
            {"task": "b", "status": "ok", "metrics": {"ade20k_miou": 2.0}},
        ]
        with self.assertRaises(ValueError) as e:
            arssl.aggregate(cells)
        self.assertIn("ade20k_miou", str(e.exception))


class TestTheEndToEndSmoke(unittest.TestCase):
    """Drive one real task runner through the whole chain: compose -> subprocess
    -> contract.verify -> aggregate -> write."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ds-arssl-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def _ade_key(self) -> str:
        return next(k for k, m in arssl.discover_tasks().items() if m == "ade20k")

    def _run(self, cfg: dict):
        cfg_path = self.tmp / "arssl.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "downstream.arssl", "--config", str(cfg_path),
             "--out", str(self.out)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        return cfg_path, r

    def _cfg(self, data_root) -> dict:
        return {"task": "arssl", "seed": 0, "device": "cpu",
                "backbone": dict(SHARED_BACKBONE),
                "tasks": {self._ade_key(): {"data_root": str(data_root),
                                            "probe": dict(SMOKE_PROBE)}}}

    @needs_timm
    def test_it_drives_a_real_task_and_satisfies_the_downstream_contract(self):
        data = tiny_ade(self.tmp / "data")
        cfg_path, r = self._run(self._cfg(data))
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        # The ARSSL run itself is contract-checkable (its manifest lists every
        # nested task artifact).
        ok, violations = contract.verify(self.out, cfg_path, r.returncode)
        self.assertTrue(ok, violations)
        results = json.loads((self.out / "arssl_results.json").read_text())
        self.assertEqual(results["status"], "ok")
        self.assertIn("ade20k_miou", results["metrics"])
        # The per-task run landed in its own subdir with its own ok manifest.
        sub = self.out / "ade20k"
        self.assertTrue((sub / "run_manifest.json").is_file())
        sub_man = json.loads((sub / "run_manifest.json").read_text())
        self.assertEqual(sub_man["status"], "ok")

    @needs_timm
    def test_a_task_failure_fails_the_battery_and_is_not_silent(self):
        # A bad data_root makes the ADE20K runner fail; the battery must go
        # failed, exit nonzero, and carry the reason -- never a silent skip.
        cfg_path, r = self._run(self._cfg(self.tmp / "does-not-exist"))
        self.assertEqual(r.returncode, 1, r.stdout[-2000:] + r.stderr[-2000:])
        results = json.loads((self.out / "arssl_results.json").read_text())
        self.assertEqual(results["status"], "failed")
        cell = results["tasks"][self._ade_key()]
        self.assertEqual(cell["status"], "failed")
        self.assertTrue(cell.get("error"), "a failed task must carry its reason")

    def test_a_refused_top_config_exits_two_and_writes_no_manifest(self):
        # Misuse (an unknown device) is refused before anything runs: exit 2, no
        # manifest -- distinct from a run that failed.
        cfg = self._cfg("/tmp/nowhere") if HAVE_TIMM else {
            "task": "arssl", "seed": 0, "device": "cpu",
            "backbone": {}, "tasks": {"nope": {}}}
        cfg["device"] = "tpu"
        _, r = self._run(cfg)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertFalse((self.out / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
