"""Exercise schedule test discovery in complete and partial method environments."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

from tests.test_downstream_coco import needs_coco, needs_deps


ROOT = Path(__file__).resolve().parents[1]
PURE = {
    "test_selection_and_invalid_profiles_fail_before_data_access",
    "test_all_update_boundaries_and_parameter_updates",
    "test_loop_advances_only_after_successful_optimizer_updates",
}
CLI = {
    "test_schedule_horizon_and_step_cap_reporting",
    "test_runner_milestones_use_full_loader_despite_step_cap",
    "test_real_cli_schedule_metadata_and_frozen_state",
    "test_cuda_cli",
}


@needs_deps
class TestScheduleDependencies(unittest.TestCase):
    def run_environment(self, missing):
        # A fresh interpreter prevents cached modules from hiding missing extras.
        code = r'''
import importlib.abc, json, sys, unittest
missing = set(json.loads(sys.argv[1]))
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in missing:
            raise ModuleNotFoundError("deliberately absent: " + fullname, name=fullname)
sys.meta_path.insert(0, Block())
for name in missing:
    try:
        __import__(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("dependency blocker did not work: " + name)
import torch
available = torch.cuda.is_available
# Check CUDA test selection even on CPU-only CI; restore before executing tests.
if missing:
    torch.cuda.is_available = lambda: True
from tests.test_coco_schedule import TestSchedule
torch.cuda.is_available = available
suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSchedule)
ids = [test.id().rsplit(".", 1)[1] for test in suite]
result = unittest.TestResult()
suite.run(result)
print("DEPENDENCY_RESULT=" + json.dumps({
    "ids": ids, "run": result.testsRun,
    "skipped": [test.id().rsplit(".", 1)[1] for test, _ in result.skipped],
    "errors": [text for _, text in result.errors],
    "failures": [text for _, text in result.failures],
    "cuda": torch.cuda.is_available(),
}))
'''
        proc = subprocess.run([sys.executable, "-c", code, json.dumps(missing)],
                              cwd=ROOT, text=True, capture_output=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        reports = [line.removeprefix("DEPENDENCY_RESULT=")
                   for line in proc.stdout.splitlines()
                   if line.startswith("DEPENDENCY_RESULT=")]
        self.assertEqual(len(reports), 1, proc.stdout + proc.stderr)
        report = json.loads(reports[0])
        self.assertEqual(set(report["ids"]), PURE | CLI)
        self.assertEqual(report["run"], len(PURE | CLI))
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["failures"], [])
        return report

    def test_partial_locks_keep_numerical_tests_active(self):
        for missing in (["timm"], ["pycocotools"], ["timm", "pycocotools"]):
            with self.subTest(missing=missing):
                report = self.run_environment(missing)
                self.assertEqual(set(report["skipped"]), CLI)

    @needs_coco
    def test_complete_environment_runs_every_cpu_contract(self):
        report = self.run_environment([])
        self.assertEqual(set(report["skipped"]),
                         set() if report["cuda"] else {"test_cuda_cli"})


if __name__ == "__main__":
    unittest.main()
