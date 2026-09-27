"""Fresh-process checks for component-only and complete task environments."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class TestDependencyCI(unittest.TestCase):
    def test_complete_environment_regression_runs_in_downstream_job(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not (ROOT / '.github/workflows/tests.yml').is_file() or not HAVE_YAML:
            self.skipTest('checkout and YAML parser required')
        command = next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                       if s.get('name') == 'Run Basic5 component contracts with downstream dependencies')
        self.assertTrue(_runs_finetune_tests(command, module='tests.test_basic5_component_dependencies'))


@unittest.skipUnless(HAVE_TORCH, "component execution requires torch")
class TestComponentEnvironments(unittest.TestCase):
    def run_environment(self, missing):
        code = r'''
import importlib.abc, importlib.util, json, sys, unittest
missing = set(json.loads(sys.argv[1]))
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in missing:
            raise ModuleNotFoundError('deliberately absent: ' + fullname, name=fullname)
sys.meta_path.insert(0, Block())
real_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: (
    None if name.split('.')[0] in missing else real_find_spec(name, *a, **kw))
for name in missing:
    assert importlib.util.find_spec(name) is None
    try:
        __import__(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('dependency blocker did not work: ' + name)
from tests.test_method_vggt_omega import TestOmega
suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestOmega)
ids = [t.id().rsplit('.', 1)[1] for t in suite]
result = unittest.TestResult()
suite.run(result)
print('COMPONENT_RESULT=' + json.dumps(dict(
    ids=ids, run=result.testsRun,
    skipped=[t.id().rsplit('.', 1)[1] for t, _ in result.skipped],
    errors=[text for _, text in result.errors],
    failures=[text for _, text in result.failures])))
'''
        proc = subprocess.run([sys.executable, "-c", code, json.dumps(missing)],
                              cwd=ROOT, capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        reports = [json.loads(line.removeprefix("COMPONENT_RESULT="))
                   for line in proc.stdout.splitlines() if line.startswith("COMPONENT_RESULT=")]
        self.assertEqual(len(reports), 1, proc.stdout + proc.stderr)
        report = reports[0]
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["run"], len(report["ids"]))
        return report

    def test_missing_task_extras_keep_encoder_and_detector_contracts_active(self):
        report = self.run_environment(["pycocotools", "scipy", "h5py", "av"])
        self.assertEqual(set(report["skipped"]), {
            "test_native_detector_and_category_roundtrip_preserve_sparse_coco_ids",
            "test_all_fourteen_task_routes_use_the_native_video_and_category_profiles"})
        self.assertIn("test_native_detector_preserves_spatial_features_and_head", report["ids"])
        self.assertGreaterEqual(report["run"] - len(report["skipped"]), 4)

    def test_complete_environment_executes_every_component_contract(self):
        try:
            import pycocotools, scipy, h5py, av, timm, einops
        except ImportError:
            self.skipTest("complete downstream environment required")
        report = self.run_environment([])
        self.assertEqual(report["skipped"], [])
        self.assertGreaterEqual(report["run"], 6)

    def test_one_missing_extra_is_not_hidden_by_other_installed_extras(self):
        try:
            import scipy, h5py, av, timm, einops
        except ImportError:
            self.skipTest("remaining downstream extras required")
        report = self.run_environment(["pycocotools"])
        self.assertEqual(set(report["skipped"]), {
            "test_native_detector_and_category_roundtrip_preserve_sparse_coco_ids",
            "test_all_fourteen_task_routes_use_the_native_video_and_category_profiles"})


if __name__ == "__main__":
    unittest.main()
