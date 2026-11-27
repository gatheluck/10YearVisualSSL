"""Recompute Figure 4 from explicit primary scores and checkpoint horizons."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TASKS = ("ImageNet-1K", "COCO", "ADE20K", "NYUv2", "SSv2")


def source():
    rows = []
    for index in range(32):
        metrics = []
        for task in TASKS:
            metrics.append({"task": task, "asis": "1.100", "ctrl_100": "0.5", "ctrl_200": "1.0",
                                "ctrl_300": "1.1001" if index < 2 else "1.0000",
                                "asis_rank": 999, "delta": "999"})
        rows.append({"method": f"Model {index}", "metrics": metrics})
    return {"cohort": 32, "records": rows, "summary": "stale"}


class ControlledComparison(unittest.TestCase):
    def invoke(self, data, directory):
        root = Path(directory)
        path = root / "source.json"
        path.write_text(json.dumps(data))
        return subprocess.run([sys.executable, "-m", "downstream.transfer_comparison",
                               "--figure4", "--input", str(path), "--out", str(root / "new.json")],
                              capture_output=True, text=True, check=False)

    def test_cli_recomputes_unrounded_changes_ranks_and_ties(self):
        data = source()
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(data, directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            out = Path(directory) / "new.json"
            before = out.read_bytes()
            report = json.loads(before)
            self.assertEqual(report['input_sha256'],
                             hashlib.sha256((Path(directory) / "source.json").read_bytes()).hexdigest())
            self.assertFalse(report['historical_run_verified'])
            settings = report['tasks']
            self.assertEqual(set(settings), set(TASKS))
            self.assertEqual(settings['COCO']['metric'], 'bbox_ap_percent')
            coco = settings['COCO']
            self.assertEqual(coco['counts'], {"improved": 2, "worse": 30, "equal": 0})
            self.assertEqual(coco['ctrl_winners'], ['Model 0', 'Model 1'])
            self.assertEqual(coco['methods']['Model 2']['ranks']['ctrl_300'], 3)
            self.assertEqual(coco['methods']['Model 0']['ranks']['asis'], 1)
            self.assertEqual(coco['methods']['Model 0']['scores']['ctrl_100'], '0.5')
            depth = settings['NYUv2']
            self.assertEqual(depth['counts'], {"improved": 30, "worse": 2, "equal": 0})
            self.assertEqual(depth['methods']['Model 1']['ranks']['ctrl_300'], 31)
            self.assertEqual(depth['methods']['Model 1']['improvement'], '-0.0001')
            self.assertNotEqual(self.invoke(data, directory).returncode, 0)
            self.assertEqual(before, out.read_bytes())

    def test_saved_rows_use_public_ids_and_alias_collisions_fail(self):
        from downstream.controlled_comparison import compare_controlled

        data = source()
        old = "38_clip"
        name = old.split("_", 1)[1]
        data["records"][0]["method"] = old
        report = compare_controlled(data)
        self.assertIn(name, report["tasks"]["COCO"]["methods"])
        self.assertEqual(data["records"][0]["method"], old)
        data["records"][1]["method"] = name
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare_controlled(data)

    def test_refuses_wrong_or_partial_panels_without_creating_report(self):
        for change in ('method', 'task', 'track', 'count', 'unit', 'nonfinite', 'classification', 'wrong_metric'):
            data = source()
            cells = data['records'][0]['metrics']
            if change == 'method':
                data['records'][1]['method'] = 'Model 0'
            if change == 'task':
                cells[1]['task'] = cells[0]['task']
            if change == 'track':
                del cells[1]['ctrl_200']
            if change == 'count':
                data['records'].pop()
            if change == 'unit':
                cells[1]['ctrl_100'] = '101'
            if change == 'nonfinite':
                cells[3]['ctrl_100'] = 'NaN'
            if change == 'wrong_metric':
                cells[1]['metric'] = 'category_presence_micro_f1'
            if change == 'classification':
                cells[1]['task'] = 'COCO category presence'
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                self.assertNotEqual(self.invoke(data, directory).returncode, 0)
                self.assertFalse((Path(directory) / "new.json").exists())
