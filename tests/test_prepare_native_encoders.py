"""Batch native exports validate all inputs before invoking isolated workers."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bin/prepare-native-encoders.py"

class TestBatch(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("batch_native", TOOL)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.methods = self.root / "methods"
        self.sources = {}
        for name in ("example_a", "example_b"):
            d = self.methods / name
            (d / "configs").mkdir(parents=True)
            (d / "configs/linear_eval.yaml").write_text("train: {}\n")
            (d / "provenance.json").write_text(json.dumps({"step1_native_artifact": {
                "sha256": "a" * 64, "state_key": "state_dict", "strip_prefix": "module."}}))
            src = self.root / (name + ".pth")
            src.write_bytes(b"source")
            self.sources[name] = str(src)
        self.out = self.root / "out"

    def test_commands_keep_per_method_identity_and_explicit_mapping(self):
        jobs = self.mod.plan(self.sources, self.methods, self.out, "python-test")
        self.assertEqual(len(jobs), 2)
        for name, command in jobs:
            self.assertEqual(command[0], "python-test")
            self.assertEqual(command[command.index("--source") + 1], self.sources[name])
            self.assertEqual(command[command.index("--state-key") + 1], "state_dict")
            self.assertEqual(command[command.index("--sha256") + 1], "a" * 64)
            self.assertEqual(command[command.index("--out") + 1], str(self.out / name))

    def test_bad_later_entry_prevents_every_worker(self):
        self.sources["example_b"] = str(self.root / "missing")
        with patch.object(self.mod.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                self.mod.execute(self.sources, self.methods, self.out, "python-test")
            run.assert_not_called()
            self.assertFalse(self.out.exists())

    def test_existing_output_is_never_reused(self):
        self.out.mkdir()
        (self.out / "sentinel").write_text("keep")
        with self.assertRaises(ValueError):
            self.mod.plan(self.sources, self.methods, self.out, "python-test")
        self.assertEqual((self.out / "sentinel").read_text(), "keep")

    def test_path_escape_and_invalid_hash_are_rejected(self):
        with self.assertRaises(ValueError):
            self.mod.plan({"../example": self.sources["example_a"]}, self.methods, self.out, "python")
        d = self.methods / "example_a/provenance.json"
        r = json.loads(d.read_text())
        r["step1_native_artifact"]["sha256"] = "not-a-hash"
        d.write_text(json.dumps(r))
        with self.assertRaises(ValueError):
            self.mod.plan(self.sources, self.methods, self.out, "python")

    def test_failed_worker_reported_and_other_worker_still_runs(self):
        calls = []
        def worker(command, **kwargs):
            calls.append(command)
            class Result: returncode = 7 if len(calls) == 1 else 0
            return Result()
        with patch.object(self.mod.subprocess, "run", side_effect=worker):
            result = self.mod.execute(self.sources, self.methods, self.out, "python")
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([r["returncode"] for r in result["methods"]], [7, 0])
        self.assertEqual(json.loads((self.out / "batch.json").read_text()), result)
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            self.mod.plan({}, self.methods, self.out, "python")

    def test_launch_error_is_failure_and_does_not_stop_later_models(self):
        with patch.object(self.mod.subprocess, "run", side_effect=OSError("cannot launch")):
            result = self.mod.execute(self.sources, self.methods, self.out, "python")
        self.assertEqual(result["status"], "failed")
        self.assertEqual([r["returncode"] for r in result["methods"]], [127, 127])

    def test_interrupted_batch_is_not_reported_as_complete(self):
        calls = []
        def worker(command, **kwargs):
            calls.append(command)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            class Result: returncode = 0
            return Result()
        with patch.object(self.mod.subprocess, "run", side_effect=worker):
            with self.assertRaises(KeyboardInterrupt):
                self.mod.execute(self.sources, self.methods, self.out, "python")
        result = json.loads((self.out / "batch.json").read_text())
        self.assertEqual(result["status"], "incomplete")

    def test_symlinked_method_outside_root_is_rejected(self):
        external = self.root / "external"
        external.mkdir()
        (self.methods / "escape").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.mod.plan({"escape": self.sources["example_a"]}, self.methods, self.out, "python")

    def test_mapping_must_be_explicit_even_when_prefix_is_empty(self):
        d = self.methods / "example_a/provenance.json"
        value = json.loads(d.read_text())
        del value["step1_native_artifact"]["strip_prefix"]
        d.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "explicit strip_prefix"):
            self.mod.plan(self.sources, self.methods, self.out, "python")

    def test_cli_refuses_bad_input_without_creating_output(self):
        import subprocess
        import sys
        sources = self.root / "sources.json"
        sources.write_text("{}")
        result = subprocess.run([sys.executable, str(TOOL), "--sources", str(sources),
                                 "--out", str(self.out)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonempty", result.stderr)
        self.assertFalse(self.out.exists())

    def test_optional_module_map_reaches_export_worker(self):
        p = self.methods / 'example_a/provenance.json'
        record = json.loads(p.read_text())
        record['step1_native_artifact']['module_map'] = {'conv': 'encoder.0'}
        p.write_text(json.dumps(record))
        jobs = dict(self.mod.plan(self.sources, self.methods, self.out, 'python'))
        command = jobs['example_a']
        self.assertEqual(json.loads(command[command.index('--module-map') + 1]),
                         {'conv': 'encoder.0'})
