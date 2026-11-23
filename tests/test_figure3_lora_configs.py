"""Behavior of the local path binding for the full three-task LoRA campaign."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CampaignFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.inputs = self.root / 'inputs.json'
        self.out = self.root / 'configs'
        example = Path(__file__).resolve().parents[1] / 'configs' / 'figure3-lora-paths.example.json'
        self.models = list(json.loads(example.read_bytes())['encoders'])
        self.tasks = ['imagenet', 'ssv2', 'coco']
        self.paths = {
            'encoders': {m: str(self.root / 'local encoders' / m) for m in self.models},
            'datasets': {t: str(self.root / 'data' / t) for t in self.tasks},
        }

    def invoke(self):
        self.inputs.write_text(json.dumps(self.paths))
        return subprocess.run([sys.executable, '-m', 'downstream.figure3_lora_configs',
                               '--paths', str(self.inputs), '--out', str(self.out)],
                              capture_output=True, text=True, check=False)


class CampaignConfigsTests(CampaignFixture):
    def test_local_bindings_are_private_under_shared_default_umask(self):
        self.inputs.write_text(json.dumps(self.paths))
        subprocess.run([sys.executable, '-m', 'downstream.figure3_lora_configs',
                        '--paths', str(self.inputs), '--out', str(self.out)], umask=0o022, check=True)
        self.assertEqual(self.out.stat().st_mode & 0o077, 0)

    def test_preserves_existing_directory(self):
        self.out.mkdir()
        marker = self.out / 'existing'
        marker.write_text('preserve')
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertEqual(list(self.out.iterdir()), [marker])
        self.assertEqual(marker.read_text(), 'preserve')

    def test_rejects_missing_extra_and_relative_bindings_before_writing(self):
        for key, val in [('datasets', {'ssv2': '/only/one'}),
                         ('encoders', dict(self.paths['encoders'], unexpected='/other')),
                         ('encoders', dict(self.paths['encoders'], dinov3_hf='relative/path'))]:
            with self.subTest(key=key, val=val):
                previous = self.paths[key]
                self.paths[key] = val
                self.assertNotEqual(self.invoke().returncode, 0)
                self.assertFalse(self.out.exists())
                self.paths[key] = previous
