"""Actual Step-1 checkpoint identity, measured from the referenced artifact."""
import json
from pathlib import Path
import unittest

class TestArtifact(unittest.TestCase):
    def test_recorded_native_identity(self):
        root = Path(__file__).resolve().parents[1]
        r = json.loads((root / "methods/03_colorization/provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(r["sha256"], "3c4789582471e3a363cf1d5ed3fc05d1a642f15bff381a802a50ae5fe97d5f34")
        self.assertEqual(r["state_key"], "state_dict")
        self.assertEqual(r["checkpoint_epoch_zero_based"], 299)
        self.assertEqual(r["availability"], "user_supplied")
        self.assertNotIn("url", r)
