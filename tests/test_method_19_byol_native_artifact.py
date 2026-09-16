"""Actual Step-1 checkpoint identity, measured from the referenced artifact."""
import json
from pathlib import Path
import unittest

class TestArtifact(unittest.TestCase):
    def test_recorded_native_identity(self):
        root = Path(__file__).resolve().parents[1]
        r = json.loads((root / "methods/19_byol/provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(r["sha256"], "85c2d6427079fcfd280f09c2fff40bdebe0ec78a03b45a337cbf0d87bc090cd7")
        self.assertEqual(r["state_key"], "state_dict")
        self.assertEqual(r["checkpoint_epoch_zero_based"], 999)
        self.assertEqual(r["availability"], "user_supplied")
        self.assertNotIn("url", r)
