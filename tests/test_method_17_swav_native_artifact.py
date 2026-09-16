"""Pinned Step-1 weight identity measured from the evaluated native checkpoint."""
import json
from pathlib import Path
import unittest

class TestArtifact(unittest.TestCase):
    def test_actual_step1_identity_and_explicit_mapping(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "methods/17_swav/provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(record["sha256"], "43f45ef20af9eaf1174c6f18834d8b512138519ce8dd4cb318f003a95faee85f")
        self.assertEqual(record["state_key"], "state_dict")
        self.assertEqual(record["strip_prefix"], "")
        self.assertEqual(record["checkpoint_epoch_zero_based"], 199)
        self.assertEqual(record["availability"], "user_supplied")
        self.assertNotIn("url", record)
        self.assertTrue(record["selection_evidence"])
