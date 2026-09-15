"""Pinned Step-1 weight identity measured from the evaluated native checkpoint."""
import json
from pathlib import Path
import unittest

class TestArtifact(unittest.TestCase):
    def test_actual_step1_identity_and_explicit_mapping(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "methods/18_sela/provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(record["sha256"], "2f15140b58a1d07b747fc678120fb972cc64accd163dbc87366e86b4f75dfc5e")
        self.assertEqual(record["state_key"], "state_dict")
        self.assertEqual(record["strip_prefix"], "")
        self.assertEqual(record["checkpoint_epoch_zero_based"], 399)
        self.assertEqual(record["availability"], "user_supplied")
        self.assertNotIn("url", record)
        self.assertTrue(record["selection_evidence"])
