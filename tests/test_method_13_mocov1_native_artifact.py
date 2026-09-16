"""Pinned Step-1 weight identity measured from the evaluated native checkpoint."""
import json
from pathlib import Path
import unittest

class TestArtifact(unittest.TestCase):
    def test_actual_step1_identity_and_explicit_mapping(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "methods/13_mocov1/provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(record["sha256"], "36ea58a02dd8f522c406c5bf54726f99dc26f8c80be40316a6ac4f0f094c0879")
        self.assertEqual(record["state_key"], "state_dict")
        self.assertEqual(record["strip_prefix"], "")
        self.assertEqual(record["checkpoint_epoch_zero_based"], 199)
        self.assertEqual(record["availability"], "user_supplied")
        self.assertNotIn("url", record)
        self.assertTrue(record["selection_evidence"])
