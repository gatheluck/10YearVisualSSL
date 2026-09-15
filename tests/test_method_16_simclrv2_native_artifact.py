"""Pinned Step-1 weight identity measured from the evaluated native checkpoint."""
import json
from pathlib import Path
import unittest

class TestArtifact(unittest.TestCase):
    def test_actual_step1_identity_and_explicit_mapping(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "methods/16_simclrv2/provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(record["sha256"], "7fb5be6598220292b6490b3e2305c3d6195ea14d96c5d6b47f886c4a37891ec9")
        self.assertEqual(record["state_key"], "state_dict")
        self.assertEqual(record["strip_prefix"], "")
        self.assertEqual(record["checkpoint_epoch_zero_based"], 799)
        self.assertEqual(record["availability"], "user_supplied")
        self.assertNotIn("url", record)
        self.assertTrue(record["selection_evidence"])
