"""Actual Step-1 checkpoint identity, measured from the referenced artifact."""
import json
from pathlib import Path
import unittest

class TestArtifact(unittest.TestCase):
    def test_recorded_native_identity(self):
        root = Path(__file__).resolve().parents[1]
        r = json.loads((root / "methods/11_cpc/provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(r["sha256"], "4ea97c315875d3d677670e8f1a17fcb310b9189b528ec7ec1a133d4918710f76")
        self.assertEqual(r["state_key"], "state_dict")
        self.assertEqual(r["checkpoint_epoch_zero_based"], 199)
        self.assertEqual(r["availability"], "user_supplied")
        self.assertNotIn("url", r)
