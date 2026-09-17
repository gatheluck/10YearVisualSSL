"""Measured native checkpoint identity and protocol."""
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
METHOD=ROOT / "methods" / '21_barlow_twins'

class TestArtifact(unittest.TestCase):
    def test_exact_checkpoint_and_protocol(self):
        r=json.loads((METHOD / "provenance.json").read_text())["step1_native_artifact"]
        expected={'sha256': '52088398b58999aa03ef6edc71026cdb1d5ccee1d8d08fa0105de3a82fa0069b', 'state_key': '', 'strip_prefix': '', 'availability': 'user_supplied', 'add_prefix': 'backbone.'}
        self.assertEqual({k:r[k] for k in expected},expected)
        self.assertNotIn("url",r)
