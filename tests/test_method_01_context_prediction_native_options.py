"""Measured native checkpoint identity and protocol."""
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
METHOD=ROOT / "methods" / '01_context_prediction'

class TestArtifact(unittest.TestCase):
    def test_exact_checkpoint_and_protocol(self):
        r=json.loads((METHOD / "provenance.json").read_text())["step1_native_artifact"]
        expected={'sha256': '251ca703367e782f24da76648bcceaa389db62ba1dd41b3b4378a568a179296c', 'state_key': 'state_dict', 'strip_prefix': '', 'availability': 'user_supplied'}
        self.assertEqual({k:r[k] for k in expected},expected)
        self.assertNotIn("url",r)
