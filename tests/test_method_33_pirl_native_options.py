"""Exact native artifact identity must be explicit before acquisition."""
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
class TestNativeIdentity(unittest.TestCase):
    def test_evaluated_checkpoint_is_registered(self):
        provenance = json.loads((ROOT / 'methods/33_pirl/provenance.json').read_text())
        self.assertIn('step1_native_artifact', provenance)
        record = provenance['step1_native_artifact']
        self.assertEqual(record['sha256'], 'e9d52059d65d23e79566e688ae1a65099d20e7112cf5c1c32c8b57ce483f70a0')
        self.assertEqual(record['availability'], 'user_supplied')
        self.assertEqual(Path(record['filename']).name, record['filename'])
        self.assertGreater(record['bytes'], 0)
