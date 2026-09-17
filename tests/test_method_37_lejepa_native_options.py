"""Exact native artifact identity must be explicit before acquisition."""
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
class TestNativeIdentity(unittest.TestCase):
    def test_evaluated_checkpoint_is_registered(self):
        provenance = json.loads((ROOT / 'methods/37_lejepa/provenance.json').read_text())
        self.assertIn('step1_native_artifact', provenance)
        record = provenance['step1_native_artifact']
        self.assertEqual(record['sha256'], 'f1ffa4491b5354941cca355151a684fd82434fe6963c12ec9779aff8353eed80')
        self.assertEqual(record['availability'], 'user_supplied')
        self.assertEqual(Path(record['filename']).name, record['filename'])
        self.assertGreater(record['bytes'], 0)
