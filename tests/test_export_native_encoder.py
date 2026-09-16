"""Native checkpoint mapping must be explicit and preserve tensor keys."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'bin' / 'export-native-encoder.py'

class TestNativeMapping(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('native_export', TOOL)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_explicit_wrapper_and_leading_prefix(self):
        value = object()
        got = self.mod.unwrap({'state_dict': {'module.encoder.weight': value}}, 'state_dict', 'module.')
        self.assertEqual(got, {'encoder.weight': value})
        self.assertIs(got['encoder.weight'], value)

    def test_prefix_inside_name_is_preserved(self):
        self.assertEqual(self.mod.unwrap({'encoder.module.weight': 1}, '', 'module.'),
                         {'encoder.module.weight': 1})

    def test_colliding_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.mod.unwrap({'module.encoder.weight': 1, 'encoder.weight': 2}, '', 'module.')

    def test_missing_wrapper_does_not_guess(self):
        with self.assertRaisesRegex(ValueError, 'state_dict'):
            self.mod.unwrap({'model': {'weight': 1}}, 'state_dict', '')

    def test_empty_or_non_mapping_state_is_rejected(self):
        for value in ({}, [], 3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.mod.unwrap({'state_dict': value}, 'state_dict', '')

class TestNativeExport(unittest.TestCase):
    def test_real_tensor_export_round_trip_and_wrong_hash_refusal(self):
        import hashlib
        import json
        import subprocess
        import sys
        import tempfile
        try:
            import torch
            import yaml
        except ImportError:
            self.skipTest('torch/PyYAML unavailable; run in the method environment')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method = root / 'example'
            method.mkdir()
            (method / 'adapter.py').write_text(
                'import torch\n'
                'def extract_encoder(state): return {k:v for k,v in state.items() if k == "weight"}\n'
                'def load_encoder(state, config):\n'
                ' m=torch.nn.Linear(2, 1, bias=False)\n'
                ' m.load_state_dict(state, strict=True)\n'
                ' return m\n')
            source = root / 'native.pt'
            torch.save({'state_dict': {'module.native': torch.ones(1, 2),
                                       'module.head': torch.zeros(1)}}, source)
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            config = root / 'config.yaml'
            config.write_text('train: {}\n')
            out = root / 'out'
            cmd = [sys.executable, str(TOOL), '--source', str(source), '--sha256', sha,
                   '--method-dir', str(method), '--config', str(config), '--out', str(out),
                   '--state-key', 'state_dict', '--strip-prefix', 'module.',
                   '--module-map', json.dumps({'native': 'weight'}),
                   '--feature-options', json.dumps({'pool': 'cls'})]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            state = torch.load(out / 'encoder.pt', weights_only=True)
            self.assertEqual(set(state), {'weight'})
            self.assertTrue(torch.equal(state['weight'], torch.ones(1, 2)))
            self.assertEqual(json.loads((out / 'export.json').read_text())['source_sha256'], sha)
            self.assertEqual(json.loads((out / 'export.json').read_text())['feature_options'], {'pool': 'cls'})
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), sha)
            # Existing output is immutable; rerunning cannot replace it.
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            cmd[cmd.index('--out') + 1] = str(root / 'bad')
            cmd[cmd.index('--sha256') + 1] = '0' * 64
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('sha256 mismatch', r.stderr)
            self.assertFalse((root / 'bad').exists())

class TestNativeModuleMap(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('native_export_map', TOOL)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_exact_module_name_mapping_preserves_other_keys_and_values(self):
        tensor = object()
        self.assertEqual(self.mod.rename_modules(
            {'conv.weight': tensor, 'conv_extra.weight': 2, 'head.bias': 3},
            {'conv': 'encoder.0'}),
            {'encoder.0.weight': tensor, 'conv_extra.weight': 2, 'head.bias': 3})

    def test_mapping_cannot_overwrite_a_native_key(self):
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.mod.rename_modules({'conv.weight': 1, 'encoder.0.weight': 2},
                                    {'conv': 'encoder.0'})

    def test_unused_mapping_is_a_layout_error(self):
        with self.assertRaisesRegex(ValueError, 'unused'):
            self.mod.rename_modules({'conv.weight': 1}, {'missing': 'encoder.0'})

    def test_invalid_mapping_is_rejected(self):
        for mapping in ([], {'conv': ''}, {'conv.part': 'encoder.0'}, {'conv': 3}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.mod.rename_modules({'conv.weight': 1}, mapping)
