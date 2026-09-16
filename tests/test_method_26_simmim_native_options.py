"""Native export protocol must reach the actual provider, preserving defaults."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods" / "26_simmim"

@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestOptions(unittest.TestCase):
    def test_provider_rejects_unsupported_options(self):
        spec = importlib.util.spec_from_file_location('native_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provider)
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'
            torch.save({}, encoder)
            options = {'interpolation': 'bilinear', 'resize_short_side': 219}
            record = {'method': METHOD.name, 'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options': options}
            encoder.with_name('export.json').write_text(json.dumps(record))
            # The real entry point must read and validate options before inference.
            # Unsupported option is a decoy carrying the valid name as prefix.
            record['feature_options'] = {'pool_extra': 'cls'}
            encoder.with_name('export.json').write_text(json.dumps(record))
            with patch.object(provider.importlib, 'import_module', side_effect=AssertionError('invalid options reached model imports')), self.assertRaisesRegex(ValueError, 'unsupported'):
                provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)

    def test_options_reach_inference_and_defaults_are_preserved(self):
        from types import SimpleNamespace
        from torchvision import transforms as T
        spec = importlib.util.spec_from_file_location('native_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        pools = []
        class Tiny(torch.nn.Module):
            blocks = [None] * 4
            def forward(self, x): return torch.ones((1, 1))
            def get_backbone(self): return self
            def get_encoder(self, pool):
                pools.append(pool)
                return self
            def get_intermediate_layers(self, x, n):
                return [torch.ones((1, 2, 1)) for _ in range(n)]
        model = Tiny()
        adapter = SimpleNamespace(load_encoder=lambda *args: model, eval_pool=lambda cfg: 'avg')
        dataset = SimpleNamespace(transform=T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)]))
        def extract(backbone, *args): return backbone(torch.zeros(1, 3)), torch.zeros(1, dtype=torch.long)
        evaluator = SimpleNamespace(_build_loader=lambda *args: (dataset, None), extract_features=extract)
        imports = {'adapter': adapter, 'evaluate_linear_simmim': evaluator}
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            config = {'train': {'img_size': 224, 'pool': 'avg'}}
            def run():
                with patch.object(provider.importlib, 'import_module', side_effect=imports.__getitem__), patch.object(provider, '_load_config', return_value=config):
                    return provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)
            default, _, default_meta = run()
            self.assertEqual(default.shape, (1, 1))
            self.assertEqual(default_meta['native_feature_options'], {})
            options = {'interpolation': 'bilinear', 'resize_short_side': 219}
            encoder.with_name('export.json').write_text(json.dumps({'method': METHOD.name, 'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options': options}))
            features, labels, meta = run()
            self.assertEqual(features.shape, (1, 1))
            self.assertEqual(meta['native_feature_options'], options)
            self.assertEqual(dataset.transform.transforms[0].interpolation, T.InterpolationMode.BILINEAR)
            self.assertEqual(dataset.transform.transforms[0].size, 219)

class TestArtifact(unittest.TestCase):
    def test_exact_native_identity_and_protocol(self):
        r = json.loads((METHOD / 'provenance.json').read_text())['step1_native_artifact']
        self.assertEqual(r['sha256'], 'ab40cb952816b59d56db0cea5ce037568267da1a49238a087d29973b0c7b7548')
        self.assertEqual(r['feature_options'], {'interpolation': 'bilinear', 'resize_short_side': 219})
        self.assertEqual(r['state_key'], 'state_dict')
        self.assertEqual(r['availability'], 'user_supplied')
        self.assertNotIn('url', r)
