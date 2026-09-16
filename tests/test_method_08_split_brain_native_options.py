"""Native profile identity must be checked at the actual provider entry point."""
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
METHOD = ROOT / 'methods' / '08_split_brain'

@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeProfile(unittest.TestCase):
    def test_rejects_unknown_option_before_loading_model(self):
        import provider_support
        spec = importlib.util.spec_from_file_location('profile_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            record = {'method': METHOD.name, 'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(),
                      'feature_options': {'config_overrides_extra': {}}}
            encoder.with_name('export.json').write_text(json.dumps(record))
            with patch('importlib.import_module', side_effect=AssertionError('invalid options reached imports')), patch.object(provider_support, 'import_sibling', side_effect=AssertionError('invalid options reached sibling imports')), self.assertRaisesRegex(ValueError, 'unsupported'):
                provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)

    def test_profile_reaches_model_inference_and_preserves_defaults(self):
        import provider_support
        from types import SimpleNamespace
        from torchvision import transforms as T
        spec = importlib.util.spec_from_file_location('positive_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        configs, calls = [], []
        class Tiny(torch.nn.Module):
            def get_encoder(self): return self
        def load(state, cfg):
            configs.append(cfg)
            return Tiny()
        def extract(*args, **kwargs):
            calls.append(kwargs)
            return torch.ones((1, 3)), torch.zeros(1, dtype=torch.long)
        dataset = SimpleNamespace(transform=T.Compose([T.Resize(224), T.CenterCrop(224)]))
        evaluator = SimpleNamespace(_build_loader=lambda *args: (dataset, None), extract_features=extract,
                                   _extract=extract, _is_mnist=lambda root: False,
                                   _imagefolder_transform=lambda size: dataset.transform,
                                   _IMAGENET_MEAN=(.485,.456,.406), _IMAGENET_STD=(.229,.224,.225))
        adapter = SimpleNamespace(load_encoder=load)
        def sibling(name): return adapter if name == 'adapter' else evaluator
        original = provider._load_config()
        options = {'lab_mode': 'numpy_float32'}
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            def run():
                with patch('importlib.import_module', side_effect=sibling), patch.object(provider_support, 'import_sibling', side_effect=lambda directory, name: sibling(name)), patch('torchvision.datasets.ImageFolder', return_value=dataset), patch('torch.utils.data.DataLoader', return_value=None):
                    return provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)
            _, _, meta = run()
            self.assertEqual(configs[-1], original)
            self.assertEqual(meta['native_feature_options'], {})
            encoder.with_name('export.json').write_text(json.dumps({'method': METHOD.name,
                'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options': options}))
            _, _, meta = run()
            self.assertEqual(configs[-1], provider_support.configure_native_config(original, options))
            self.assertEqual(meta['native_feature_options'], options)
            self.assertEqual(provider._load_config(), original)
            self.assertEqual(dataset.native_eval, True)


class TestArtifact(unittest.TestCase):
    def test_exact_checkpoint_and_protocol(self):
        r = json.loads((METHOD / "provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(r['sha256'], 'e3491f8273fa1db79cf612f456c8d26c90226052a6a256eda53d527cdba13e78')
        self.assertEqual(r['availability'], 'user_supplied')
        self.assertEqual(r['state_key'], 'state_dict')
        self.assertEqual(r['strip_prefix'], '')
        self.assertEqual(r['feature_options'], {'lab_mode': 'numpy_float32'})
        self.assertEqual(r['checkpoint_epoch_stored'], 99)
        self.assertNotIn("url", r)

@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeLab(unittest.TestCase):
    def test_float32_native_conversion_before_channel_normalization(self):
        import numpy as np
        import provider_support
        from PIL import Image
        data = provider_support.import_sibling(METHOD, 'data.split_brain_dataset')
        rgb = np.array([[[0,0,0],[255,255,255],[255,0,0],[12,123,231],[3,7,11]]], dtype=np.uint8)
        expected = np.array([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [53.24079895019531, 80.0924301147461, 67.20320892333984], [51.84928894042969, 12.918353080749512, -62.20698547363281], [1.7656440734863281, -0.4862472414970398, -1.8920332193374634]]], dtype=np.float32)
        actual = data.rgb2lab(rgb, native_eval=True)
        self.assertTrue(np.array_equal(actual, expected), (actual, expected))
        l, ab, _, _ = data._to_lab_tensors(Image.fromarray(rgb), native_eval=True)
        self.assertTrue(torch.equal(l, torch.from_numpy((expected[...,0]-50.)/50.).unsqueeze(0)))
        self.assertTrue(torch.equal(ab, torch.from_numpy(expected[...,1:]/128.).permute(2,0,1)))

    def test_unknown_lab_mode_is_refused_before_model_import(self):
        import provider_support
        spec = importlib.util.spec_from_file_location('bad_lab_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            encoder.with_name('export.json').write_text(json.dumps({'method': METHOD.name,
                'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(),
                'feature_options': {'lab_mode': 'numpy_float32_extra'}}))
            with patch('importlib.import_module', side_effect=AssertionError('invalid Lab mode reached model imports')), self.assertRaisesRegex(ValueError, 'Lab conversion'):
                provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)
