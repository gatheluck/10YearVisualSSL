"""The evaluated official backbone must not silently select the port default."""
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
METHOD = ROOT / 'methods' / '04_context_encoder'
PROFILE = 'official_caffe_pool5'

class TestArtifact(unittest.TestCase):
    def test_exact_native_artifact(self):
        record = json.loads((METHOD / 'provenance.json').read_text())['step1_native_artifact']
        self.assertEqual(record['sha256'], '6b3645908f719331309d3a9b19ba2a9e662112ab0171f6eed2f7e8f86488d31e')
        self.assertEqual(record['feature_options'], {'feature_profile': PROFILE})
        self.assertEqual(record['availability'], 'user_supplied')

@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeProfile(unittest.TestCase):
    def test_unknown_profile_refused_before_model_import(self):
        import provider_support
        spec = importlib.util.spec_from_file_location('native_provider', METHOD / 'feature_provider.py')
        p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            for options in ({'feature_profile': PROFILE + '_extra'}, {'feature_profile_extra': PROFILE}):
                encoder.with_name('export.json').write_text(json.dumps({'method': METHOD.name,
                    'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options': options}))
                with self.subTest(options=options), patch.object(provider_support, 'import_sibling', side_effect=AssertionError('premature import')), self.assertRaises(ValueError):
                    p.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)

    def test_native_pixels_and_geometry(self):
        import provider_support
        from PIL import Image
        native = provider_support.import_sibling(METHOD, 'native_step1')
        transform = native.make_transform()
        out = transform(Image.new('RGB', (620, 650), (20, 80, 160)))
        self.assertEqual(tuple(out.shape), (3, 227, 227))
        self.assertTrue(torch.equal(out[:, 0, 0], torch.tensor([56., -37., -103.])))

    def test_strict_official_state_and_actual_pool5(self):
        import provider_support
        native = provider_support.import_sibling(METHOD, 'native_step1')
        adapter = provider_support.import_sibling(METHOD, 'adapter')
        model = native.OfficialFeatures().eval()
        state = model.state_dict()
        selected = adapter.extract_encoder(state)
        self.assertEqual(set(selected), set(state))
        with self.assertRaisesRegex(RuntimeError, 'mixed'):
            adapter.extract_encoder(dict(state, decoder_extra=torch.zeros(1)))
        loaded = adapter.load_encoder(selected, {'train': {}}).eval()
        with torch.no_grad():
            x = torch.linspace(-1, 1, 3*227*227).reshape(1,3,227,227)
            self.assertEqual(tuple(loaded(x)[1].shape), (1,9216))
            self.assertTrue(torch.equal(loaded(x)[1], model(x)[1]))
        bad = dict(state); bad.pop('features.0.weight')
        with self.assertRaises(RuntimeError): native.load_encoder(bad)
        bad = dict(state); bad['features_extra.weight'] = torch.zeros(1)
        with self.assertRaises(RuntimeError): native.load_encoder(bad)
        self.assertEqual(model.features[4].groups, 2)
        self.assertEqual(model.features[10].groups, 2)
        self.assertEqual(model.features[12].groups, 2)
        self.assertEqual(model.features[3].size, 5)
        self.assertEqual(model.features[3].alpha, 1e-4)
        self.assertEqual(model.features[3].k, 1.)

    def test_provider_uses_actual_native_pixels_and_preserves_default(self):
        import provider_support
        from PIL import Image
        from types import SimpleNamespace
        from torchvision.datasets import ImageFolder
        from torchvision import transforms as T
        original_import = provider_support.import_sibling
        native = original_import(METHOD, 'native_step1')
        real_data = original_import(METHOD, 'datasets')
        spec = importlib.util.spec_from_file_location('integration_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, x, **kwargs):
                return (None, x.mean((2,3)))
        def loader(root, split, image_size, batch_size, num_workers):
            ds = ImageFolder(str(Path(root) / split), transform=T.ToTensor())
            return ds, torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)
        def ce_loader(task, root, *, split, batch_size, num_workers, img_size, preprocess):
            return real_data.create_dataloader(task, root, split=split, batch_size=batch_size, num_workers=0, img_size=img_size, preprocess=preprocess)
        def extract(model, data, device, *args):
            self.assertFalse(model.training)
            self.assertFalse(model.weight.requires_grad)
            features, labels = [], []
            for x, y in data:
                features.append(model(x)[1].detach()); labels.append(y)
            return torch.cat(features), torch.cat(labels)
        ev = SimpleNamespace(_build_loader=loader, create_dataloader=ce_loader, extract_features=extract)
        adapter = SimpleNamespace(load_encoder=lambda state, config: Toy())
        def sibling(directory, name):
            return {'adapter': adapter, 'native_step1': native}.get(name, ev)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'val/a').mkdir(parents=True)
            image = Image.new('RGB', (620,650), (20,80,160)); image.save(root/'val/a/one.png')
            encoder = root/'encoder.pt'
            for use_native in (False, True):
                torch.save({'features.0.weight': torch.zeros(1)} if use_native else {}, encoder)
                options = {'feature_profile': PROFILE} if use_native else {}
                encoder.with_name('export.json').write_text(json.dumps({'method':METHOD.name,
                    'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options':options}))
                with patch.object(provider_support, 'import_sibling', side_effect=sibling):
                    features, labels, meta = provider.extract_val_features(encoder_path=str(encoder), data_root=tmp,
                        split='val', device='cpu', batch_size=1, num_workers=0)
                transform = native.make_transform() if use_native else T.Compose([T.Resize(256), T.CenterCrop(227), T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
                torch.testing.assert_close(torch.from_numpy(features[0]), transform(image).mean((1,2)), rtol=0, atol=1e-5)
                self.assertEqual(labels.tolist(), [0])
                self.assertEqual(meta['native_feature_options'], options)
                self.assertEqual(meta['image_size'], 227 if use_native else provider._load_config()['train']['img_size'])
            # Even a valid sidecar must not reinterpret weights from another architecture.
            torch.save({}, encoder)
            encoder.with_name('export.json').write_text(json.dumps({'method':METHOD.name,
                'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options':{'feature_profile':PROFILE}}))
            with patch.object(provider_support, 'import_sibling', side_effect=sibling), self.assertRaisesRegex(ValueError, 'together'):
                provider.extract_val_features(encoder_path=str(encoder), data_root=tmp,
                    split='val', device='cpu', batch_size=1, num_workers=0)
