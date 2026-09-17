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
METHOD = ROOT / 'methods' / '31_dinov3'
PROFILE = 'official_vitb16_cls512'

class TestArtifact(unittest.TestCase):
    def test_exact_native_artifact(self):
        record = json.loads((METHOD / 'provenance.json').read_text())['step1_native_artifact']
        self.assertEqual(record['sha256'], '73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c')
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
        self.assertEqual(tuple(out.shape), (3, 512, 512))
        torch.testing.assert_close(out[:,0,0], (torch.tensor([20.,80.,160.])/255-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225]), rtol=0, atol=1e-6)
        self.assertEqual(transform.transforms[0].size, 585)
        self.assertEqual(transform.transforms[0].interpolation.value, 'bicubic')

    def test_official_loader_is_local_strict_and_selects_normalized_cls(self):
        import provider_support
        native = provider_support.import_sibling(METHOD, 'native_step1')
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.storage_tokens = torch.nn.Parameter(torch.ones(1,4,3))
            def forward_features(self, x):
                return {'x_norm_clstoken': x.mean((2,3)), 'x_norm_patchtokens': x.flatten(2)}
        adapter = provider_support.import_sibling(METHOD, 'adapter')
        backbone = Backbone()
        with patch.object(native, 'build_backbone', return_value=backbone):
            model = adapter.load_encoder(backbone.state_dict(), {'train': {}})
            x = torch.ones(2,3,4,4)
            with patch.object(torch, 'autocast', wraps=torch.autocast) as amp:
                self.assertTrue(torch.equal(model(x, is_global=True)[0], torch.ones(2,3)))
            amp.assert_called_once_with(device_type='cuda', dtype=torch.float16, enabled=False)
            with self.assertRaises(RuntimeError): native.load_encoder({})
        from unittest.mock import Mock
        factory = Mock(return_value=Backbone())
        with patch.object(native, 'official_factory', return_value=factory):
            native.build_backbone()
        factory.assert_called_once_with(pretrained=False)
        with patch.object(torch.hub, 'load', side_effect=AssertionError('unrelated hub heads loaded')):
            actual_factory = native.official_factory()
        self.assertEqual(Path(actual_factory.__code__.co_filename).resolve(), ROOT / 'third_party/dinov3/dinov3/hub/backbones.py')

    def test_provider_uses_actual_native_pixels_and_preserves_default(self):
        import provider_support
        from PIL import Image
        from types import SimpleNamespace
        from torchvision.datasets import ImageFolder
        from torchvision import transforms as T
        original_import = provider_support.import_sibling
        native = original_import(METHOD, 'native_step1')
        spec = importlib.util.spec_from_file_location('integration_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, x, **kwargs):
                return (x.mean((2,3)), None)
        def loader(root, split, image_size, batch_size, num_workers):
            ds = ImageFolder(str(Path(root) / split), transform=T.ToTensor())
            return ds, torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)
        def ce_loader(task, root, *, split, batch_size, num_workers, img_size, preprocess):
            return loader(root, split, img_size, batch_size, num_workers)[1]
        def extract(model, data, device, *args):
            self.assertFalse(model.training)
            self.assertFalse(model.weight.requires_grad)
            features, labels = [], []
            for x, y in data:
                features.append(model(x)[0].detach()); labels.append(y)
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
                torch.save({'storage_tokens':torch.zeros(1)} if use_native else {}, encoder)
                options = {'feature_profile': PROFILE} if use_native else {}
                encoder.with_name('export.json').write_text(json.dumps({'method':METHOD.name,
                    'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options':options}))
                with patch.object(provider_support, 'import_sibling', side_effect=sibling):
                    features, labels, meta = provider.extract_val_features(encoder_path=str(encoder), data_root=tmp,
                        split='val', device='cpu', batch_size=1, num_workers=0)
                transform = native.make_transform() if use_native else T.ToTensor()
                torch.testing.assert_close(torch.from_numpy(features[0]), transform(image).mean((1,2)), rtol=0, atol=1e-5)
                self.assertEqual(labels.tolist(), [0])
                self.assertEqual(meta['native_feature_options'], options)
                self.assertEqual(meta['image_size'], 512 if use_native else 224)
            # Even a valid sidecar must not reinterpret weights from another architecture.
            torch.save({}, encoder)
            encoder.with_name('export.json').write_text(json.dumps({'method':METHOD.name,
                'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options':{'feature_profile':PROFILE}}))
            with patch.object(provider_support, 'import_sibling', side_effect=sibling), self.assertRaisesRegex(ValueError, 'together'):
                provider.extract_val_features(encoder_path=str(encoder), data_root=tmp,
                    split='val', device='cpu', batch_size=1, num_workers=0)
