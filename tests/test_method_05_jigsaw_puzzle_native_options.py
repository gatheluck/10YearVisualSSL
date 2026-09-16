"""Native profiles must change the actual feature path and input pixels."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
try:
    import torch
    from torchvision import transforms as T
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / 'methods' / '05_jigsaw_puzzle'
PROFILE = 'imagenet_cfn_pool4'

@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeProfile(unittest.TestCase):
    def provider(self):
        spec = importlib.util.spec_from_file_location('native_profile_provider', METHOD / 'feature_provider.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def sidecar(self, encoder, options):
        encoder.with_name('export.json').write_text(json.dumps({'method': METHOD.name,
            'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options': options}))

    def test_refuses_unknown_keys_and_profile_decoys_before_imports(self):
        import provider_support
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            for options in [{'feature_profile_extra': PROFILE}, {'feature_profile': PROFILE + '_extra'}]:
                self.sidecar(encoder, options)
                p = self.provider()
                with self.subTest(options=options), patch.object(p, 'importlib', SimpleNamespace(import_module=Mock(side_effect=AssertionError('premature import')))), patch.object(provider_support, 'import_sibling', side_effect=AssertionError('premature import')), self.assertRaises(ValueError):
                    p.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)

    def test_profile_changes_actual_pixels_features_and_keeps_default(self):
        import provider_support
        import numpy as np
        from PIL import Image
        from torchvision.datasets import ImageFolder
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(18)])
                self.cfn = torch.nn.Identity()
                self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            def forward(self, x): return x.mean((2,3)) + 10
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.encoder = Encoder()
            def get_encoder(self): return self.encoder
        configs = []; models = []
        def load(state, cfg):
            configs.append(cfg); model = Model(); models.append(model); return model
        def loader(root, split, size, batch, workers):
            ds = ImageFolder(str(Path(root)/split), T.Compose([T.Resize(int(size*256/224)),T.CenterCrop(size),T.ToTensor()]))
            return ds, torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False)
        def extract(enc, dl, device):
            values, labels = [], []
            for x,y in dl: values.append(enc(x).detach()); labels.append(y)
            return torch.cat(values), torch.cat(labels)
        ev = SimpleNamespace(_build_loader=loader, extract_features=extract)
        def sibling(name): return SimpleNamespace(load_encoder=load) if name=='adapter' else ev
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'val/n00000001').mkdir(parents=True)
            pixels=np.zeros((256,256,3),dtype=np.uint8);pixels[:,:,0]=255;pixels[:,:,1]=64
            Image.fromarray(pixels).save(root/'val/n00000001/a.png')
            encoder=root/'encoder.pt';torch.save({},encoder)
            p=self.provider(); original=p._load_config()
            with patch.object(p, 'importlib', SimpleNamespace(import_module=Mock(side_effect=sibling))), patch.object(provider_support,'import_sibling',side_effect=lambda directory,name:sibling(name)):
                default,_,_=p.extract_val_features(encoder_path=str(encoder),data_root=tmp,split='val',device='cpu',batch_size=1,num_workers=0)
                self.sidecar(encoder, {'feature_profile':PROFILE})
                actual,labels,meta=p.extract_val_features(encoder_path=str(encoder),data_root=tmp,split='val',device='cpu',batch_size=1,num_workers=0)
            normal=T.Normalize([.485,.456,.406],[.229,.224,.225])(T.ToTensor()(Image.fromarray(pixels))[:,16:240,16:240]).unsqueeze(0)
            expected=torch.nn.functional.adaptive_avg_pool2d(normal,(4,4)).flatten(1)
            np.testing.assert_allclose(actual,expected.numpy(),rtol=0,atol=1e-5)
            self.assertFalse(np.array_equal(actual,default))
            self.assertEqual(meta['image_size'],224)
            self.assertEqual(meta['native_feature_options'],{'feature_profile':PROFILE})
            self.assertEqual(p._load_config(),original)
            self.assertEqual(labels.tolist(),[0])

class TestArtifact(unittest.TestCase):
    def test_exact_checkpoint_and_protocol(self):
        r=json.loads((METHOD / "provenance.json").read_text())["step1_native_artifact"]
        expected={'sha256': '3c6e29e415dff98bd6ab03aa31a19176d368aaf671f2e8d3e21f2e9058199e0b', 'state_key': 'state_dict', 'strip_prefix': '', 'availability': 'user_supplied', 'feature_options': {'feature_profile': 'imagenet_cfn_pool4'}}
        self.assertEqual({k:r[k] for k in expected},expected)
        self.assertNotIn("url",r)
