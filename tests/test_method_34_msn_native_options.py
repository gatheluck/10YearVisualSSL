"""Exact native artifact identity must be explicit before acquisition."""
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
class TestNativeIdentity(unittest.TestCase):
    def test_evaluated_checkpoint_is_registered(self):
        provenance = json.loads((ROOT / 'methods/34_msn/provenance.json').read_text())
        self.assertIn('step1_native_artifact', provenance)
        record = provenance['step1_native_artifact']
        self.assertEqual(record['sha256'], '2f9362b906de76cb269d8f6583b9316efab5cab9c3b126611120dabb2f779849')
        self.assertEqual(record['availability'], 'user_supplied')
        self.assertEqual(Path(record['filename']).name, record['filename'])
        self.assertGreater(record['bytes'], 0)

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeFeatures(unittest.TestCase):
    def test_native_tensor_protocol(self):
        import provider_support
        from unittest.mock import patch
        native = provider_support.import_sibling(ROOT / 'methods/34_msn', 'native_step1')
        class Backbone(torch.nn.Module):
            def forward_blocks(self, x, num_blocks):
                self.blocks = num_blocks
                return x.mean((2,3))*3
            def forward(self, x):
                raise AssertionError('normalized training output selected')
        backbone = Backbone(); x = torch.arange(24,dtype=torch.float32).reshape(2,3,2,2)
        actual, unused = native.NativeFeatures(backbone)(x, return_before_head=True)
        self.assertEqual(backbone.blocks, 1)
        self.assertIsNone(unused)
        torch.testing.assert_close(actual, x.mean((2,3))*3, rtol=0, atol=0)

    def test_unknown_profiles_fail_before_loading_model(self):
        import hashlib, tempfile
        from unittest.mock import patch
        import provider_support
        method = ROOT / 'methods/34_msn'
        provider = provider_support.import_sibling(method, 'feature_provider')
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp)/'encoder.pt'; torch.save({},encoder)
            for options in ({'feature_profile': 'target_block1_extra', 'interpolation': 'bilinear'}, {'feature_profile_extra':'target_block1'}, {'feature_profile':'target_block1', 'interpolation':'nearest'}, {'feature_profile':'target_block1'}):
                encoder.with_name('export.json').write_text(json.dumps({'method':method.name,'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(),'feature_options':options}))
                with patch.object(provider_support,'import_sibling',side_effect=AssertionError('model imported')), self.assertRaises(ValueError):
                    provider.extract_val_features(encoder_path=str(encoder),data_root=tmp,split='val',device='cpu',batch_size=1,num_workers=0)

    def test_provider_wires_native_pipeline_and_freezes_encoder(self):
        import hashlib,tempfile
        import provider_support
        from PIL import Image
        from unittest.mock import patch
        method=ROOT/'methods/34_msn'
        provider=provider_support.import_sibling(method,'feature_provider')
        adapter=provider_support.import_sibling(method,'adapter')
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.ones(1));self.input=None
            def forward(self,x,layer=None):
                self.input=x
                if x.ndim==5:return x.mean((3,4)).transpose(1,2)
                if layer is not None:
                    if layer!=5:raise AssertionError('wrong layer')
                    return x[:,:1].mean((2,3),keepdim=True).expand(-1,128,6,6), x[:,1:2].mean((2,3),keepdim=True).expand(-1,128,6,6)
                raise AssertionError('native wrapper not selected')
            def forward_blocks(self,x,num_blocks):
                self.input=x
                if num_blocks!=1:raise AssertionError('wrong blocks')
                return x.mean((2,3))*3
        model=Backbone()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'val/a').mkdir(parents=True);Image.new('RGB',(320,270),(0,0,0)).save(root/'val/a/black.png')
            encoder=root/'encoder.pt';torch.save({},encoder)
            options=json.loads((method/'provenance.json').read_text())['step1_native_artifact']['feature_options']
            encoder.with_name('export.json').write_text(json.dumps({'method':method.name,'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(),'feature_options':options}))
            with patch.object(adapter,'load_encoder',return_value=model) as load:
                features,labels,meta=provider.extract_val_features(encoder_path=str(encoder),data_root=tmp,split='val',device='cpu',batch_size=1,num_workers=0)
            self.assertFalse(model.training);self.assertFalse(model.weight.requires_grad)
            self.assertEqual(labels.tolist(),[0]);self.assertEqual(meta['native_feature_options'],options)
            pixels=(-torch.tensor([.485,.456,.406])/torch.tensor([.229,.224,.225])).reshape(1,3,1,1).expand(1,3,224,224).contiguous()
            torch.testing.assert_close(model.input,pixels,rtol=0,atol=0)
            expected=pixels.mean((2,3))*3
            self.assertEqual(tuple(model.input.shape),(1,3,224,224))
            torch.testing.assert_close(torch.from_numpy(features),expected,rtol=0,atol=0)
