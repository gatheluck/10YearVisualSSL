"""Exact native artifact identity must be explicit before acquisition."""
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
class TestNativeIdentity(unittest.TestCase):
    def test_evaluated_checkpoint_is_registered(self):
        provenance = json.loads((ROOT / 'methods/12_cmc/provenance.json').read_text())
        self.assertIn('step1_native_artifact', provenance)
        record = provenance['step1_native_artifact']
        self.assertEqual(record['sha256'], '8b4d54582aadd12397e068b254534a0745fafab245c77894a846db4399be72be')
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
    def test_native_lab_matches_scikit_image_reference(self):
        import numpy as np
        from PIL import Image
        import provider_support
        data = provider_support.import_sibling(ROOT/'methods/12_cmc', 'data.cmc_dataset')
        pixels = np.array([[[0,0,0],[1,2,3],[10,20,30],[255,0,0],[0,255,0],[0,0,255],[123,64,211]]], dtype=np.uint8)
        # Independent reference: scikit-image 0.25.2 rgb2lab(uint8), float32.
        expected = np.array([[[0.,0.,0.],[0.5098250508308411,-0.1224903017282486,-0.4704960882663727],[5.948470592498779,-0.6693108081817627,-8.136411666870117],[53.2405891418457,80.0923080444336,67.20275115966797],[87.73509979248047,-86.18302917480469,83.17970275878906],[32.29567337036133,79.18559265136719,-107.8572998046875],[42.12358856201172,55.321231842041016,-66.34892272949219]]], dtype=np.float32)
        np.testing.assert_array_equal(data.RGB2Lab(native_step1=True)(Image.fromarray(pixels)), expected)

    def test_native_tensor_protocol(self):
        import provider_support
        from unittest.mock import patch
        native = provider_support.import_sibling(ROOT / 'methods/12_cmc', 'native_step1')
        class Backbone(torch.nn.Module):
            def forward(self, x, layer):
                self.layer = layer
                return x[:, :1].repeat(1,128,1,1), x[:, 1:2].repeat(1,128,1,1)
        backbone = Backbone()
        x = torch.arange(3*6*6, dtype=torch.float32).reshape(1,3,6,6)
        actual = native.NativeFeatures(backbone)(x)
        self.assertEqual(backbone.layer, 5)
        expected = torch.cat([x[:,:1].repeat(1,128,1,1), x[:,1:2].repeat(1,128,1,1)],1).flatten(1)
        self.assertEqual(actual.shape, (1,9216))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unknown_profiles_fail_before_loading_model(self):
        import hashlib, tempfile
        from unittest.mock import patch
        import provider_support
        method = ROOT / 'methods/12_cmc'
        provider = provider_support.import_sibling(method, 'feature_provider')
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp)/'encoder.pt'; torch.save({},encoder)
            for options in ({'feature_profile': 'layer5_pool6_extra'}, {'feature_profile_extra':'layer5_pool6'}, {'feature_profile':'layer5_pool6', 'interpolation':'nearest'}):
                encoder.with_name('export.json').write_text(json.dumps({'method':method.name,'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(),'feature_options':options}))
                with patch.object(provider_support,'import_sibling',side_effect=AssertionError('model imported')), self.assertRaises(ValueError):
                    provider.extract_val_features(encoder_path=str(encoder),data_root=tmp,split='val',device='cpu',batch_size=1,num_workers=0)

    def test_provider_wires_native_pipeline_and_freezes_encoder(self):
        import hashlib,tempfile
        import provider_support
        from PIL import Image
        from unittest.mock import patch
        method=ROOT/'methods/12_cmc'
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
            root=Path(tmp);(root/'val/a').mkdir(parents=True);Image.new('RGB',(320,270),(10,20,30)).save(root/'val/a/black.png')
            encoder=root/'encoder.pt';torch.save({},encoder)
            options=json.loads((method/'provenance.json').read_text())['step1_native_artifact']['feature_options']
            encoder.with_name('export.json').write_text(json.dumps({'method':method.name,'encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest(),'feature_options':options}))
            with patch.object(adapter,'load_encoder',return_value=model) as load:
                features,labels,meta=provider.extract_val_features(encoder_path=str(encoder),data_root=tmp,split='val',device='cpu',batch_size=1,num_workers=0)
            self.assertFalse(model.training);self.assertFalse(model.weight.requires_grad)
            self.assertEqual(labels.tolist(),[0]);self.assertEqual(meta['native_feature_options'],options)
            pixels=((torch.tensor([5.948470592498779,-0.6693108081817627,-8.136411666870117])-torch.tensor([50.,(-86.183+98.233)/2,(-107.857+94.478)/2]))/torch.tensor([50.,(86.183+98.233)/2,(107.857+94.478)/2])).reshape(1,3,1,1).expand(1,3,224,224).contiguous()
            torch.testing.assert_close(model.input,pixels,rtol=0,atol=0)
            means=pixels.mean((2,3))
            expected=torch.cat([means[:,:1].expand(-1,128*36),means[:,1:2].expand(-1,128*36)],1)
            self.assertEqual(tuple(model.input.shape),(1,3,224,224))
            torch.testing.assert_close(torch.from_numpy(features),expected,rtol=0,atol=2e-6)
