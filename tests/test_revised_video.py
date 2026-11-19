"""Source clip population, normalized geometry and differentiable readout."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import numpy as np
    import torch
    from torch import nn

@unittest.skipUnless(HAVE, 'requires torch')
class RevisedVideo(unittest.TestCase):
    def test_reader_population_geometry_and_failure(self):
        from downstream.revised_video import RevisedSSV2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'labels').mkdir(); (root/'videos').mkdir()
            (root/'labels/labels.json').write_text(json.dumps({'Putting [something]': '3'}))
            (root/'labels/validation.json').write_text(json.dumps([
                {'id':'1','template':'Putting [something]'},
                {'id':'2','template':'Putting [something]'}]))
            (root/'videos/1.mp4').touch()
            dataset = RevisedSSV2(root, 'val')
            self.assertEqual((dataset.n_entries,len(dataset),dataset.missing), (2,1,['2']))
            seen = []
            class Reader:
                def __init__(self,*args,**kwargs): pass
                def __len__(self): return 3
                def get_batch(self, indices):
                    seen.extend(indices)
                    return SimpleNamespace(asnumpy=lambda: np.full((16,260,300,3),255,dtype=np.uint8))
            with patch.dict('sys.modules', {'decord': SimpleNamespace(VideoReader=Reader)}):
                clip, label = dataset[0]
            self.assertEqual(label,3)
            self.assertEqual(tuple(clip.shape),(16,3,224,224))
            self.assertTrue(torch.allclose(clip[:,0],torch.full_like(clip[:,0],(1-.485)/.229)))
            self.assertEqual(set(seen),{0,1,2})
            with patch.dict('sys.modules', {'decord':SimpleNamespace(VideoReader=lambda *a,**k: [])}), self.assertRaises(ValueError):
                dataset[0]

    def test_native_video_gradient_and_mode(self):
        from downstream.revised_video import VideoTask
        class Body(nn.Module):
            out_channels=3
            def __init__(self):
                super().__init__(); self.scale=nn.Parameter(torch.tensor([1.,2.,3.]))
            def video_tokens(self,x): return x.mean((-2,-1))*self.scale
        body=Body(); model=VideoTask(body).train()
        clips=torch.randn(2,16,3,4,4)
        expected=torch.nn.functional.normalize(body.video_tokens(clips).mean(1),dim=-1)
        torch.testing.assert_close(model.features(clips),expected)
        self.assertFalse(body.training); self.assertTrue(model.head.training)
        self.assertEqual(tuple(model(clips).shape),(2,174))
        model(clips).square().sum().backward()
        assert body.scale.grad is not None
        self.assertGreater(body.scale.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError): model.features(clips[:,0])

    def test_provider_readout_is_used_without_renormalization(self):
        from downstream.revised_video import VideoTask
        class Body(nn.Module):
            out_channels=3
            def classification_features(self,x,*,adaptation,video):
                self.observed=(adaptation,video)
                return x.mean((1,3,4))*7
        body=Body(); model=VideoTask(body)
        clips=torch.ones(2,16,3,4,4)
        torch.testing.assert_close(model.features(clips),torch.full((2,3),7.))
        self.assertEqual(body.observed,('frozen',True))
