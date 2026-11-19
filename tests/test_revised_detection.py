"""Revised detection normalization, source head and parameter boundaries."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(x) for x in ('torch','torchvision','pycocotools'))
if HAVE:
    import torch
    from PIL import Image
    from torch import nn

@unittest.skipUnless(HAVE, 'requires detection dependencies')
class RevisedDetection(unittest.TestCase):
    def body(self):
        class Body(nn.Module):
            out_channels=3
            native_pyramid_style='bilinear'
            def __init__(self):
                super().__init__(); self.base=nn.Parameter(torch.ones(3),requires_grad=False)
                self.lora_A=nn.Parameter(torch.ones(3)); self.lora_B=nn.Parameter(torch.zeros(3))
            def forward_features(self,x):
                self.seen=x
                return torch.nn.functional.avg_pool2d(x,16)*(self.base+self.lora_A*self.lora_B)[None,:,None,None]
        return Body()

    def test_normalization_geometry_head_and_gradient(self):
        from downstream.revised_detection import build_detector, detection_features
        body=self.body(); model=build_detector(body, diagnostic={'size':32,'proposals':8})
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features,91)
        images,_=model.transform([torch.ones(3,32,40)],None)
        height, _width = images.image_sizes[0]
        self.assertLess(height, images.tensors.shape[-2])
        padding = images.tensors[0,:,height:,:]
        self.assertGreater(padding.numel(), 0)
        self.assertEqual(float(padding.abs().sum()), 0.)
        torch.testing.assert_close(images.tensors[0,:,0,0],torch.tensor([(1-.485)/.229,(1-.456)/.224,(1-.406)/.225]))
        model.train(); self.assertFalse(body.training)
        loss=model([torch.ones(3,32,40)],[{'boxes':torch.tensor([[4.,4.,24.,24.]]),'labels':torch.tensor([1])}])
        sum(loss.values()).backward()
        self.assertGreater(body.lora_B.grad.abs().sum().item(),0)
        detection_features(body,torch.ones(1,3,32,32))
        torch.testing.assert_close(body.seen[0,:,0,0],torch.tensor([(1-.485)/.229,(1-.456)/.224,(1-.406)/.225]))

    def test_parameter_boundary(self):
        from downstream.revised_detection import (
            build_detector,
            detector_parameter_groups,
        )
        body=self.body(); model=build_detector(body,diagnostic={'size':32,'proposals':8})
        groups=detector_parameter_groups(model,{"attention":body},SimpleNamespace(head_lr=.02,weight_decay=.0001))
        actual=[id(p) for g in groups for p in g['params']]
        self.assertEqual(set(actual),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(len(actual),len(set(actual)))
        with self.assertRaises(ValueError):
            detector_parameter_groups(model,[body,self.body()],SimpleNamespace(head_lr=.02,weight_decay=.0001))
        body.base.requires_grad_(True)
        with self.assertRaises(ValueError): detector_parameter_groups(model,[body],SimpleNamespace(head_lr=.02,weight_decay=.0001))

    def test_reader_raw_python_flip_and_box_filter(self):
        from downstream.revised_detection import RevisedCoco
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images/train2017').mkdir(parents=True); (root/'annotations').mkdir()
            Image.new('RGB',(20,16),(255,0,0)).save(root/'images/train2017/a.png')
            annotations=[{'id':i,'image_id':1,'category_id':7,'bbox':b,'iscrowd':crowd,'area':10} for i,b,crowd in [(1,[2,3,5,6],0),(2,[0,0,1,3],0),(3,[0,0,4,4],1)]]
            (root/'annotations/instances_train2017.json').write_text(json.dumps({'images':[{'id':1,'file_name':'a.png','width':20,'height':16}],'categories':[{'id':7,'name':'x'}],'annotations':annotations}))
            data=RevisedCoco(root,'train')
            with patch('random.random',return_value=0): image,target=data[0]
            self.assertEqual(image[:,0,0].tolist(),[1.,0.,0.])
            self.assertEqual(target['boxes'].tolist(),[[13.,3.,18.,9.]])
            self.assertEqual(target['labels'].tolist(),[1])

    def test_evaluation_percent_ap_and_unused_label_discard(self):
        from torch.utils.data import DataLoader

        from downstream.revised_detection import (
            RevisedCoco,
            collate,
            evaluate_detection,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images/val2017').mkdir(parents=True); (root/'annotations').mkdir()
            Image.new('RGB',(32,32)).save(root/'images/val2017/a.png')
            (root/'annotations/instances_val2017.json').write_text(json.dumps({'info':{},'images':[{'id':1,'file_name':'a.png','width':32,'height':32}],'categories':[{'id':7,'name':'x'}],'annotations':[{'id':1,'image_id':1,'category_id':7,'bbox':[4,4,16,16],'iscrowd':0,'area':256}]}))
            class Detector(nn.Module):
                def forward(self,images):
                    return [{'boxes':torch.tensor([[4.,4.,20.,20.],[0.,0.,10.,10.]]),'scores':torch.tensor([.9,.99]),'labels':torch.tensor([1,81])} for image in images]
            loader=DataLoader(RevisedCoco(root,'val'),collate_fn=collate)
            result=evaluate_detection(Detector(),loader,torch.device('cpu'))
            self.assertAlmostEqual(result['bbox_ap'],100.,places=5)
            self.assertEqual((result['n_images'],result['n_detections']),(1,1))

    def test_explicit_provider_bindings_and_diagnostics(self):
        from downstream.revised_detection import build_detector, detection_features
        body=self.body()
        body.native_pyramid_style='transposed'
        body.forward_detection_features=body.forward_features
        body.family='map_pool'
        detection_features(body,torch.full((1,3,32,32),.3))
        torch.testing.assert_close(body.seen,torch.full_like(body.seen,-.4))
        del body.family
        detection_features(body,torch.full((1,3,32,32),.3))
        torch.testing.assert_close(body.seen,torch.full_like(body.seen,.3))
        body.family='uninspected'
        with self.assertRaises(ValueError): build_detector(body)
        body=self.body()
        for diagnostic in ({'size':0},{'proposals':False},{'unknown':1}):
            with self.subTest(diagnostic=diagnostic), self.assertRaises(ValueError):
                build_detector(body,diagnostic)
        model=build_detector(body,{'size':32,'proposals':8})
        model.unexpected=nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            from downstream.revised_detection import detector_parameter_groups
            detector_parameter_groups(model,[body],SimpleNamespace(head_lr=.02,weight_decay=.0001))

    def test_evaluation_rejects_invalid_population_and_predictions(self):
        from downstream.revised_detection import evaluate_detection
        class Loader:
            dataset=SimpleNamespace(label_to_category={1:7})
            def __init__(self, count): self.count=count
            def __iter__(self):
                for _ in range(self.count):
                    yield [torch.zeros(3,32,32)],[{'image_id':torch.tensor([1])}]
        class Detector(nn.Module):
            def __init__(self, bad=False, missing=False):
                super().__init__(); self.bad=bad; self.missing=missing
            def forward(self, images):
                if self.missing: return []
                return [{'boxes':torch.tensor([[1.,1.,4.,4.]]),
                         'scores':torch.tensor([float('nan') if self.bad else .9]),
                         'labels':torch.tensor([1])}]
        for model,loader in ((Detector(True),Loader(1)),(Detector(missing=True),Loader(1)),
                             (Detector(),Loader(0)),(Detector(),Loader(2))):
            with self.assertRaises(ValueError): evaluate_detection(model,loader,'cpu')
