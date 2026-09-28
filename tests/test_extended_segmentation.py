"""Extended segmentation preserves pixel populations and explicit label identity."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from torch import nn
    import numpy as np
    from PIL import Image
    import torchvision
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, 'requires downstream image dependencies')
class TestExtendedSegmentation(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.extended_segmentation'),
                             'Extended segmentation execution is missing')
        from downstream import extended_segmentation as seg
        self.seg = seg
        torch.set_num_threads(1)

    def test_confusion_accumulates_pixels_not_image_averages_and_ignores_255(self):
        score = self.seg.SegmentationScore(3)
        score.update(torch.tensor([[[0, 1], [2, 1]]]), torch.tensor([[[0, 1], [255, 0]]]))
        score.update(torch.tensor([[[2, 2]]]), torch.tensor([[[2, 2]]]))
        result = score.result()
        self.assertAlmostEqual(result['miou'], 200/3)
        self.assertEqual(result['pixel_accuracy'], 80)
        self.assertEqual(result['pixels'], 5)
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['classes_with_union'], 3)
        absent=self.seg.SegmentationScore(4)
        absent.update(torch.zeros(1,2,2,dtype=torch.long),torch.zeros(1,2,2,dtype=torch.long))
        self.assertEqual(absent.result()['miou'],100)
        empty = self.seg.SegmentationScore(2)
        empty.update(torch.zeros(1, 2, 2, dtype=torch.long), torch.full((1, 2, 2), 255))
        with self.assertRaisesRegex(ValueError, 'valid'): empty.result()
        for pred, target in [(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2)),
                             (torch.tensor([[[3]]]), torch.tensor([[[0]]])),
                             (torch.tensor([[[0]]]), torch.tensor([[[-1]]]))]:
            with self.assertRaises(ValueError): score.update(pred, target)

    def test_dense_head_freezes_encoder_and_updates_both_adapter_variants(self):
        class Body(nn.Module):
            out_channels = 8
            def __init__(self):
                super().__init__(); self.conv = nn.Conv2d(3, 8, 1)
            def forward_features(self, x): return self.conv(x)
        for profile in (None, 'captured_single_block_v1', 'captured_cross_self_v1'):
            torch.manual_seed(19)
            model = self.seg.DenseProbe(Body(), 3, profile)
            before = copy.deepcopy(model.backbone.state_dict())
            first = model.head.weight.detach().clone()
            self.assertEqual(model.head.bias.count_nonzero().item(), 0)
            self.assertLess(model.head.weight.std().item(), .025)
            self.assertGreater(model.head.weight.std().item(), .005)
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
            for _ in range(3):
                model.train(); opt.zero_grad()
                logits = model(torch.ones(2, 3, 2, 3), (5, 7))
                self.assertEqual(tuple(logits.shape), (2, 3, 5, 7))
                self.seg.pixel_loss(logits, torch.zeros(2, 5, 7, dtype=torch.long)).backward()
                opt.step()
            self.assertFalse(model.backbone.training)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()))
            self.assertFalse(torch.equal(first, model.head.weight))
            for key, value in before.items(): torch.testing.assert_close(value, model.backbone.state_dict()[key])
            if profile: self.assertGreater(model.adapter.output_projection.weight.abs().sum().item(), 0)
        with self.assertRaises(ValueError): self.seg.DenseProbe(Body(), 3, 'unknown')
        with self.assertRaisesRegex(ValueError, 'valid'):
            self.seg.pixel_loss(torch.zeros(1, 3, 2, 2), torch.full((1, 2, 2), 255))

    def fixture(self, root):
        rows=[]
        for i in range(4):
            Image.new('RGB', (18, 16), (80, 40, 20)).save(root/f'{i}.jpg')
            a=np.zeros((16,18), dtype=np.uint8); a[:,9:]=7
            Image.fromarray(a).save(root/f'{i}.png')
            rows.append(dict(image=f'{i}.jpg', mask=f'{i}.png'))
        data=dict(schema_version=1, classes=['a','b'], label_map={'0':0,'7':1,'255':255},
                  split_evidence='synthetic membership, not official', train=rows[:2], validation=rows[2:])
        path=root/'samples.json'; path.write_text(json.dumps(data)); return path,data

    def test_mask_mapping_precedes_nearest_resize_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,data=self.fixture(root)
            train,val,meta=self.seg.load_data(path,root,'captured_fixed224_v1')
            image,target=val[0]
            self.assertEqual(tuple(target.shape),(224,224))
            self.assertEqual(set(target.unique().tolist()),{0,1})
            self.assertEqual(target[0,0],0); self.assertEqual(target[0,-1],1)
            self.assertAlmostEqual(image[0,0,0].item(),(80/255-.485)/.229,places=1)
            self.assertEqual(meta['validation_count'],2)
            for change in ('overlap','unknown_id','bool','escape','extra','alias'):
                bad=copy.deepcopy(data)
                if change=='overlap': bad['validation'][0]['image']='0.jpg'
                if change=='unknown_id': del bad['label_map']['7']
                if change=='bool': bad['schema_version']=True
                if change=='escape': bad['train'][0]['mask']='../outside.png'
                if change=='extra': bad['extra']=1
                if change=='alias': bad['label_map']['07']=1
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError,msg=change):
                    datasets=self.seg.load_data(path,root,'captured_fixed224_v1'); datasets[0][0]
            path.write_text(json.dumps(data)); Image.new('RGB',(18,16)).save(root/'0.png')
            with self.assertRaisesRegex(ValueError,'mask'):
                self.seg.load_data(path,root,'captured_fixed224_v1')[0][0]

    def test_recipe_refuses_conflicting_metrics_and_scales_physical_batch(self):
        for adaptation in ('frozen','attentive'):
            spec=self.seg.recipe('bdd100k',adaptation)
            self.assertEqual(spec['epochs'],20)
            self.assertEqual(spec['weight_decay'],.05 if adaptation=='attentive' else .0001)
            model=nn.Linear(2,2)
            opt,scheduler=self.seg.optimizer(model,spec,batch_size=8,steps_per_epoch=2)
            self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-6)
            for _ in range(2): opt.step(); scheduler.step()
            self.assertAlmostEqual(opt.param_groups[0]['lr'],.001)
            for _ in range(38): opt.step(); scheduler.step()
            self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-6)
        for dataset in ('sun_rgbd','spacenet2_vegas','livecell','cifar10'):
            with self.assertRaises(ValueError): self.seg.recipe(dataset,'frozen')

    def test_pair_crop_keeps_image_and_mask_geometry_aligned(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,data=self.fixture(root)
            pixels=np.zeros((300,600,3),dtype=np.uint8); pixels[:,300:]=255
            labels=np.zeros((300,600),dtype=np.uint8); labels[:,300:]=7
            Image.fromarray(pixels).save(root/'wide.png'); Image.fromarray(labels).save(root/'mask.png')
            data['train'][0]=dict(image='wide.png',mask='mask.png'); path.write_text(json.dumps(data))
            train,_,_=self.seg.load_data(path,root,'captured_pair_crop224_v1')
            with patch('random.uniform',return_value=1.), patch('random.randint',side_effect=[0,150]), patch('random.random',return_value=0.):
                image,target=train[0]
            decoded=image[0]*.229+.485
            self.assertGreater(target[:,0].sum().item(),0)
            self.assertEqual(target[:,-1].sum().item(),0)
            self.assertLess((decoded-target.float()).abs().mean().item(),.02)

    def test_ade_crop_preserves_uncapped_geometry_before_random_crop(self):
        from unittest.mock import patch
        self.assertIn('captured_ade_crop224_v1',self.seg.TRANSFORMS,
                      'uncapped ADE training profile is missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,data=self.fixture(root)
            pixels=np.zeros((300,1200,3),dtype=np.uint8); pixels[:,600:]=255
            labels=np.zeros((300,1200),dtype=np.uint8); labels[:,600:]=7
            Image.fromarray(pixels).save(root/'wide.png'); Image.fromarray(labels).save(root/'mask.png')
            data['train'][0]=dict(image='wide.png',mask='mask.png'); path.write_text(json.dumps(data))
            for profile,expected_white in [('captured_ade_crop224_v1',0),('captured_pair_crop224_v1',168*224)]:
                train,_,_=self.seg.load_data(path,root,profile)
                with patch('random.uniform',return_value=1.), patch('random.randint',side_effect=[0,200]), patch('random.random',return_value=1.):
                    image,target=train[0]
                self.assertEqual(target.sum().item(),expected_white,profile)
                self.assertLess(((image[0]*.229+.485)-target.float()).abs().mean().item(),.02)


if __name__ == '__main__': unittest.main()
