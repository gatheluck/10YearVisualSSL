"""Native multi-frame aggregator readout, never an image-only substitution."""
import json
import tempfile
from pathlib import Path
import unittest

try:
    import torch
    from torch.nn import functional as F
    from downstream import spatial_backbones as sb
    from provider_support import prepare_upstream
    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAVE, 'torch required')
class TestOmega(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.assertIn('vggt_omega',sb.discover_providers(),'native aggregator provider missing')
        self.provider = sb._load_provider(sb.discover_providers()['vggt_omega'])

    def fixture(self):
        prepare_upstream(ROOT/'third_party/vggt-omega',('vggt_omega',))
        from vggt_omega.models.aggregator import Aggregator
        from vggt_omega.models.layers.vision_transformer import init_weights_vit
        model = Aggregator(embed_dim=64,depth=2,num_heads=4,
                           register_attention_block_indices=[0],cached_layer_indices=(0,1))
        # The released loader expects checkpoint-initialized blocks; the bare
        # author constructor deliberately leaves masked bias and LayerScale
        # buffers uninitialized. Initialize only this synthetic checkpoint.
        for block in list(model.frame_blocks)+list(model.inter_frame_blocks):
            for name,module in block.named_modules():
                init_weights_vit(module,name)
        path = Path(self.tmp.name)/'omega.pth'
        torch.save({'model':{'aggregator.'+k:v for k,v in model.state_dict().items()}},path)
        return dict(kind='vggt_omega',arch='fixture',encoder=str(path),img_size=224,patch_size=16,
                    embed_dim=64,depth=2,num_heads=4),model

    def test_native_spatiotemporal_tokens_summary_normalization_and_updates(self):
        spec, ref = self.fixture()
        body = self.provider.build_trainable(spec)
        raw = torch.rand(2,3,3,32,32)
        mean=raw.new_tensor([.485,.456,.406])[None,None,:,None,None]
        std=raw.new_tensor([.229,.224,.225])[None,None,:,None,None]
        x=((raw-mean)/std).requires_grad_()
        y=raw.clone().requires_grad_()
        left=torch.optim.SGD(body.parameters(),lr=.01)
        right=torch.optim.SGD(ref.parameters(),lr=.01)
        for _ in range(3):
            left.zero_grad();right.zero_grad()
            expected=ref(y)[0][-1][:,:,17:,:].flatten(1,2)
            actual=body.video_tokens(x)
            self.assertEqual(actual.shape,(2,12,128))
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
            target=torch.randn_like(actual)
            (actual-target).square().mean().backward()
            (expected-target).square().mean().backward()
            torch.testing.assert_close(x.grad,y.grad*std,atol=1e-5,rtol=1e-4)
            for n,p in ref.named_parameters():
                torch.testing.assert_close(p.grad,dict(body.aggregator.named_parameters())[n].grad,atol=1e-5,rtol=1e-4)
            left.step();right.step()
            x.grad=None;y.grad=None
        expected=body.video_tokens(x).mean(1)
        torch.testing.assert_close(body.classification_features(x,adaptation='frozen',video=True),F.normalize(expected,dim=-1))
        torch.testing.assert_close(body.classification_features(x,adaptation='finetune',video=True),expected)
        image=x[:,0].detach()
        single=body.aggregator(raw[:,0:1])[0][-1][:,0,17:]
        torch.testing.assert_close(body.forward_features(image).flatten(2).transpose(1,2),single,atol=1e-5,rtol=1e-5)
        self.assertFalse(any('head' in n for n,p in body.named_parameters()))

    def test_freezing_checkpoint_validation_geometry_and_layer_groups(self):
        spec,ref=self.fixture()
        body=self.provider.build(spec);body.train()
        self.assertFalse(body.training)
        self.assertFalse(body.video_tokens(torch.randn(1,2,3,32,32)).requires_grad)
        body=self.provider.build_trainable(spec)
        endpoint,entries=body.finetune_group_policy()
        self.assertEqual(endpoint,2)
        self.assertEqual(entries['aggregator.frame_blocks.0.attn.qkv.weight'],(1,False))
        self.assertEqual(entries['aggregator.inter_frame_blocks.1.attn.qkv.weight'],(2,False))
        self.assertEqual(entries['aggregator.camera_token'],(0,True))
        for change in ({'encoder':''},{'arch':'unknown'},{'arch':'released'},{'patch_size':14}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.provider.build(dict(spec,**change))
        released={k:v for k,v in spec.items() if k not in ('embed_dim','depth','num_heads')}
        released['arch']='released'
        from unittest import mock
        # Fail before constructing a billion-parameter model for a wrong file.
        import sys
        module=sys.modules['vggt_omega.models.aggregator']
        with mock.patch.object(self.provider,'prepare_upstream'), mock.patch.object(
                module,'Aggregator',side_effect=AssertionError('unverified large model constructed')) as build:
            with self.assertRaisesRegex(ValueError,'checkpoint identity'):
                self.provider.build(released)
            build.assert_not_called()
        with mock.patch.object(self.provider,'prepare_upstream'), mock.patch.object(
                self.provider,'RELEASED_SHA256',self.provider.sha256_file(Path(spec['encoder']))), mock.patch.object(
                module,'Aggregator',return_value=ref) as build:
            accepted=self.provider.build(released)
            self.assertIs(accepted.aggregator,ref)
            build.assert_called_once_with()
        for x in (torch.randn(1,3,31,32),torch.randn(1,4,32,32)):
            with self.assertRaises(ValueError):body.forward_features(x)
        state={'aggregator.'+k:v for k,v in ref.state_dict().items()}
        state.pop(next(iter(state)))
        torch.save(state,spec['encoder'])
        with self.assertRaisesRegex((ValueError,RuntimeError),'[Mm]issing'):self.provider.build(spec)

    def test_bad_inputs_cached_outputs_and_incomplete_groups_fail_closed(self):
        from unittest import mock
        spec,_=self.fixture()
        body=self.provider.build_trainable(spec)
        for x in (torch.randn(1,3,32,32),torch.randn(1,0,3,32,32),torch.randn(1,2,4,32,32)):
            with self.assertRaises(ValueError):body.video_tokens(x)
        with self.assertRaisesRegex(ValueError,'classification'):
            body.classification_features(torch.randn(1,3,32,32),adaptation='unknown')
        for result in (([None],17),([torch.randn(1,1,21,128)],16),([torch.randn(1,1,22,128)],17)):
            with mock.patch.object(body.aggregator,'forward',return_value=result):
                with self.assertRaises(ValueError):body.forward_features(torch.randn(1,3,32,32))
        body.aggregator.inter_frame_blocks[-1].requires_grad_(False)
        with self.assertRaisesRegex(ValueError,'complete paired'):body.finetune_group_policy()

    def test_native_detector_preserves_spatial_features_and_head(self):
        from downstream import coco
        spec,reference=self.fixture()
        self.assertTrue(sb.supports_native_detection(spec['kind']), 'source detector route missing')
        self.assertEqual(sb.detection_label_space(spec['kind']),'contiguous')
        self.assertEqual(sb.detection_label_space('vit'),'category_id')
        model=coco.build_frozen_detector(spec,dict(min_size=32,max_size=64,anchor_sizes=[8,16,32,64]),
            torch.device('cpu'),profile='capture_basic5_components',detector_profile='captured_native_detection_v1')
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features,81)
        self.assertEqual(model.transform.image_mean,[0.,0.,0.])
        self.assertEqual(model.transform.image_std,[1.,1.,1.])
        raw=torch.rand(1,3,32,48)
        expected=reference(raw.unsqueeze(1))[0][-1][:,0,17:]
        actual=model.backbone.body.forward_detection_features(raw).flatten(2).transpose(1,2)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        from downstream.native_detection import TransposedFeaturePyramid
        self.assertIsInstance(model.backbone.fpn,TransposedFeaturePyramid)

    def test_native_detector_and_category_roundtrip_preserve_sparse_coco_ids(self):
        try:
            import pycocotools.coco
        except ImportError:
            self.skipTest('COCO mapping integration requires pycocotools')
        from downstream import coco
        from tests.test_downstream_coco import tiny_coco
        spec,_=self.fixture()
        root=tiny_coco(Path(self.tmp.name)/'coco')
        ann=root/'annotations/instances_val2017.json'
        obj=json.loads(ann.read_text())
        obj['categories'][1]['id']=90
        obj['annotations'][1]['category_id']=90
        ann.write_text(json.dumps(obj))
        args=(str(root/'images/val2017'),str(ann))
        ds=coco.CocoDetectionForFRCNN(*args,label_space='contiguous')
        self.assertEqual(ds[1][1]['labels'].tolist(),[2])
        self.assertEqual(coco.CocoDetectionForFRCNN(*args)[1][1]['labels'].tolist(),[90])
        class Perfect(torch.nn.Module):
            def forward(self, images):
                return [dict(boxes=torch.tensor([[8.,8.,32.,32.]]),labels=torch.tensor([i+1]),scores=torch.tensor([1.]))
                        for i in range(len(images))]
        loader=torch.utils.data.DataLoader(ds,batch_size=2,collate_fn=coco.collate)
        measured=coco.evaluate(Perfect(),loader,ds,torch.device('cpu'),profile='capture_basic5_components')
        self.assertAlmostEqual(measured['bbox_mAP'],1.)
        class Invalid(Perfect):
            def forward(self, images):
                outputs=super().forward(images)
                outputs[0]['labels'].fill_(81)
                return outputs
        with self.assertRaisesRegex(ValueError,'outside the dataset'):
            coco.evaluate(Invalid(),loader,ds,torch.device('cpu'),profile='capture_basic5_components')
        with self.assertRaisesRegex(ValueError,'label_space'):
            coco.CocoDetectionForFRCNN(*args,label_space='unknown')
        from tests.test_downstream_coco import smoke_config
        from unittest import mock
        cfg=smoke_config(root,backbone=spec,profile='capture_basic5_components',
                        detector_profile='captured_native_detection_v1',
                        optimizer_profile='basic5_frozen_v1')
        cfg['detector'].update(lr='protocol',anchor_sizes=[8,16,32,64])
        with mock.patch.object(coco,'build_frozen_detector',side_effect=AssertionError('built before mapping validation')) as build:
            with self.assertRaisesRegex(ValueError,'category mappings differ'):
                coco.run(cfg,Path(self.tmp.name)/'out')
            build.assert_not_called()
        # Category IDs remain meaningful even if a legacy subset declares
        # different categories per split. Do not impose contiguous-map rules.
        with mock.patch.object(coco,'build_frozen_detector',side_effect=AssertionError('legacy mapping accepted')):
            with self.assertRaisesRegex(AssertionError,'legacy mapping accepted'):
                coco.run(smoke_config(root),Path(self.tmp.name)/'legacy-out')

    def test_all_fourteen_task_routes_use_the_native_video_and_category_profiles(self):
        from tests.test_method_raev2 import TestK7Backbone
        def prepare(module, root):
            if module.TASK == 'coco_detection':
                # Keep the full 80-slot vocabulary even in the two-image smoke.
                for path in (root/'annotations').glob('*.json'):
                    obj=json.loads(path.read_text())
                    obj['categories']=[dict(id=i,name=str(i)) for i in list(range(1,80))+[90]]
                    path.write_text(json.dumps(obj))
        TestK7Backbone._run_task_routes(self,'captured_cross_self_v1',prepare)


class TestOmegaDelivery(unittest.TestCase):
    def test_guide_and_ci_expose_the_new_component(self):
        doc=ROOT/'docs/BASIC5_OMEGA.md'
        self.assertTrue(doc.is_file(),'native aggregator usage guide missing')
        overlay=json.loads(doc.read_text().split('```json\n',1)[1].split('```',1)[0])
        self.assertEqual(overlay['backbone']['kind'],'vggt_omega')
        if HAVE:
            from downstream.imagenet import validate_config
            from tests.test_basic5_imagenet import TestImageNet
            cfg=TestImageNet().config(Path('/path/to/data'));cfg.update(overlay)
            validate_config(cfg)
        if (ROOT/'.github/workflows/tests.yml').is_file():
            from tests.test_ci import parsed,HAVE_YAML
            from tests.test_basic5_finetune_tasks import _runs_finetune_tests
            if not HAVE_YAML:self.skipTest('PyYAML required')
            command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                         if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
            self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_vggt_omega'))


if __name__=='__main__':
    unittest.main()
