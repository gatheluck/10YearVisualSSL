"""Explicit source-family AP and detection execute without reinterpreting legacy runs."""
import json
from pathlib import Path
import unittest
from unittest import mock

from tests import test_method_hf_basic5 as hf
from tests import test_method_patch_basic5 as patch
from tests import test_basic5_optimization as opt

HAVE = hf.HAVE and patch.HAVE and opt.HAVE
if HAVE:
    import torch
    from torch.nn import functional as F

KINDS = hf.KINDS + patch.KINDS
SINGLE = 'captured_single_block_v1'
CROSS_SELF = 'captured_cross_self_v1'
DETECTION = 'captured_native_detection_v1'


def reader_profile(kind):
    return SINGLE if kind == 'dinov3_hf' else CROSS_SELF


@unittest.skipUnless(HAVE, 'complete downstream dependencies required')
class TestNativeBasic5Paths(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.hf = hf.TestVisionFamilies()
        self.hf.setUp()
        self.addCleanup(self.hf.doCleanups)
        self.patch = patch.TestPatchFamilies()
        self.patch.setUp()
        self.addCleanup(self.patch.doCleanups)

    def fixture(self, kind, trainable=False):
        helper = self.hf if kind in hf.KINDS else self.patch
        spec, _ = helper.fixture(kind)
        return spec, helper.build(spec, trainable)

    def test_explicit_reader_profiles_validate_and_ambiguous_or_unused_options_fail(self):
        from downstream.attention import validate_adaptation
        for kind in KINDS:
            cfg = dict(profile='capture_basic5_components', adaptation='attentive',
                       backbone=dict(kind=kind), reader_profile=reader_profile(kind))
            self.assertEqual(validate_adaptation(cfg), 'attentive')
            for bad in (None, 'unknown', CROSS_SELF if kind == 'dinov3_hf' else SINGLE):
                with self.subTest(kind=kind, bad=bad), self.assertRaisesRegex(ValueError, 'reader'):
                    validate_adaptation(dict(cfg, reader_profile=bad))
            with self.assertRaisesRegex(ValueError, 'reader'):
                validate_adaptation(dict(cfg, adaptation='frozen'))
        self.assertEqual(validate_adaptation(dict(adaptation='frozen', backbone=dict(kind='vit'))), 'frozen')

    def test_readers_and_native_pyramid_reject_invalid_boundaries(self):
        from types import SimpleNamespace
        from downstream.captured_readers import CrossSelfQueryReader, SingleBlockSpatialAdapter, query_reader
        from downstream.native_detection import NativePyramidBackbone, validate_profile
        query, spatial = CrossSelfQueryReader(8), SingleBlockSpatialAdapter(8)
        for x in (torch.randn(2,8), torch.randn(2,0,8), torch.randn(2,3,7)):
            with self.assertRaisesRegex(ValueError, 'tokens'):
                query(x)
        for x in (torch.randn(2,8), torch.randn(2,8,0,3), torch.randn(2,7,3,3)):
            with self.assertRaisesRegex(ValueError, 'spatial'):
                spatial(x)
        with self.assertRaisesRegex(ValueError, 'match'):
            query_reader(SimpleNamespace(reader_profile=SINGLE, out_channels=8), CROSS_SELF)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            query_reader(SimpleNamespace(reader_profile='unknown', out_channels=8), 'unknown')
        body = torch.nn.Module()
        body.out_channels = 8
        body.native_pyramid_style = 'unverified'
        with self.assertRaisesRegex(ValueError, 'unverified'):
            NativePyramidBackbone(body, adaptation='frozen')
        body.native_pyramid_style = 'transposed'
        pyramid = NativePyramidBackbone(body, adaptation='frozen')
        for shape in ((2,8,1,3), (2,7,3,3), (2,8,3)):
            body.forward_detection_features = lambda x, shape=shape: torch.zeros(shape)
            with self.assertRaisesRegex(ValueError, 'real grid'):
                pyramid(torch.zeros(2,3,32,32))
        for kind, profile in (('vit','capture_basic5_components'), ('clip_hf','legacy')):
            with self.assertRaisesRegex(ValueError, 'detector_profile'):
                validate_profile(dict(backbone=dict(kind=kind), profile=profile, detector_profile=DETECTION))

    def test_factory_selection_matches_reference_components(self):
        from downstream.captured_readers import query_reader, spatial_adapter
        from downstream.captured_readers import CrossSelfQueryReader, SingleBlockSpatialAdapter
        from downstream.attention import QueryReader, SpatialAdapter
        for kind in KINDS:
            _, body = self.fixture(kind)
            profile = reader_profile(kind)
            for factory, expected in ((query_reader, QueryReader if profile == SINGLE else CrossSelfQueryReader),
                                      (spatial_adapter, SingleBlockSpatialAdapter if profile == SINGLE else SpatialAdapter)):
                torch.manual_seed(19)
                actual = factory(body, profile)
                torch.manual_seed(19)
                reference = expected(body.out_channels)
                self.assertIs(type(actual), expected)
                self.assertEqual(set(actual.state_dict()), set(reference.state_dict()))
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)

    def test_public_attentive_builder_preserves_explicit_profile_and_frozen_scope(self):
        from downstream import spatial_backbones as sb
        for kind in KINDS:
            spec, body = self.fixture(kind)
            with mock.patch.object(sb, 'build_frozen_backbone') as loader:
                with self.assertRaisesRegex(ValueError, 'reader_profile'):
                    sb.build_attentive_backbone(spec, torch.device('cpu'))
                loader.assert_not_called()
            wrapped = sb.build_attentive_backbone(spec, torch.device('cpu'), reader_profile=reader_profile(kind))
            wrapped.train()
            x = torch.randn(2,3,32,32, requires_grad=True)
            expected = body.forward_features(x)
            actual = wrapped.forward_features(x)
            torch.testing.assert_close(actual, expected)
            actual.square().mean().backward()
            self.assertIsNone(x.grad)
            self.assertFalse(wrapped.backbone.training)
            self.assertTrue(all(p.grad is None for p in wrapped.backbone.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in wrapped.adapter.parameters()))

    def test_image_and_video_ap_receive_actual_tokens_and_update_only_reader_and_head(self):
        from downstream.imagenet import ImageClassifier
        from downstream.ssv2 import FrozenFrameAverageClassifier
        for kind in KINDS:
            for video in (False, True):
                with self.subTest(kind=kind, video=video):
                    _, bb = self.fixture(kind)
                    model = (FrozenFrameAverageClassifier(bb, 3, adaptation='attentive', reader_profile=reader_profile(kind))
                             if video else ImageClassifier(bb, 'attentive', reader_profile=reader_profile(kind)))
                    x = torch.randn((2, 3, 3, 32, 32) if video else (2, 3, 32, 32), requires_grad=True)
                    spatial = bb.forward_features(x.flatten(0, 1) if video else x)
                    tokens = (spatial.mean((-2, -1)).reshape(2, 3, -1) if video
                              else spatial.flatten(2).transpose(1, 2))
                    seen = []
                    model.reader.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
                    initial = {n: p.clone() for n, p in bb.named_parameters()}
                    before = [p.clone() for p in model.reader.parameters()]
                    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(x)
                        F.cross_entropy(logits, torch.tensor([0, 1])).backward()
                        optimizer.step()
                    torch.testing.assert_close(seen[0], tokens)
                    self.assertIsNone(x.grad)
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.reader.parameters())))
                    for n, p in bb.named_parameters():
                        torch.testing.assert_close(p, initial[n], rtol=0, atol=0)
                        self.assertIsNone(p.grad)
                    # Ordered frame tokens reach attention; no invented temporal encoding.
                    with torch.no_grad():
                        a = model.reader(tokens)
                        b = model.reader(tokens.flip(1))
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_detection_native_normalization_padding_and_pyramids(self):
        from downstream import coco
        for kind in KINDS:
            with self.subTest(kind=kind):
                spec, bb = self.fixture(kind)
                settings = dict(min_size=32, max_size=64, anchor_sizes=[8, 16, 32, 64])
                model = coco.build_frozen_detector(spec, settings, torch.device('cpu'),
                    profile='capture_basic5_components', detector_profile=DETECTION)
                model.eval()
                images = [torch.rand(3, 27, 43), torch.rand(3, 31, 35)]
                with torch.no_grad():
                    transformed, _ = model.transform(images)
                    raw = transformed.tensors
                    mean = raw.new_tensor(model.transform.image_mean)[None, :, None, None]
                    std = raw.new_tensor(model.transform.image_std)[None, :, None, None]
                    # Invert the transform, including padded values, then independently
                    # take the ordinary provider's ImageNet-normalized input route.
                    pixels = raw * std + mean
                    normalized = (pixels - raw.new_tensor([.485,.456,.406])[None,:,None,None]) / raw.new_tensor([.229,.224,.225])[None,:,None,None]
                    expected = bb.forward_features(normalized)
                    actual = model.backbone.body.forward_detection_features(raw)
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                    maps = model.backbone(raw)
                    h, w = expected.shape[-2:]
                    self.assertEqual([tuple(v.shape[-2:]) for v in maps.values()],
                                     [(h*4,w*4),(h*2,w*2),(h,w),(h//2,w//2)])
                    self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 91)
                    self.assertEqual(model.roi_heads.score_thresh, 0. if kind == "sam3_trunk" else .05)
                    self.assertEqual(model.transform.size_divisible, 32)
                    self.assertTrue(all(not p.requires_grad for p in model.backbone.body.parameters()))
                    # For patch-14/merged maps the native grid is retained, not relabeled stride 16.
                    self.assertEqual(actual.shape, expected.shape)

    def test_dense_ap_styles_are_distinct_and_frozen(self):
        from downstream.ade20k import FrozenSegModel
        for kind in KINDS:
            spec, bb = self.fixture(kind)
            model = FrozenSegModel(bb, adaptation='attentive', reader_profile=reader_profile(kind))
            x = torch.randn(2, 3, 32, 32, requires_grad=True)
            before = {k: v.clone() for k, v in bb.named_parameters()}
            initial = [p.clone() for p in model.adapter.parameters()]
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                model(x).square().mean().backward()
                optimizer.step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(initial,model.adapter.parameters())))
            for n,p in bb.named_parameters():
                torch.testing.assert_close(p,before[n],atol=0,rtol=0)
            self.assertIsNone(x.grad)

    def test_patch14_detector_retains_native_non_square_grid(self):
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        from downstream import coco
        folder = self.hf.root / 'patch14'
        model = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            image_size=28, patch_size=14, projection_dim=24))
        model.save_pretrained(folder)
        spec = dict(kind='clip_hf', arch='fixture', encoder=str(folder), img_size=28, patch_size=14)
        detector = coco.build_frozen_detector(spec,
            dict(min_size=32, max_size=160, anchor_sizes=[8,16,32,64]), torch.device('cpu'),
            profile='capture_basic5_components', detector_profile=DETECTION)
        detector.eval()
        with torch.no_grad():
            batch, _ = detector.transform([torch.rand(3,32,97)])
            self.assertEqual(batch.tensors.shape[-2:], (32,128))
            reference = model(pixel_values=batch.tensors, interpolate_pos_encoding=True).last_hidden_state[:,1:]
            features = detector.backbone.body.forward_detection_features(batch.tensors)
            # At width 128, patch 14 yields nine tokens; stride 16 yields eight.
            self.assertEqual(features.shape[-2:], (2,9))
            torch.testing.assert_close(features.flatten(2).transpose(1,2), reference)
            self.assertEqual([x.shape[-2:] for x in detector.backbone(batch.tensors).values()],
                             [(8,36),(4,18),(2,9),(1,4)])

    def test_documented_profile_overlays_validate_on_real_task_configs(self):
        from downstream import coco, imagenet
        from tests.test_basic5_imagenet import TestImageNet
        doc = Path(__file__).resolve().parents[1] / 'docs/BASIC5_NATIVE_PATHS.md'
        self.assertTrue(doc.is_file(), 'explicit profile usage guide missing')
        overlays = json.loads(doc.read_text().split('```json\n',1)[1].split('```',1)[0])
        self.assertEqual({x['backbone']['kind'] for x in overlays}, set(KINDS))
        for overlay in overlays:
            cfg = TestImageNet().config(self.hf.root / 'data')
            cfg.update(overlay)
            imagenet.validate_config(cfg)
            for module, _, factory in opt.TestOptimization().cases():
                cfg = opt.TestOptimization().config(module, factory, self.hf.root/'data', 'attentive')
                cfg.update(overlay)
                if module is coco:
                    cfg['detector_profile'] = DETECTION
                module.validate_config(cfg)

    def test_all_new_task_entrypoints_write_profiled_verified_artifacts(self):
        from tests.test_basic5_imagenet import TestImageNet
        from downstream import imagenet, contract, coco, nyuv2
        from scipy.io import savemat
        for kind in KINDS:
            spec, _ = self.fixture(kind)
            for module, fixture, factory in list(opt.TestOptimization().cases()) + [(imagenet, TestImageNet().fixture, TestImageNet().config)]:
                adaptations = ('frozen','attentive','finetune') if module is coco else ('attentive',)
                for adaptation in adaptations:
                    with self.subTest(kind=kind, task=module.TASK, adaptation=adaptation):
                        folder = self.hf.root / 'native-runs' / kind / module.TASK / adaptation
                        root, path, out = folder/'data', folder/'cfg.json', folder/'out'
                        fixture(root)
                        if module is nyuv2:
                            savemat(root/'labeled/splits.mat', {'trainNdxs':[[1],[2],[3]], 'testNdxs':[[4],[5],[6]]})
                        cfg = factory(root) if module is imagenet else opt.TestOptimization().config(module, factory, root, adaptation)
                        cfg.update(backbone=spec, adaptation=adaptation)
                        if adaptation == 'attentive':
                            cfg.update(reader_profile=reader_profile(kind), optimizer_profile=opt.PROFILE)
                        elif adaptation == 'finetune':
                            cfg['optimizer_profile']='basic5_finetune_v1'
                        if module is coco:
                            cfg['detector_profile']=DETECTION
                        cfg['detector' if module is coco else 'probe'].update(epochs=1, max_val_samples=2)
                        path.write_text(json.dumps(cfg))
                        self.assertEqual(module.main(['--config',str(path),'--out',str(out)]),0)
                        self.assertTrue(contract.verify(out,path,0)[0])
                        result=json.loads((out/'results.json').read_text())
                        self.assertEqual(result.get('reader_profile'),cfg.get('reader_profile'))
                        self.assertEqual(result.get('detector_profile'),cfg.get('detector_profile'))
                        self.assertFalse(result['canonical_eligible'])
                        self.assertFalse(result['record_value'])

    def test_detection_profile_is_required_for_family_ap_and_rejects_ignored_choices(self):
        from downstream import coco
        for kind in KINDS:
            spec = dict(kind=kind)
            settings = dict(min_size=32, max_size=64, anchor_sizes=[8,16,32,64])
            with mock.patch.object(coco, 'build_frozen_backbone') as loader:
                for bad in (None, 'typo'):
                    with self.assertRaisesRegex((ValueError,NotImplementedError), 'detector_profile'):
                        coco.build_frozen_detector(spec,settings,torch.device('cpu'),
                            profile='capture_basic5_components', adaptation='attentive',
                            reader_profile=reader_profile(kind),detector_profile=bad)
                loader.assert_not_called()
            for anchors in ([8,16,32], [8,16,32,False], [8,16,32,-1]):
                with self.assertRaisesRegex(ValueError, 'anchor'):
                    coco.build_frozen_detector(spec,dict(settings,anchor_sizes=anchors),torch.device('cpu'),
                        profile='capture_basic5_components',detector_profile=DETECTION)

    def test_detector_losses_update_exactly_the_selected_trainable_scope(self):
        from downstream import coco
        for kind in KINDS:
            spec,_ = self.fixture(kind)
            for adaptation in ('frozen','attentive','finetune'):
                with self.subTest(kind=kind,adaptation=adaptation):
                    model = coco.build_frozen_detector(spec,
                        dict(min_size=32,max_size=64,anchor_sizes=[8,16,32,64]),torch.device('cpu'),
                        profile='capture_basic5_components',adaptation=adaptation,
                        reader_profile=reader_profile(kind) if adaptation=='attentive' else None,
                        detector_profile=DETECTION)
                    initial={n:p.detach().clone() for n,p in model.backbone.body.named_parameters()}
                    fpn_before=[p.clone() for p in model.backbone.fpn.parameters()]
                    reader_before=([p.clone() for p in model.backbone.adapter.parameters()]
                                   if adaptation=='attentive' else [])
                    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.005)
                    images=[torch.rand(3,32,40) for _ in range(2)]
                    targets=[dict(boxes=torch.tensor([[4.,4.,20.,22.]]),labels=torch.tensor([1])) for _ in images]
                    for _ in range(3):
                        model.train();optimizer.zero_grad(set_to_none=True)
                        loss=sum(model(images,targets).values())
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward();optimizer.step()
                    changed=any(not torch.equal(initial[n],p) for n,p in model.backbone.body.named_parameters())
                    self.assertEqual(changed,adaptation=='finetune')
                    self.assertTrue(any(not torch.equal(a,b) for a,b in zip(fpn_before,model.backbone.fpn.parameters())))
                    if reader_before:
                        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(reader_before,model.backbone.adapter.parameters())))
                    if adaptation!='finetune':
                        self.assertTrue(all(p.grad is None for p in model.backbone.body.parameters()))


class TestNativePathCI(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / '.github/workflows/tests.yml').is_file(),
                         'workflow definitions are not shipped in runtime images')
    def test_downstream_ci_runs_new_paths_with_full_dependencies(self):
        from tests.test_ci import parsed,HAVE_YAML
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML:
            self.skipTest('PyYAML required')
        command=next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                     if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
        self.assertTrue(_runs_finetune_tests(command,module='tests.test_native_basic5_paths'))
        self.assertTrue(_runs_finetune_tests(command,module='tests.test_captured_reader_reference'))
