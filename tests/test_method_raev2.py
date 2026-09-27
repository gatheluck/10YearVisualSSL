"""The captured seven-layer tokenizer must not become a final-layer substitute."""
import json
import tempfile
from pathlib import Path
import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from downstream import spatial_backbones as sb
    from provider_support import prepare_upstream
    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAVE, "torch required")
class TestK7Backbone(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.assertIn("raev2_k7", sb.discover_providers(), "K7 downstream provider is missing")
        self.provider = sb._load_provider(sb.discover_providers()["raev2_k7"])

    def fixture(self):
        prepare_upstream(ROOT / "third_party/dinov3", ("dinov3",))
        from dinov3.models.vision_transformer import DinoVisionTransformer
        model = DinoVisionTransformer(img_size=224, patch_size=16, embed_dim=32,
                                     depth=24, num_heads=4, n_storage_tokens=4,
                                     pos_embed_rope_rescale_coords=2.)
        model.init_weights()
        path = Path(self.tmp.name)/"encoder.pth"
        torch.save(model.state_dict(), path)
        return dict(kind="raev2_k7", arch="fixture", encoder=str(path), img_size=224,
                    patch_size=16, embed_dim=32, depth=24, num_heads=4), model

    def test_seven_selected_normalized_layers_global_term_padding_and_gradients(self):
        spec, reference = self.fixture()
        reference.norm = nn.LayerNorm(32, elementwise_affine=False)
        reference.train()
        reference.rope_embed.eval()
        actual = self.provider.build_trainable(spec)
        self.assertEqual(set(dict(actual.model.named_parameters())), set(dict(reference.named_parameters())))
        x = torch.randn(2,3,31,47, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        layers = reference.get_intermediate_layers(F.pad(x, (0,1,0,1)),
                    n=[11,13,15,17,19,21,23], reshape=False, return_class_token=False, norm=True)
        expected = torch.stack(layers).mean(0) + layers[-1].mean(1,keepdim=True)
        observed = actual.forward_features(y).flatten(2).transpose(1,2)
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(observed, layers[-1]))
        self.assertFalse(torch.allclose(observed, torch.stack(layers).sum(0)))
        target = torch.randn_like(expected)
        ((observed-target)**2).mean().backward()
        ((expected-target)**2).mean().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
        self.assertFalse(actual.model.rope_embed.training)
        for name, param in reference.named_parameters():
            other = dict(actual.model.named_parameters())[name]
            torch.testing.assert_close(param.grad, other.grad, rtol=1e-5, atol=1e-6)
        self.assertEqual(actual.forward_features(y).shape, (2,32,2,3))

    def test_freezing_temporal_readout_and_native_detection_input(self):
        spec, _ = self.fixture()
        body = self.provider.build(spec)
        body.train()
        self.assertFalse(body.training)
        x = torch.randn(2,3,32,32, requires_grad=True)
        patches = body.forward_features(x)
        self.assertFalse(patches.requires_grad)
        for mode in ("frozen", "finetune"):
            torch.testing.assert_close(body.classification_features(x, adaptation=mode),
                                       F.normalize(patches.mean((-2,-1)),dim=-1))
        clip = torch.randn(2,3,3,32,32)
        frames = body.forward_features(clip.flatten(0,1)).mean((-2,-1)).reshape(2,3,32)
        torch.testing.assert_close(body.video_tokens(clip), frames)
        torch.testing.assert_close(body.classification_features(clip, adaptation="frozen",video=True),
                                   F.normalize(frames,dim=-1).mean(1))
        raw = torch.rand(2,3,32,32)
        norm = (raw-raw.new_tensor([.485,.456,.406])[None,:,None,None])/raw.new_tensor([.229,.224,.225])[None,:,None,None]
        torch.testing.assert_close(body.forward_detection_features(raw), body.forward_features(norm))
        self.assertEqual(body.detection_normalization(), ((0.,0.,0.),(1.,1.,1.)))

    def test_checkpoint_and_geometry_refuse_substitutes(self):
        spec, _ = self.fixture()
        for change in ({"arch":"unknown"}, {"patch_size":14}, {"depth":12}, {"encoder":""},
                       {"arch":"released"}, {"img_size":518}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.provider.build(dict(spec, **change))
        state = torch.load(spec["encoder"],weights_only=True)
        del state["blocks.23.attn.qkv.weight"]
        torch.save(state,spec["encoder"])
        with self.assertRaisesRegex((ValueError,RuntimeError), "[Mm]issing"):
            self.provider.build(spec)

    def test_invalid_inputs_outputs_and_incomplete_layer_groups_fail_closed(self):
        from unittest import mock
        spec,_ = self.fixture()
        body = self.provider.build_trainable(spec)
        for image in (torch.randn(1,4,32,32),torch.randn(1,3,0,32),torch.randn(3,32,32)):
            with self.assertRaisesRegex(ValueError,'images'):body.forward_features(image)
        for clip in (torch.randn(1,3,32,32),torch.randn(1,0,3,32,32),torch.randn(1,2,4,32,32)):
            with self.assertRaisesRegex(ValueError,'video'):body.video_tokens(clip)
        with self.assertRaisesRegex(ValueError,'classification'):
            body.classification_features(torch.randn(1,3,32,32),adaptation='unknown')
        for layers in ([torch.randn(1,4,32)]*6,[torch.randn(1,5,32)]*7):
            with mock.patch.object(body.model,'get_intermediate_layers',return_value=layers):
                with self.assertRaises(ValueError):body.forward_features(torch.randn(1,3,32,32))
        body.model.blocks[-1].requires_grad_(False)
        with self.assertRaisesRegex(ValueError,'all 24'):body.finetune_group_policy()

    def test_finetune_layer_policy_and_state_updates(self):
        spec,_ = self.fixture()
        body = self.provider.build_trainable(spec)
        endpoint, entries = body.finetune_group_policy()
        self.assertEqual(endpoint,25)
        self.assertEqual(set(entries), {n for n,p in body.named_parameters() if p.requires_grad})
        self.assertEqual(entries['model.blocks.0.attn.qkv.weight'], (1,False))
        self.assertEqual(entries['model.blocks.23.norm1.weight'], (24,True))
        self.assertEqual(entries['model.storage_tokens'], (0,True))
        before = {n:p.clone() for n,p in body.named_parameters()}
        opt = torch.optim.SGD(body.parameters(),lr=.01)
        for _ in range(3):
            opt.zero_grad()
            (body.forward_features(torch.randn(2,3,32,32))-.2).square().mean().backward()
            opt.step()
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in body.named_parameters()))

    def test_all_fourteen_task_routes_write_verified_noncanonical_artifacts(self):
        self._run_task_routes('captured_single_block_v1')

    def _run_task_routes(self, reader_profile, prepare=None):
        try:
            # Dataset libraries are imported lazily by some runners. Check the
            # complete route environment before starting any training fixture.
            import pycocotools.coco, h5py, av, timm, einops
            from tests import test_basic5_optimization as opt
            from tests.test_basic5_imagenet import TestImageNet
            from downstream import imagenet, coco, nyuv2, contract
            from scipy.io import savemat
        except ImportError:
            self.skipTest("complete downstream dependencies required")
        spec, _ = self.fixture()
        count = 0
        cases = list(opt.TestOptimization().cases()) + [(imagenet, TestImageNet().fixture, TestImageNet().config)]
        for module, fixture, factory in cases:
            for adaptation in (("frozen", "attentive") if module is imagenet else ("frozen", "attentive", "finetune")):
                with self.subTest(task=module.TASK, adaptation=adaptation):
                    folder = Path(self.tmp.name)/module.TASK/adaptation
                    root, path, out = folder/'data', folder/'config.json', folder/'out'
                    fixture(root)
                    if prepare is not None:
                        prepare(module,root)
                    if module is nyuv2:
                        savemat(root/'labeled/splits.mat', {'trainNdxs':[[1],[2],[3]], 'testNdxs':[[4],[5],[6]]})
                    cfg = factory(root) if module is imagenet else opt.TestOptimization().config(module,factory,root,adaptation)
                    cfg.update(backbone=spec, adaptation=adaptation, profile='capture_basic5_components')
                    cfg['optimizer_profile'] = 'basic5_finetune_v1' if adaptation == 'finetune' else opt.PROFILE
                    if adaptation == 'attentive':
                        cfg['reader_profile'] = reader_profile
                    if module is coco:
                        cfg['detector_profile'] = 'captured_native_detection_v1'
                    cfg['detector' if module is coco else 'probe'].update(epochs=1, max_val_samples=2)
                    path.write_text(json.dumps(cfg))
                    self.assertEqual(module.main(['--config',str(path),'--out',str(out)]),0)
                    self.assertTrue(contract.verify(out,path,0)[0])
                    result = json.loads((out/'results.json').read_text())
                    self.assertFalse(result['canonical_eligible'])
                    self.assertFalse(result['record_value'])
                    self.assertEqual(result.get('reader_profile'),cfg.get('reader_profile'))
                    count += 1
        self.assertEqual(count,14)


class TestDelivery(unittest.TestCase):
    def test_usage_guide_and_ci_cover_the_real_entrypoints(self):
        doc = ROOT/'docs/BASIC5_K7.md'
        self.assertTrue(doc.is_file(), 'K7 usage guide is missing')
        overlay = json.loads(doc.read_text().split('```json\n',1)[1].split('```',1)[0])
        self.assertEqual(overlay['backbone']['kind'], 'raev2_k7')
        if HAVE:
            from downstream.imagenet import validate_config
            from tests.test_basic5_imagenet import TestImageNet
            cfg = TestImageNet().config(Path('/path/to/data'))
            cfg.update(overlay)
            validate_config(cfg)
        if (ROOT/'.github/workflows/tests.yml').is_file():
            from tests.test_ci import parsed, HAVE_YAML
            from tests.test_basic5_finetune_tasks import _runs_finetune_tests
            if not HAVE_YAML:
                self.skipTest('PyYAML required for workflow parsing')
            command = next(s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                           if s.get('name')=='Run Basic5 component contracts with downstream dependencies')
            self.assertTrue(_runs_finetune_tests(command,module='tests.test_method_raev2'))


if __name__ == '__main__':
    unittest.main()
