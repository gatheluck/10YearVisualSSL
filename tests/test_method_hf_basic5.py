"""Released vision-family readouts, strict local loading and task integration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import (CLIPVisionConfig, CLIPVisionModelWithProjection,
                              SiglipVisionConfig, SiglipVisionModel,
                              DINOv3ViTConfig, DINOv3ViTModel)
    HAVE = True
except ImportError:
    HAVE = False

from tests._checkout import needs_checkout

KINDS = ("clip_hf", "siglip2_g", "dinov3_hf")


class TestEnvironment(unittest.TestCase):
    @unittest.skipUnless(HAVE, "torch and transformers required")
    def test_without_scipy_only_task_integration_is_skipped(self):
        import subprocess
        import sys
        script = '''
import sys
import unittest
sys.modules['scipy'] = None
from tests import test_method_hf_basic5 as module
assert module.HAVE, 'model dependencies must actually be available'
suite = unittest.TestSuite(module.TestVisionFamilies(name) for name in (
    'test_real_task_entrypoints_write_verified_noncanonical_results',
    'test_detection_rejects_unreconciled_stride_or_padding_normalization'))
result = unittest.TestResult()
suite.run(result)
assert result.testsRun == 2, result.testsRun
assert not result.errors and not result.failures, (result.errors, result.failures)
assert len(result.skipped) == 1, result.skipped
assert 'downstream' in result.skipped[0][1].lower(), result.skipped
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @needs_checkout
    def test_ci_executes_family_tests_with_required_imports(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import parsed, HAVE_YAML
        import ast
        import shlex
        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        commands = [s['run'] for s in parsed()['tests.yml']['jobs']['downstream']['steps']
                    if s.get('name') == 'Run Basic5 component contracts with downstream dependencies']
        self.assertEqual(len(commands), 1)
        self.assertTrue(_runs_finetune_tests(commands[0], module="tests.test_method_hf_basic5"))
        imports = set()
        for line in commands[0].replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[:2] == ['.venv/bin/python', '-c']:
                for node in ast.walk(ast.parse(words[2])):
                    if isinstance(node, ast.ImportFrom) and node.module == 'transformers':
                        imports.update(a.name for a in node.names)
        self.assertTrue({'CLIPVisionModelWithProjection', 'SiglipVisionModel', 'DINOv3ViTModel'} <= imports)


@unittest.skipUnless(HAVE, "torch and transformers required")
class TestVisionFamilies(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, kind):
        common = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, image_size=32, patch_size=16)
        if kind == "clip_hf":
            model = CLIPVisionModelWithProjection(CLIPVisionConfig(**common, projection_dim=24))
        elif kind == "siglip2_g":
            model = SiglipVisionModel(SiglipVisionConfig(**common))
        else:
            model = DINOv3ViTModel(DINOv3ViTConfig(**common, num_register_tokens=4,
                                                pos_embed_rescale=None))
        model.eval()
        folder = self.root / kind
        model.save_pretrained(folder)
        return dict(kind=kind, encoder=str(folder), arch="fixture", img_size=32, patch_size=16), model

    def build(self, spec, trainable=False):
        from downstream import spatial_backbones as sb
        self.assertIn(spec["kind"], sb.KINDS, "released-family downstream provider missing")
        return (sb.build_trainable_backbone if trainable else sb.build_frozen_backbone)(spec, torch.device("cpu"))

    def expected(self, kind, model, x):
        if kind == "clip_hf":
            mean, std = [.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711]
        elif kind == "siglip2_g":
            mean, std = [.5]*3, [.5]*3
        else:
            mean, std = [.485, .456, .406], [.229, .224, .225]
        # Input is the task's ImageNet-normalized tensor; compare independently.
        pixel = (x * x.new_tensor([.229,.224,.225])[None,:,None,None]
                 + x.new_tensor([.485,.456,.406])[None,:,None,None])
        pixel = (pixel-x.new_tensor(mean)[None,:,None,None])/x.new_tensor(std)[None,:,None,None]
        if kind == "dinov3_hf":
            pixel = F.pad(pixel, (0, -pixel.shape[-1]%16, 0, -pixel.shape[-2]%16))
            out = model(pixel_values=pixel)
            pooled, patches = out.pooler_output, out.last_hidden_state[:,5:]
        else:
            out = model(pixel_values=pixel, interpolate_pos_encoding=True)
            pooled = out.image_embeds if kind == "clip_hf" else out.pooler_output
            patches = out.last_hidden_state[:,1:] if kind == "clip_hf" else out.last_hidden_state
        return pooled, patches, pixel.shape[-2:]

    def test_official_global_and_spatial_readouts_and_normalization(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                spec, model = self.fixture(kind)
                bb = self.build(spec)
                x = torch.randn(2,3,23,39)
                pooled, patches, hw = self.expected(kind, model, x)
                torch.testing.assert_close(bb.classification_features(x, adaptation="frozen"), F.normalize(pooled,dim=-1), atol=2e-5,rtol=2e-5)
                spatial = bb.forward_features(x)
                torch.testing.assert_close(spatial.flatten(2).transpose(1,2), patches, atol=2e-5,rtol=2e-5)
                self.assertEqual(spatial.shape[-2:], (hw[0]//16,hw[1]//16))
                self.assertEqual(bb.global_channels, pooled.shape[-1])
                self.assertFalse(any('text' in n for n,_ in bb.named_parameters()))
                bb.train()
                self.assertFalse(bb.training)
                self.assertFalse(spatial.requires_grad)

    def test_frame_readout_uses_pooler_and_preserves_family_normalization_order(self):
        from downstream.ssv2 import FrozenFrameAverageClassifier
        for kind in KINDS:
            spec, model = self.fixture(kind)
            bb = self.build(spec)
            clip = torch.randn(2,3,3,32,32)
            pooled, _, _ = self.expected(kind, model, clip.flatten(0,1))
            expected = (F.normalize(pooled,dim=-1) if kind=="dinov3_hf" else pooled).reshape(2,3,-1).mean(1)
            torch.testing.assert_close(bb.classification_features(clip,adaptation="frozen",video=True),expected,atol=2e-5,rtol=2e-5)
            task = FrozenFrameAverageClassifier(bb,5)
            with torch.no_grad(): task.classifier.weight.normal_()
            torch.testing.assert_close(task(clip),task.classifier(expected),atol=2e-5,rtol=2e-5)
            self.assertEqual(task.classifier.in_features,bb.global_channels)

    def test_imagenet_uses_global_projection_and_family_head_initialization(self):
        from downstream.imagenet import ImageClassifier
        for kind in KINDS:
            spec,_ = self.fixture(kind)
            bb=self.build(spec)
            torch.manual_seed(91)
            task=ImageClassifier(bb,"frozen")
            self.assertEqual(task.classifier.in_features,bb.global_channels)
            if kind=="dinov3_hf": self.assertEqual(torch.count_nonzero(task.classifier.weight),0)
            else: self.assertGreater(task.classifier.weight.std().item(),.005)
            x=torch.randn(2,3,32,32)
            with torch.no_grad(): task.classifier.weight.normal_()
            torch.testing.assert_close(task(x),task.classifier(bb.classification_features(x,adaptation="frozen")))

    def test_classifier_initialization_matches_one_captured_head_draw(self):
        from downstream.ssv2 import FrozenFrameAverageClassifier
        from downstream.imagenet import ImageClassifier
        for kind in KINDS:
            spec, _ = self.fixture(kind)
            bb = self.build(spec)
            for constructor, classes in ((lambda: ImageClassifier(bb, 'frozen'), 1000),
                                         (lambda: FrozenFrameAverageClassifier(bb, 5), 5)):
                torch.manual_seed(123)
                expected = nn.Linear(bb.global_channels, classes)
                if kind == 'dinov3_hf':
                    nn.init.zeros_(expected.weight)
                else:
                    nn.init.normal_(expected.weight, std=.01)
                nn.init.zeros_(expected.bias)
                torch.manual_seed(123)
                actual = constructor().classifier
                torch.testing.assert_close(actual.weight, expected.weight, atol=0, rtol=0)
                torch.testing.assert_close(actual.bias, expected.bias, atol=0, rtol=0)

    def test_documented_backbone_examples_validate_with_online_image_config(self):
        from tests.test_basic5_imagenet import TestImageNet
        from downstream.imagenet import validate_config
        path = Path(__file__).resolve().parents[1]/'docs/BASIC5_VISION_PROVIDERS.md'
        self.assertTrue(path.is_file(), 'provider usage documentation missing')
        text = path.read_text()
        specs = json.loads(text.split('```json\n',1)[1].split('```',1)[0])
        self.assertEqual({spec['kind'] for spec in specs}, set(KINDS))
        for spec in specs:
            cfg = TestImageNet().config(Path('/data/imagenet'))
            cfg['backbone'] = spec
            validate_config(cfg)
            self.assertEqual(spec['arch'], 'released')

    def test_trainable_forward_gradients_and_no_grad_context(self):
        for kind in KINDS:
            spec, model=self.fixture(kind)
            bb=self.build(spec,True)
            x=torch.randn(2,3,32,32,requires_grad=True)
            y=x.detach().clone().requires_grad_()
            expected,_,_=self.expected(kind,model,y)
            if kind=="dinov3_hf": expected=F.normalize(expected,dim=-1)
            actual=bb.classification_features(x,adaptation="finetune")
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            weights=torch.randn_like(actual)
            (actual*weights).sum().backward(); (expected*weights).sum().backward()
            torch.testing.assert_close(x.grad,y.grad,atol=4e-5,rtol=4e-5)
            self.assertTrue(any(p.grad is not None for p in bb.parameters()))
            with torch.no_grad(): self.assertFalse(bb.forward_features(x).requires_grad)

    def test_local_checkpoint_is_required_and_incompatible_shapes_are_refused(self):
        for kind in KINDS:
            spec,_=self.fixture(kind)
            with self.assertRaises((ValueError,RuntimeError)): self.build(dict(spec,arch="released"))
            with self.assertRaises((ValueError,FileNotFoundError)): self.build(dict(spec,encoder=""))
            with self.assertRaises((ValueError,FileNotFoundError)): self.build(dict(spec,encoder="remote/model"))
            with self.assertRaises(ValueError): self.build(dict(spec,patch_size=14))
            with self.assertRaises(ValueError): self.build(dict(spec,arch='typo'))
            with self.assertRaises(ValueError): self.build(dict(spec,img_size=224))
            other = 'siglip2_g' if kind != 'siglip2_g' else 'dinov3_hf'
            with self.assertRaises(ValueError): self.build(dict(spec,kind=other))

    def test_full_multimodal_snapshot_preserves_projection_without_loading_text(self):
        from transformers import CLIPConfig, CLIPModel, SiglipConfig, SiglipModel
        for kind, config_class, model_class in (
                ('clip_hf', CLIPConfig, CLIPModel), ('siglip2_g', SiglipConfig, SiglipModel)):
            spec, vision = self.fixture(kind)
            cfg = config_class(vision_config=vision.config.to_dict(),
                               text_config=dict(hidden_size=32, intermediate_size=64,
                                                num_hidden_layers=2, num_attention_heads=4,
                                                vocab_size=32), projection_dim=20)
            full = model_class(cfg).eval()
            full.save_pretrained(spec['encoder'])
            bb = self.build(spec)
            pixels = torch.randn(2,3,32,32)
            normalized = bb._pixels(pixels)
            with torch.no_grad():
                output = full.vision_model(pixel_values=normalized)
                expected = (full.visual_projection(output.pooler_output) if kind == 'clip_hf'
                            else output.pooler_output)
            torch.testing.assert_close(bb.classification_features(pixels, adaptation='frozen'),
                                       F.normalize(expected, dim=-1), atol=2e-5, rtol=2e-5)
            self.assertFalse(any('text' in name for name,_ in bb.named_parameters()))

    def test_missing_weights_are_not_silently_randomly_initialized(self):
        from safetensors.torch import load_file,save_file
        for kind in KINDS:
            spec,_=self.fixture(kind)
            p=Path(spec['encoder'])/'model.safetensors'
            state=load_file(p); state.pop(next(iter(state))); save_file(state,p)
            with self.assertRaisesRegex((ValueError,RuntimeError),'missing|incomplete'):
                self.build(spec)

    def test_unexpected_nontext_weights_are_rejected(self):
        from safetensors.torch import load_file, save_file
        spec, _ = self.fixture('clip_hf')
        path = Path(spec['encoder'])/'model.safetensors'
        state = load_file(path)
        state['unknown_tower.weight'] = torch.ones(2)
        save_file(state, path)
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            self.build(spec)

    def test_invalid_tensor_layouts_and_incomplete_blocks_are_rejected(self):
        from types import SimpleNamespace
        for kind in KINDS:
            spec, _ = self.fixture(kind)
            bb = self.build(spec, True)
            with self.assertRaisesRegex(ValueError, 'images'):
                bb.forward_features(torch.randn(2,4,32,32))
            with self.assertRaisesRegex(ValueError, 'video'):
                bb.classification_features(torch.randn(2,3,32,32),adaptation='frozen',video=True)
            with self.assertRaisesRegex(ValueError, 'reader'):
                bb.classification_features(torch.randn(2,3,32,32),adaptation='attentive')
            with mock.patch.object(bb, '_forward', return_value=(SimpleNamespace(
                    last_hidden_state=torch.zeros(2,0,32)), (32,32))):
                with self.assertRaisesRegex(ValueError, 'grid'):
                    bb.forward_features(torch.randn(2,3,32,32))
            with mock.patch.object(bb, '_forward', return_value=(SimpleNamespace(
                    image_embeds=None,pooler_output=None), (32,32))):
                with self.assertRaisesRegex(ValueError, 'pooling'):
                    bb.classification_features(torch.randn(2,3,32,32),adaptation='frozen')
            bb.model.config.num_hidden_layers += 1
            with self.assertRaisesRegex(ValueError, 'blocks'):
                bb.finetune_group_policy()

    def test_parameter_groups_cover_all_layers_and_use_family_endpoints(self):
        for kind in KINDS:
            spec,_=self.fixture(kind); bb=self.build(spec,True)
            endpoint, policy=bb.finetune_group_policy()
            self.assertEqual(endpoint,3 if kind=="dinov3_hf" else 2)
            self.assertEqual(set(policy),set(dict(bb.named_parameters())))
            seen=set()
            for name,(layer,nd) in policy.items():
                if '.layers.0.' in name or '.layer.0.' in name: self.assertEqual(layer,1); seen.add(0)
                if '.layers.1.' in name or '.layer.1.' in name: self.assertEqual(layer,2); seen.add(1)
                if name.endswith('.bias'): self.assertTrue(nd)
                if 'visual_projection' in name: self.assertEqual(layer,2)
            self.assertEqual(seen,{0,1})

    def test_ambiguous_attentive_recipe_is_refused_before_data_access(self):
        from downstream.attention import validate_adaptation
        from downstream.spatial_backbones import build_attentive_backbone
        for kind in KINDS:
            self.assertRaisesRegex(ValueError,'attentive|reader',validate_adaptation,
                dict(profile="capture_basic5_components",adaptation="attentive",backbone=dict(kind=kind)))
            with self.assertRaisesRegex(ValueError, 'attentive|reader'):
                build_attentive_backbone(dict(kind=kind), torch.device('cpu'))

    def test_detection_rejects_unreconciled_stride_or_padding_normalization(self):
        from downstream.spatial_backbones import supports_capture_pyramid
        self.assertFalse(supports_capture_pyramid('clip_hf'))
        self.assertFalse(supports_capture_pyramid('siglip2_g'))
        self.assertTrue(supports_capture_pyramid('dinov3_hf'))

    def test_real_task_entrypoints_write_verified_noncanonical_results(self):
        from tests import test_basic5_optimization, test_basic5_imagenet
        if not (test_basic5_optimization.HAVE and test_basic5_imagenet.HAVE):
            self.skipTest("Full downstream dependencies required for task integration")
        from tests.test_basic5_optimization import TestOptimization, nyuv2, coco
        from tests.test_basic5_imagenet import TestImageNet
        from downstream import imagenet, contract
        from scipy.io import savemat
        for kind in KINDS:
            spec, _ = self.fixture(kind)
            image_helper = TestImageNet()
            cases = list(TestOptimization().cases())
            cases.append((imagenet, image_helper.fixture, image_helper.config))
            for module, fixture, factory in cases:
                if module is coco and kind != 'dinov3_hf':
                    continue  # Explicitly rejected by the separate boundary test.
                for adaptation in ('frozen', 'finetune'):
                    if module is imagenet and adaptation == 'finetune':
                        continue  # Existing runner rejects unreconciled augmentation.
                    with self.subTest(kind=kind, task=module.TASK, adaptation=adaptation):
                        folder = self.root / kind / module.TASK / adaptation
                        root, path, out = folder/'data', folder/'cfg.json', folder/'out'
                        fixture(root)
                        if module is nyuv2:
                            savemat(root/'labeled/splits.mat', {'trainNdxs': [[1],[2],[3]], 'testNdxs': [[4],[5],[6]]})
                        cfg = (factory(root) if module is imagenet else
                               TestOptimization().config(module, factory, root, adaptation))
                        cfg['backbone'] = spec
                        if adaptation == 'finetune':
                            cfg['optimizer_profile'] = 'basic5_finetune_v1'
                        settings = cfg['detector' if module is coco else 'probe']
                        settings.update(epochs=1, max_val_samples=2)
                        path.write_text(json.dumps(cfg))
                        self.assertEqual(module.main(['--config', str(path), '--out', str(out)]), 0)
                        self.assertTrue(contract.verify(out, path, 0)[0])
                        result = json.loads((out/'results.json').read_text())
                        self.assertFalse(result['canonical_eligible'])
                        self.assertFalse(result['record_value'])
