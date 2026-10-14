"""Frozen native DINO ablation readouts, distinct from released HF models."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch required")
class TestAblationEvaluation(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def provider(self):
        from downstream import spatial_backbones as sb
        providers = sb.discover_providers()
        self.assertIn("dinov3_ablation", providers, "native ablation evaluation provider missing")
        return sb._load_provider(providers["dinov3_ablation"])

    def fixture(self):
        path = Path(self.provider().__file__).parent / "models/vision_transformer.py"
        spec = importlib.util.spec_from_file_location("test_ablation_vit", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.VisionTransformer(img_size=224, patch_size=16, embed_dim=16,
            depth=2, num_heads=2, n_register_tokens=4).eval()
        path = self.root / "encoder.pt"
        torch.save(self.model.state_dict(), path)
        return {"kind": "dinov3_ablation", "encoder": str(path), "arch": "vit_base_patch16",
                "img_size": 224, "patch_size": 16}

    def build(self, spec):
        from downstream import spatial_backbones as sb
        with mock.patch.object(self.provider(), "build_encoder", return_value=copy.deepcopy(self.model)):
            return sb.build_frozen_backbone(spec, torch.device("cpu"))

    def test_discovery_separates_multiple_providers_and_ignores_decoys(self):
        from downstream import spatial_backbones as sb
        folder = self.root / "method"
        folder.mkdir()
        for name, kind in (("downstream_backbone.py", "primary"),
                           ("downstream_backbone_extra.py", "secondary"),
                           ("test_downstream_backbone_extra.py", "decoy")):
            (folder / name).write_text(f'KIND = "{kind}"\ndef build(spec): return "{kind}"\n')
        found = sb.discover_providers(self.root)
        self.assertEqual(set(found), {"primary", "secondary"})
        self.assertEqual(sb._load_provider(found["primary"]).build({}), "primary")
        self.assertEqual(sb._load_provider(found["secondary"]).build({}), "secondary")
        (folder / "downstream_backbone_duplicate.py").write_text('KIND = "primary"\ndef build(spec): pass\n')
        with self.assertRaises(RuntimeError):
            sb.discover_providers(self.root)

    def test_image_spatial_video_and_head_updates_preserve_native_teacher(self):
        from downstream.imagenet import ImageClassifier
        from downstream.ssv2 import FrozenFrameAverageClassifier
        spec = self.fixture()
        bb = self.build(spec)
        with mock.patch.object(self.provider(), "build_encoder", return_value=copy.deepcopy(self.model)):
            direct = self.provider().build(spec)
        self.assertTrue(all(not p.requires_grad for p in direct.parameters()))
        self.assertEqual(bb.detection_normalization(), ((.485,.456,.406),(.229,.224,.225)))
        before = copy.deepcopy(bb.state_dict())
        for hw in ((32, 48), (224, 224)):
            x = torch.randn(2, 3, *hw)
            with torch.no_grad():
                cls, patches = self.model(x, is_global=True)
            torch.testing.assert_close(bb.classification_features(x, adaptation="frozen"), F.normalize(cls, dim=-1))
            expected = patches.transpose(1, 2).reshape(2, 16, hw[0]//16, hw[1]//16)
            torch.testing.assert_close(bb.forward_features(x), expected)
            torch.testing.assert_close(bb.forward_detection_features(x), expected)
        for video, classes in ((False, 1000), (True, 174)):
            torch.manual_seed(5)
            head = nn.Linear(16, classes)
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)
            torch.manual_seed(5)
            actual = FrozenFrameAverageClassifier(bb) if video else ImageClassifier(bb, "frozen")
            opts = [torch.optim.SGD(m.parameters(), lr=.1, momentum=.9) for m in (head, actual)]
            for _ in range(2):
                x = torch.randn(2, 3, 3, 32, 48) if video else torch.randn(2, 3, 32, 48)
                with torch.no_grad():
                    cls, patches = self.model(x.flatten(0, 1) if video else x, is_global=True)
                features = patches.mean(1).reshape(2, 3, 16).mean(1) if video else F.normalize(cls, dim=-1)
                a, e = actual(x), head(features)
                torch.testing.assert_close(a, e)
                for opt in opts: opt.zero_grad()
                F.cross_entropy(a, torch.tensor([0, 1])).backward()
                F.cross_entropy(e, torch.tensor([0, 1])).backward()
                torch.testing.assert_close(actual.classifier.weight.grad, head.weight.grad)
                for opt in opts: opt.step()
                torch.testing.assert_close(actual.classifier.weight, head.weight)
        bb.train()
        self.assertFalse(bb.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in bb.parameters()))
        for k,v in before.items(): torch.testing.assert_close(bb.state_dict()[k], v, rtol=0, atol=0)

    def test_invalid_geometry_weights_and_adaptations_are_refused(self):
        from downstream import spatial_backbones as sb
        spec = self.fixture()
        bb = self.build(spec)
        for changes in ({"encoder": ""}, {"arch": "other"}, {"img_size": 32},
                        {"patch_size": 14}, {"embed_dim": 16}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.build(dict(spec, **changes))
        for value in ({}, {"teacher_state_dict": self.model.state_dict()},
                      dict(self.model.state_dict(), unexpected=torch.ones(1))):
            torch.save(value, spec["encoder"])
            with self.assertRaises((ValueError, RuntimeError)): self.build(spec)
        for x in (torch.rand(0,3,32,32),torch.rand(2,1,32,32),torch.rand(2,3,31,32)):
            with self.assertRaises(ValueError): bb.forward_features(x)
        for adaptation in ("attentive", "finetune"):
            self.assertFalse(sb.supports_adaptation(spec["kind"], adaptation))
            with self.assertRaises(ValueError): bb.classification_features(torch.rand(1,3,32,32), adaptation=adaptation)
        for x in (torch.rand(2,0,3,32,32), torch.rand(2,3,32,32)):
            with self.assertRaisesRegex(ValueError, "teacher video requires nonempty"):
                bb.classification_features(x, adaptation="frozen", video=True)

    def test_five_real_task_entries_verify_exports_and_noncanonical_results(self):
        from tests import test_basic5_imagenet as image
        from tests import test_basic5_optimization as opt
        if not (opt.HAVE and image.HAVE):
            self.skipTest("complete downstream dependencies required")
        from scipy.io import savemat

        from downstream import contract, imagenet
        spec = self.fixture()
        helper = opt.TestOptimization()
        cases = list(helper.cases()) + [(imagenet, image.TestImageNet().fixture, image.TestImageNet().config)]
        self.assertEqual(len(cases), 5)
        for module, fixture, factory in cases:
            with self.subTest(task=module.TASK):
                root = self.root / module.TASK / "data"
                fixture(root)
                if module is opt.nyuv2:
                    savemat(root / "labeled/splits.mat", {"trainNdxs": [[1], [2], [3]], "testNdxs": [[4], [5], [6]]})
                cfg = factory(root) if module is imagenet else helper.config(module, factory, root, "frozen")
                cfg["backbone"] = spec
                if module is opt.coco:
                    cfg["detector_profile"] = "captured_native_detection_v1"
                if module is imagenet:
                    cfg["preprocessing_profile"] = "captured_provider_v1"
                if module is opt.ssv2:
                    cfg["video_recipe"] = "captured_provider_video_v1"
                cfg["detector" if module is opt.coco else "probe"].update(epochs=1, max_val_samples=2)
                path, out = root.parent / "config.json", root.parent / "out"
                path.write_text(json.dumps(cfg))
                with mock.patch.object(self.provider(), "build_encoder", return_value=copy.deepcopy(self.model)):
                    self.assertEqual(module.main(["--config", str(path), "--out", str(out)]), 0)
                self.assertTrue(contract.verify(out, path, 0)[0])
                result = json.loads((out / "results.json").read_text())
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])

    def test_example_and_ci_select_this_provider(self):
        from downstream import imagenet
        from tests._checkout import is_checkout
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed
        if not is_checkout(): self.skipTest("checkout required")
        path = Path(__file__).resolve().parents[1] / "docs/examples/dino_ablation_linear.json"
        self.assertTrue(path.is_file(), "native ablation example missing")
        cfg = json.loads(path.read_text())
        imagenet.validate_config(cfg)
        self.assertEqual(cfg["backbone"]["kind"], "dinov3_ablation")
        if not HAVE_YAML: self.skipTest("PyYAML required")
        cmd = next(s["run"] for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if s.get("name") == "Run Basic5 component contracts with downstream dependencies")
        self.assertTrue(_runs_finetune_tests(cmd, module="tests.test_dino_ablation_evaluation"))

    def test_native_presets_use_matching_model_geometry(self):
        provider = self.provider()
        for arch, dim, depth, heads in (("vit_base_patch16", 768, 12, 12),
                                        ("vit_large_patch16", 1024, 24, 16)):
            with torch.device("meta"):
                model = provider.build_encoder(arch)
            self.assertEqual((model.embed_dim, len(model.blocks), model.patch_size,
                              model.n_register_tokens), (dim, depth, 16, 4))
            self.assertEqual(model.blocks[0].attn.num_heads, heads)
        with self.assertRaises(ValueError): provider.build_encoder("unknown")
