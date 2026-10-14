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


class _UntrustedCheckpointObject:
    def __reduce__(self):
        return eval, ("'unexpected-deserialization'",)


@unittest.skipUnless(HAVE, "torch required")
class TestNativeTeacherImport(unittest.TestCase):
    root: Path
    model: "nn.Module"
    setUp = TestAblationEvaluation.setUp
    provider = TestAblationEvaluation.provider
    fixture = TestAblationEvaluation.fixture

    def native(self, arch="vit_base_patch16"):
        self.fixture()
        dim, depth, heads = self.provider().PRESETS[arch]
        return {"epoch": 300, "cfg": {"model": {"arch": arch, "embed_dim": dim,
            "depth": depth, "num_heads": heads, "n_register_tokens": 4,
            "use_rope": True, "rope_base": 100., "rope_rescale_coords": 2.},
            "data": {"train_path": "private-input-location"}},
            "teacher_state_dict": {**{"backbone."+k: v for k,v in self.model.state_dict().items()},
                                   "dino_head.weight": torch.randn(3, 3)},
            "student_state_dict": {"decoy": torch.ones(1)},
            "experiment_name": "private-experiment-identity", "optimizer_state_dict": {}}

    def convert(self, blob, **overrides):
        from downstream.contract import sha256_file
        source = self.root / "native.pth"
        torch.save(blob, source)
        args = {"source": source, "sha256": sha256_file(source), "arch": "vit_base_patch16",
                "epoch": 300, "out": self.root / "converted"}
        args.update(overrides)
        provider = self.provider()
        self.assertTrue(callable(getattr(provider, "import_native", None)), "native teacher importer missing")
        empty = copy.deepcopy(self.model)
        with torch.no_grad():
            for param in empty.parameters(): param.zero_()
        with mock.patch.object(provider, "build_encoder", return_value=empty):
            return provider.import_native(**args)

    def test_both_presets_export_only_teacher_and_preserve_readouts(self):
        from downstream.contract import sha256_file
        for arch in self.provider().PRESETS:
            with self.subTest(arch=arch):
                blob = self.native(arch)
                for head in ("ibot_head", "prototypes", "dino_prototypes", "ibot_prototypes",
                             "shared_mlp", "token_affine_cls", "token_affine_patch", "gram_head"):
                    blob["teacher_state_dict"][head] = torch.ones(1)
                    blob["teacher_state_dict"][head+".weight"] = torch.ones(1)
                out = self.root / arch
                record = self.convert(blob, arch=arch, out=out)
                state = torch.load(out/"encoder.pt", weights_only=True)
                self.assertEqual(set(state), set(self.model.state_dict()))
                for key, expected in self.model.state_dict().items():
                    torch.testing.assert_close(state[key], expected, rtol=0, atol=0)
                self.assertEqual({p.name for p in out.iterdir()}, {"encoder.pt", "export.json"})
                self.assertEqual(record, json.loads((out/"export.json").read_text()))
                self.assertEqual(set(record), {"source_sha256", "encoder_sha256", "arch", "epoch",
                                              "teacher_key", "tensor_count", "validation"})
                self.assertEqual(record["encoder_sha256"], sha256_file(out/"encoder.pt"))
                self.assertEqual(record["source_sha256"], sha256_file(self.root/"native.pth"))
                self.assertEqual((record["epoch"], record["arch"]), (300, arch))
                self.assertNotIn("private-", (out/"export.json").read_text())
                spec = {"kind": "dinov3_ablation", "arch": arch, "encoder": str(out/"encoder.pt"),
                        "img_size": 224, "patch_size": 16}
                with mock.patch.object(self.provider(), "build_encoder", return_value=copy.deepcopy(self.model)):
                    bb = self.provider().build(spec)
                images = torch.randn(2, 3, 32, 48)
                with torch.no_grad():
                    cls, patches = self.model(images, is_global=True)
                torch.testing.assert_close(bb.classification_features(images, adaptation="frozen"), F.normalize(cls, dim=-1))
                torch.testing.assert_close(bb.forward_features(images), patches.transpose(1,2).reshape(2,16,2,3))
                video = images[:, None].repeat(1,3,1,1,1)
                torch.testing.assert_close(bb.classification_features(video, adaptation="frozen", video=True), patches.mean(1))

    def test_identity_config_and_namespace_fail_closed(self):
        changes = [({"epoch": 299}, {}), ({"epoch": True}, {}), ({"epoch": "300"}, {}),
                   ({"epoch": 300.0}, {}), ({}, {"epoch": 300.0}),
                   ({}, {"sha256": "0"*64}), ({}, {"arch": "unknown"}), ({}, {"epoch": 0}),
                   ({}, {"epoch": True}), ({"cfg": {}}, {}), ({"cfg": None}, {}),
                   ({"teacher_state_dict": {}}, {}), ({"teacher_state_dict": []}, {}),
                   ({"teacher_state_dict": {7: torch.ones(1)}}, {})]
        for updates, kwargs in changes:
            with self.subTest(updates=list(updates), kwargs=kwargs):
                blob = self.native(); blob.update(updates)
                with self.assertRaises(ValueError): self.convert(blob, **kwargs)
                self.assertFalse((self.root/"converted").exists())
        for key, value in {"arch": "vit_large_patch16", "embed_dim": 1024, "depth": 24,
                           "num_heads": 16, "patch_size": 14, "mlp_ratio": 3.,
                           "n_register_tokens": 0, "use_rope": False, "rope_base": 10.,
                           "rope_rescale_coords": None}.items():
            with self.subTest(key=key):
                blob = self.native(); blob["cfg"]["model"][key] = value
                with self.assertRaises(ValueError): self.convert(blob)
                self.assertFalse((self.root/"converted").exists())
        for decoy in ("dino_head_decoy.weight", "student.backbone.weight", "module.backbone.weight"):
            blob = self.native(); blob["teacher_state_dict"][decoy] = torch.ones(1)
            with self.assertRaises(ValueError): self.convert(blob)
        blob = self.native()
        with mock.patch.object(torch, "load") as load:
            with self.assertRaises(ValueError): self.convert(blob, sha256="0"*64)
            load.assert_not_called()

    def test_bad_tensors_are_not_cast_or_partially_loaded(self):
        for defect in ("missing", "extra", "shape", "dtype", "nan", "inf", "non_tensor"):
            with self.subTest(defect=defect):
                blob = self.native(); state = blob["teacher_state_dict"]
                key = next(k for k in state if k.startswith("backbone."))
                if defect == "missing": del state[key]
                elif defect == "extra": state["backbone.decoy"] = torch.ones(1)
                elif defect == "shape": state[key] = torch.ones(1)
                elif defect == "dtype": state[key] = state[key].double()
                elif defect in ("nan", "inf"): state[key] = torch.full_like(state[key], float(defect))
                else: state[key] = "not-a-tensor"
                with self.assertRaises(ValueError): self.convert(blob)
                self.assertFalse((self.root/"converted").exists())

    def test_existing_output_source_change_and_unsafe_pickle_are_refused(self):
        import pickle
        blob = self.native()
        out = self.root/"occupied"; out.mkdir(); (out/"sentinel").write_text("preserve")
        with self.assertRaises(FileExistsError): self.convert(blob, out=out)
        self.assertEqual((out/"sentinel").read_text(), "preserve")
        provider = self.provider()
        self.assertTrue(callable(getattr(provider, "import_native", None)), "native teacher importer missing")
        from downstream.contract import sha256_file
        source = self.root/"native.pth"; torch.save(blob, source)
        digest = sha256_file(source)
        with (mock.patch.object(provider, "sha256_file", side_effect=[digest, "changed"]),
              self.assertRaisesRegex(ValueError, "changed")):
            self.convert(blob)
        self.assertFalse((self.root/"converted").exists())
        blob["untrusted"] = _UntrustedCheckpointObject()
        with self.assertRaises(pickle.UnpicklingError): self.convert(blob)
        self.assertFalse((self.root/"converted").exists())

    def test_cli_requires_explicit_identity_and_uses_importer(self):
        provider = self.provider()
        self.assertTrue(callable(getattr(provider, "main", None)), "native teacher CLI missing")
        args = ["--source", "input.pth", "--sha256", "a"*64, "--arch", "vit_base_patch16",
                "--epoch", "300", "--out", "new-output"]
        with mock.patch.object(provider, "import_native") as importer:
            self.assertEqual(provider.main(args), 0)
            importer.assert_called_once_with(source="input.pth", sha256="a"*64,
                arch="vit_base_patch16", epoch=300, out="new-output")
        with self.assertRaises(SystemExit): provider.main(args[:-2])
