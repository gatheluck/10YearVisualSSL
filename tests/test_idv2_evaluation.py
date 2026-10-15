"""IDv2 scaling exports enter frozen BasicFive without changing their readout."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._checkout import needs_checkout

try:
    import torch
    from timm.models.vision_transformer import VisionTransformer
    from torch import nn
    from torch.nn import functional as F
    HAVE = True
except ImportError:
    HAVE = False


class TestCI(unittest.TestCase):
    def test_downstream_job_runs_the_evaluation_contract(self):
        from tests._checkout import is_checkout
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed
        if not HAVE_YAML or not is_checkout():
            self.skipTest("checkout and PyYAML required")
        command = next(s["run"] for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
                       if s.get("name") == "Run Basic5 component contracts with downstream dependencies")
        self.assertTrue(_runs_finetune_tests(command, module="tests.test_idv2_evaluation"))


@unittest.skipUnless(HAVE, "torch and timm required")
class TestScalingEvaluation(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, arch="vit_base_patch16_224"):
        self.reference = VisionTransformer(img_size=224, patch_size=16,
            embed_dim=16, depth=2, num_heads=2, num_classes=0, global_pool="",
            norm_layer=lambda dim, **kw: nn.LayerNorm(dim, eps=1e-6, **kw)).eval()
        self.export: dict = {"scaling_encoder_version": 1, "model": arch, "readout": "cls",
                           "state_dict": self.reference.state_dict()}
        path = self.root / "encoder.pt"
        torch.save(self.export, path)
        return {"kind": "idv2_scaling", "arch": arch, "encoder": str(path),
                    "img_size": 224, "patch_size": 16}

    def tiny_factory(self, arch, **kwargs):
        self.assertIn(arch, ("vit_base_patch16_224", "vit_large_patch16_224"))
        self.assertFalse(kwargs["pretrained"])
        self.assertEqual(kwargs["num_classes"], 0)
        self.assertEqual(kwargs["global_pool"], "")
        return VisionTransformer(img_size=224, patch_size=16, embed_dim=16,
            depth=2, num_heads=2, num_classes=0, global_pool="",
            dynamic_img_size=kwargs["dynamic_img_size"],
            norm_layer=lambda dim, **kw: nn.LayerNorm(dim, eps=1e-6, **kw))

    @needs_checkout
    def test_documented_image_configuration_validates(self):
        from downstream import imagenet
        path = Path(__file__).resolve().parents[1] / "docs/examples/idv2_scaling_linear.json"
        self.assertTrue(path.is_file(), "a validated scaling evaluation example is required")
        cfg = json.loads(path.read_text())
        imagenet.validate_config(cfg)
        self.assertEqual(cfg["backbone"]["kind"], "idv2_scaling")
        self.assertEqual(cfg["preprocessing_profile"], "captured_provider_v1")

    def build(self, spec):
        from downstream import spatial_backbones as sb
        self.assertIn("idv2_scaling", sb.discover_providers(),
                      "scaling exports lack their own downstream provider")
        with mock.patch("timm.create_model", side_effect=self.tiny_factory):
            return sb.build_frozen_backbone(spec, torch.device("cpu"))

    def expected_tokens(self, pixels):
        # A static preset with explicitly resized positions is an independent
        # oracle for the dynamic encoder, including rectangular grids.
        model = copy.deepcopy(self.reference)
        if pixels.shape[-2:] != (224, 224):
            from timm.layers import resample_abs_pos_embed
            grid = tuple(s // 16 for s in pixels.shape[-2:])
            model.pos_embed = nn.Parameter(resample_abs_pos_embed(model.pos_embed,
                grid, old_size=(14, 14), num_prefix_tokens=1,
                interpolation="bicubic", antialias=True))
            model.patch_embed.img_size = tuple(pixels.shape[-2:])
            model.patch_embed.grid_size = grid
        with torch.no_grad():
            return model.forward_features(pixels)

    def native_fixture(self):
        import hashlib

        import numpy as np

        from downstream import spatial_backbones as sb

        spec = self.fixture()
        module = sb._load_provider(sb.discover_providers()["idv2_scaling"]).encoder_module()
        source = self.root / "native.pth"
        blob = {"completed_epoch": 30, "step_in_epoch": 0,
                "state_dict": {"module.vit." + k: v for k, v in self.reference.state_dict().items()},
                "rng_state": {"numpy": np.random.get_state()},
                "config": {"private_path": "/private/original"},
                "memory": torch.randn(3, 8), "Z": torch.tensor(7.)}
        blob["state_dict"]["module.fc.weight"] = torch.randn(8, 16)
        torch.save(blob, source)
        args = {"source": source, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "arch": spec["arch"], "completed_epoch": 30, "state_key": "state_dict",
                "prefix": "module.vit.", "out": self.root / "converted"}
        return module, blob, args, spec

    def test_native_import_cli_without_repository_on_pythonpath(self):
        import os
        import subprocess
        import sys

        from downstream import spatial_backbones as sb

        _, _, args, _ = self.native_fixture()
        script = sb.discover_providers()["idv2_scaling"].parent / "models/scaling_encoder.py"
        command = [str(script), "--source", str(args["source"]), "--sha256", args["sha256"],
                   "--arch", args["arch"], "--completed-epoch", "30", "--state-key", "state_dict",
                   "--prefix", "module.vit.", "--out", str(args["out"])]
        # Faithful direct-script search path, tiny real encoder to keep this bounded.
        code = """import importlib.util, runpy, sys
from unittest import mock
import timm
from timm.models.vision_transformer import VisionTransformer
from torch import nn
sys.path[0] = sys.argv[1]
assert importlib.util.find_spec('downstream') is None, 'repository unexpectedly importable'
def factory(*args, **kwargs):
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=16, depth=2,
        num_heads=2, num_classes=0, global_pool='', dynamic_img_size=True,
        norm_layer=lambda dim, **kw: nn.LayerNorm(dim, eps=1e-6, **kw))
sys.argv = sys.argv[2:]
with mock.patch.object(timm, 'create_model', side_effect=factory):
    runpy.run_path(sys.argv[0], run_name='__main__')
"""
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-c", code, str(script.parent), *command],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((args["out"] / "encoder.pt").is_file())

    def test_native_scaling_import_numpy_namespaces_with_callable_only_allowlist(self):
        import hashlib
        import io
        import zipfile

        import torch._weights_only_unpickler as unpickler

        # PyTorch 2.5 resolves each allowlisted callable by module/name. Exercise
        # real restricted deserialization, including NumPy 1.x archive names.
        def callable_globals():
            result = {}
            for item in torch.serialization.get_safe_globals():
                if isinstance(item, tuple):
                    raise TypeError("PyTorch 2.5 requires callable globals")
                name = getattr(item, "__name__", None)
                if name is None:
                    raise TypeError("PyTorch 2.5 requires named globals")
                result[f"{item.__module__}.{name}"] = item
            return result

        module, _blob, args, _spec = self.native_fixture()
        original = args["source"].read_bytes()
        for namespace in ("numpy.core.multiarray", "numpy._core.multiarray"):
            with self.subTest(namespace=namespace):
                with (zipfile.ZipFile(io.BytesIO(original)) as src,
                      zipfile.ZipFile(args["source"], "w") as dst):
                    for member in src.infolist():
                        data = src.read(member)
                        if member.filename.endswith("/data.pkl"):
                            modern = b"numpy._core.multiarray\n_reconstruct\n"
                            legacy = b"numpy.core.multiarray\n_reconstruct\n"
                            self.assertEqual(data.count(modern) + data.count(legacy), 1)
                            data = data.replace(modern, legacy).replace(
                                legacy, namespace.encode() + b"\n_reconstruct\n")
                        dst.writestr(member, data)
                options = dict(args, out=self.root / namespace,
                               sha256=hashlib.sha256(args["source"].read_bytes()).hexdigest())
                with (mock.patch.object(unpickler,
                                        "_get_user_allowed_globals", callable_globals),
                      mock.patch("timm.create_model", side_effect=self.tiny_factory)):
                    module.import_native(**options)
                export = module.read_export(self.root / namespace / "encoder.pt")
                for key, value in export["state_dict"].items():
                    torch.testing.assert_close(value, self.reference.state_dict()[key], rtol=0, atol=0)

    def test_native_scaling_import_preserves_readouts_and_excludes_private_payload(self):
        import hashlib
        module, _blob, args, spec = self.native_fixture()
        before = args["source"].read_bytes()
        with mock.patch("timm.create_model", side_effect=self.tiny_factory):
            record = module.import_native(**args)
        self.assertEqual(args["source"].read_bytes(), before)
        export = torch.load(args["out"] / "encoder.pt", weights_only=True)
        self.assertEqual(set(export), {"scaling_encoder_version", "model", "readout", "state_dict"})
        self.assertEqual(set(export["state_dict"]), set(self.reference.state_dict()))
        for key, value in export["state_dict"].items():
            torch.testing.assert_close(value, self.reference.state_dict()[key], atol=0, rtol=0)
        self.assertEqual(record["source_sha256"], args["sha256"])
        self.assertEqual(record["completed_epoch"], 30)
        self.assertEqual(record["encoder_sha256"], hashlib.sha256((args["out"] / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(json.loads((args["out"] / "export.json").read_text()), record)
        self.assertNotIn("/private/original", json.dumps(record))
        self.assertNotIn(str(args["source"]), json.dumps(record))
        spec["encoder"] = str(args["out"] / "encoder.pt")
        bb = self.build(spec)
        x = torch.randn(2, 3, 32, 48)
        torch.testing.assert_close(bb.classification_features(x, adaptation="frozen"),
            F.normalize(self.expected_tokens(x)[:, 0], dim=-1), atol=1e-6, rtol=1e-6)
        original = (args["out"] / "encoder.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            module.import_native(**args)
        self.assertEqual((args["out"] / "encoder.pt").read_bytes(), original)

    def test_native_scaling_import_rejects_identity_tensor_and_epoch_errors(self):
        import hashlib
        from contextlib import nullcontext
        module, blob, args, _spec = self.native_fixture()
        for case in ("hash", "epoch", "bool_epoch", "float_epoch", "float_step", "partial", "missing", "extra", "shape", "nan", "dtype", "prefix", "arch", "mapping", "tensor_type", "state_key", "requested_epoch", "requested_bool"):
            bad = copy.deepcopy(blob)
            options = dict(args)
            key = "module.vit.cls_token"
            if case == "epoch": bad["completed_epoch"] = 29
            if case == "bool_epoch": bad["completed_epoch"] = True
            if case == "float_epoch": bad["completed_epoch"] = 30.0
            if case == "float_step": bad["step_in_epoch"] = 0.0
            if case == "partial": bad["step_in_epoch"] = 1
            if case == "missing": del bad["state_dict"][key]
            if case == "extra": bad["state_dict"]["module.vit.decoy"] = torch.zeros(1)
            if case == "shape": bad["state_dict"][key] = torch.zeros(3)
            if case == "nan": bad["state_dict"][key].fill_(float("nan"))
            if case == "dtype": bad["state_dict"][key] = bad["state_dict"][key].double()
            if case == "prefix": options["prefix"] = "module."
            if case == "arch": options["arch"] = "unverified"
            if case == "mapping": bad["state_dict"] = []
            if case == "tensor_type": bad["state_dict"][key] = "not a tensor"
            if case == "state_key": options["state_key"] = "config"
            if case == "requested_epoch": options["completed_epoch"] = 0
            if case == "requested_bool": options["completed_epoch"] = True
            torch.save(bad, args["source"])
            options["sha256"] = "0" * 64 if case == "hash" else hashlib.sha256(args["source"].read_bytes()).hexdigest()
            loading = mock.patch.object(module.torch, "load", side_effect=AssertionError("unverified bytes deserialized")) if case == "hash" else nullcontext()
            with self.subTest(case=case), loading, mock.patch("timm.create_model", side_effect=self.tiny_factory):
                with self.assertRaises(ValueError):
                    module.import_native(**options)
                self.assertFalse(args["out"].exists())

    def test_native_scaling_import_cli_layouts_and_safe_loading(self):
        import hashlib
        import pickle
        module, blob, args, _spec = self.native_fixture()
        for index, prefix in enumerate(("vit.", "module.vit.", "encoder.vit.", "module.encoder.vit.")):
            native = dict(blob, model={prefix + k: v for k, v in self.reference.state_dict().items()})
            del native["state_dict"]
            torch.save(native, args["source"])
            options = dict(args, arch=module.PRESETS[index % 2], state_key="model", prefix=prefix,
                           out=self.root / str(index), sha256=hashlib.sha256(args["source"].read_bytes()).hexdigest())
            argv = [item for key, value in options.items() for item in ("--" + key.replace("_", "-"), str(value))]
            with mock.patch("timm.create_model", side_effect=self.tiny_factory):
                self.assertEqual(module.main(argv), 0)
            export = module.read_export(self.root / str(index) / "encoder.pt")
            self.assertEqual(export["model"], options["arch"])
            for key, value in export["state_dict"].items():
                torch.testing.assert_close(value, self.reference.state_dict()[key], rtol=0, atol=0)
        bad = dict(blob, unapproved_object=Path("unapproved"))
        torch.save(bad, args["source"])
        args["sha256"] = hashlib.sha256(args["source"].read_bytes()).hexdigest()
        with self.assertRaises(pickle.UnpicklingError):
            module.import_native(**args)
        self.assertFalse(args["out"].exists())
        torch.save(blob, args["source"])
        args["sha256"] = hashlib.sha256(args["source"].read_bytes()).hexdigest()
        with (mock.patch.object(module, "_sha256", side_effect=[args["sha256"], "changed"]),
              mock.patch("timm.create_model", side_effect=self.tiny_factory),
              self.assertRaisesRegex(ValueError, "source changed")):
            module.import_native(**args)
        self.assertFalse(args["out"].exists())

    def test_cls_patches_rectangles_and_video_use_exact_export_weights(self):
        for arch in ("vit_base_patch16_224", "vit_large_patch16_224"):
            spec = self.fixture(arch)
            bb = self.build(spec)
            before = copy.deepcopy(bb.state_dict())
            for hw in ((224, 224), (32, 48)):
                x = torch.randn(2, 3, *hw, requires_grad=True)
                tokens = self.expected_tokens(x)
                spatial = bb.forward_features(x)
                torch.testing.assert_close(spatial.flatten(2).transpose(1, 2), tokens[:, 1:], atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(bb.classification_features(x, adaptation="frozen"), F.normalize(tokens[:, 0], dim=-1), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(bb.forward_detection_features(x), spatial, atol=0, rtol=0)
                self.assertFalse(spatial.requires_grad)
            clip = torch.randn(2, 3, 3, 32, 48)
            expected = self.expected_tokens(clip.flatten(0, 1))[:, 0].reshape(2, 3, 16).mean(1)
            torch.testing.assert_close(bb.classification_features(clip, adaptation="frozen", video=True), expected, atol=1e-6, rtol=1e-6)
            self.assertFalse(torch.allclose(expected.norm(dim=-1), torch.ones(2)))
            bb.train()
            self.assertFalse(bb.training)
            self.assertFalse(any(p.requires_grad for p in bb.parameters()))
            for name, value in bb.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            self.assertEqual(bb.detection_normalization(), ((.485, .456, .406), (.229, .224, .225)))

    def test_identity_geometry_and_complete_state_are_required(self):
        spec = self.fixture()
        self.build(spec)
        for change in ({"encoder": ""}, {"encoder": str(self.root / "missing.pt")}, {"arch": "vit_large_patch16_224"},
                       {"img_size": 32}, {"patch_size": 14}, {"embed_dim": 16}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(dict(spec, **change))
        for change in ({"scaling_encoder_version": True}, {"scaling_encoder_version": 2}, {"readout": "patch_mean"},
                       {"model": "unknown"}, {"state_dict": {}}, {"model": "vit_large_patch16_224"}):
            torch.save(dict(self.export, **change), spec["encoder"])
            with self.subTest(change=list(change)), self.assertRaises((ValueError, RuntimeError)):
                self.build(spec)
        for missing in (True, False):
            state = dict(self.export["state_dict"])
            if missing:
                state.pop("norm.weight")
            else:
                state["fc.weight"] = torch.ones(128, 16)
            torch.save(dict(self.export, state_dict=state), spec["encoder"])
            with self.assertRaises(RuntimeError):
                self.build(spec)

    def test_invalid_pixels_and_unverified_adaptations_fail_closed(self):
        from downstream import spatial_backbones as sb
        from downstream.attention import validate_adaptation
        spec = self.fixture()
        bb = self.build(spec)
        for x in (torch.rand(2, 3, 31, 48), torch.rand(2, 1, 32, 32), torch.rand(0, 3, 32, 32),
                  torch.rand(2, 3, 0, 16), torch.rand(3, 32, 32), torch.rand(2, 3, 16, 32, 32)):
            with self.assertRaisesRegex(ValueError, "scaling images require nonempty"):
                bb.forward_features(x)
        with self.assertRaisesRegex(ValueError, "scaling video requires nonempty"):
            bb.classification_features(torch.rand(2, 0, 3, 32, 32), adaptation="frozen", video=True)
        with self.assertRaisesRegex(ValueError, "scaling video requires nonempty"):
            bb.classification_features(torch.rand(2, 3, 32, 32), adaptation="frozen", video=True)
        for adaptation in ("attentive", "finetune"):
            self.assertFalse(sb.supports_adaptation("idv2_scaling", adaptation))
            with self.assertRaises(ValueError):
                validate_adaptation({"backbone": spec, "profile": "capture_basic5_components", "adaptation": adaptation})
            with self.assertRaises(ValueError):
                bb.classification_features(torch.rand(1, 3, 32, 32), adaptation=adaptation)
        with self.assertRaises(ValueError):
            validate_adaptation({"backbone": spec})
        provider = sb._load_provider(sb.discover_providers()["idv2_scaling"])
        with self.assertRaises(ValueError):
            provider.encoder_module().build_encoder("unknown")
        torch.save([], spec["encoder"])
        with self.assertRaises(ValueError):
            self.build(spec)

    def test_loader_identity_and_intrinsic_freezing_precede_shared_dispatch(self):
        from downstream import spatial_backbones as sb
        spec = self.fixture()
        self.build(spec)
        provider = sb._load_provider(sb.discover_providers()["idv2_scaling"])
        with mock.patch("timm.create_model", side_effect=self.tiny_factory):
            direct = provider.build(spec)
        self.assertFalse(any(p.requires_grad for p in direct.parameters()))
        with mock.patch("timm.create_model", return_value=nn.Identity()), self.assertRaisesRegex(TypeError, "VisionTransformer"):
            provider.encoder_module().build_encoder(spec["arch"])
        for data in ([], dict(self.export, model="unknown"),
                     dict(self.export, readout="other"), dict(self.export, scaling_encoder_version=2)):
            torch.save(data, spec["encoder"])
            with self.assertRaises(ValueError):
                provider.encoder_module().read_export(spec["encoder"])

    def test_source_head_initialization_and_two_updates_preserve_encoder(self):
        from downstream.imagenet import ImageClassifier
        from downstream.ssv2 import FrozenFrameAverageClassifier
        bb = self.build(self.fixture())
        for video, classes in ((False, 1000), (True, 174)):
            torch.manual_seed(9)
            expected_head = nn.Linear(16, classes)
            nn.init.normal_(expected_head.weight, std=.01)
            nn.init.zeros_(expected_head.bias)
            torch.manual_seed(9)
            model = FrozenFrameAverageClassifier(bb) if video else ImageClassifier(bb, "frozen")
            torch.testing.assert_close(model.classifier.weight, expected_head.weight, atol=0, rtol=0)
            actual_opt = torch.optim.SGD(model.classifier.parameters(), lr=.1, momentum=.9)
            expected_opt = torch.optim.SGD(expected_head.parameters(), lr=.1, momentum=.9)
            for _ in range(2):
                images = torch.randn(2, 2, 3, 32, 32) if video else torch.randn(2, 3, 32, 32)
                tokens = self.expected_tokens(images.flatten(0, 1) if video else images)[:, 0]
                features = tokens.reshape(2, 2, 16).mean(1) if video else F.normalize(tokens, dim=-1)
                actual_opt.zero_grad(); expected_opt.zero_grad()
                a, e = model(images), expected_head(features)
                torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-6)
                F.cross_entropy(a, torch.tensor([0, 1])).backward()
                F.cross_entropy(e, torch.tensor([0, 1])).backward()
                torch.testing.assert_close(model.classifier.weight.grad, expected_head.weight.grad, atol=1e-6, rtol=1e-6)
                actual_opt.step(); expected_opt.step()
                torch.testing.assert_close(model.classifier.weight, expected_head.weight, atol=1e-6, rtol=1e-6)
                self.assertTrue(all(p.grad is None for p in bb.parameters()))

    def test_detector_uses_native_transposed_pyramid_on_normalized_grid(self):
        from downstream.native_detection import (
            NativePyramidBackbone,
            TransposedFeaturePyramid,
        )
        bb = self.build(self.fixture())
        torch.manual_seed(27)
        expected = TransposedFeaturePyramid(16)
        torch.manual_seed(27)
        actual = NativePyramidBackbone(bb, adaptation="frozen")
        x = torch.randn(2, 3, 32, 48)
        tokens = self.expected_tokens(x)[:, 1:].transpose(1, 2).reshape(2, 16, 2, 3)
        a, e = actual(x), expected(tokens)
        self.assertEqual(list(a), ["0", "1", "2", "3"])
        for key in a:
            torch.testing.assert_close(a[key], e[key], atol=1e-6, rtol=1e-6)
        sum(v.square().mean() for v in a.values()).backward()
        sum(v.square().mean() for v in e.values()).backward()
        for (_, p), (_, q) in zip(actual.fpn.named_parameters(), expected.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-6)
        self.assertTrue(all(p.grad is None for p in bb.parameters()))

    def test_explicit_video_recipe_preserves_sampling_and_bicubic_inputs(self):
        import random

        from PIL import Image
        from torchvision import transforms as T

        from downstream import video_recipe
        spec = self.fixture()
        self.build(spec)
        cfg = {"task": "ssv2_video", "profile": "capture_basic5_components", "adaptation": "frozen",
                   "optimizer_profile": "basic5_frozen_v1", "backbone": spec,
                   "video_recipe": "captured_provider_video_v1",
                   "probe": {"num_frames": 3, "image_size": 32, "num_workers": 0,
                              "max_train_samples": 2, "max_val_samples": 2, "max_steps_per_epoch": 0, "epochs": 1}}
        recipe = video_recipe.resolve(cfg)
        random.seed(7)
        expected = [random.randint(a, b) for a, b in ((0, 2), (3, 6), (7, 9))]
        random.seed(7)
        self.assertEqual(video_recipe.sample_indices(10, 3, True, recipe), expected)
        self.assertEqual(video_recipe.sample_indices(10, 3, False, recipe), [1, 4, 8])
        pixels = torch.randint(0, 256, (43, 61, 3), dtype=torch.uint8).numpy()
        frames = [Image.fromarray(pixels), Image.fromarray(pixels[:, ::-1].copy())]
        transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(32), T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
        expected = torch.stack([transform(frame) for frame in frames])
        torch.testing.assert_close(video_recipe.transform(frames, False, 32, recipe), expected, atol=0, rtol=0)
        from downstream import video_execution
        cfg["probe"]["batch_size"] = 2
        policy = video_execution.resolve(cfg)
        self.assertEqual(policy["tail_policy"], "discard")
        self.assertEqual(policy["rank_seed_stride"], 1000)
        with mock.patch("downstream.extended_distributed.launch", return_value=(0, 0, 2)), self.assertRaisesRegex(ValueError, "distributed"):
            video_execution.resolve(cfg)

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
                with mock.patch("timm.create_model", side_effect=self.tiny_factory):
                    self.assertEqual(module.main(["--config", str(path), "--out", str(out)]), 0)
                self.assertTrue(contract.verify(out, path, 0)[0])
                result = json.loads((out / "results.json").read_text())
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])
