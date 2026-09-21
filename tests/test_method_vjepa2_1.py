"""Image-branch and native-video downstream contracts for the pinned encoder."""
import copy
import unittest
from tests._checkout import needs_checkout

try:
    import torch
    import einops
    import timm
    from downstream import spatial_backbones as sb
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch required")
class TestBackbone(unittest.TestCase):
    def spec(self):
        return dict(kind="vjepa2_1", arch="vit_large", encoder="", img_size=384,
                    patch_size=16, embed_dim=48, depth=12, num_heads=4)

    def model(self, trainable=False):
        builder = sb.build_trainable_backbone if trainable else sb.build_frozen_backbone
        return builder(self.spec(), torch.device("cpu"))

    def test_image_is_true_single_frame_and_grid_is_padded_without_resize(self):
        model = self.model()
        x = torch.randn(2, 3, 19, 29)
        seen = []
        hook = model.encoder.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
        actual = model.forward_features(x)
        hook.remove()
        expected_input = torch.nn.functional.pad(x, (0, 3, 0, 13)).unsqueeze(2)
        torch.testing.assert_close(seen[0], expected_input, rtol=0, atol=0)
        expected = model.encoder(expected_input).float().transpose(1, 2).reshape(2, 48, 2, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(actual.requires_grad)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_trainable_image_path_preserves_gradients_and_updates(self):
        model = self.model(True)
        model.train()
        before = model.encoder.patch_embed_img.proj.weight.detach().clone()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model.forward_features(x)
        (output * torch.randn_like(output)).mean().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        opt.step()
        self.assertFalse(torch.equal(before, model.encoder.patch_embed_img.proj.weight))
        with torch.no_grad():
            self.assertFalse(model.forward_features(x).requires_grad)

    def test_video_tokens_use_native_temporal_encoder_without_frame_averaging(self):
        model = self.model()
        clip = torch.randn(2, 4, 3, 19, 29)
        actual = model.video_tokens(clip)
        padded = torch.nn.functional.pad(clip, (0, 3, 0, 13)).permute(0, 2, 1, 3, 4).contiguous()
        expected = model.encoder(padded).float()
        self.assertEqual(actual.shape, (2, 8, 48))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for frames in (1, 3, 0):
            with self.assertRaisesRegex(ValueError, "frames"):
                model.video_tokens(torch.randn(1, frames, 3, 32, 32))

    def test_finetune_config_accepts_only_explicit_trainable_providers(self):
        from downstream.attention import validate_adaptation
        cfg = dict(profile="capture_basic5_components", adaptation="finetune", backbone=self.spec())
        self.assertEqual(validate_adaptation(cfg), "finetune")
        for kind in ("unknown", "mae_vit"):
            cfg["backbone"]["kind"] = kind
            with self.assertRaises(ValueError):
                validate_adaptation(cfg)

    def test_new_provider_cannot_enter_a_legacy_table_producing_profile(self):
        from downstream.attention import validate_adaptation
        for adaptation in ("frozen", "attentive", "finetune"):
            for profile in (None, "legacy"):
                with self.assertRaisesRegex(ValueError, "capture_basic5_components"):
                    validate_adaptation(dict(backbone=self.spec(), adaptation=adaptation, profile=profile))

    def test_checkpoint_key_selection_is_strict_and_variant_specific(self):
        import tempfile
        from pathlib import Path
        model = self.model()
        state = {"module.backbone." + k: v for k, v in model.encoder.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "weights.pt"
            for arch, key in (("vit_large", "ema_encoder"), ("vit_giant_xformers", "target_encoder")):
                spec = self.spec() | dict(encoder=str(path), arch=arch)
                torch.save({key: state}, path)
                loaded = sb.build_frozen_backbone(spec, torch.device("cpu"))
                for name, value in model.encoder.state_dict().items():
                    torch.testing.assert_close(value, loaded.encoder.state_dict()[name], rtol=0, atol=0)
                torch.save({"encoder": state}, path)
                with self.assertRaisesRegex(ValueError, key):
                    sb.build_frozen_backbone(spec, torch.device("cpu"))
                bad = dict(state)
                bad.pop(next(iter(bad)))
                torch.save({key: bad}, path)
                with self.assertRaises(RuntimeError):
                    sb.build_frozen_backbone(spec, torch.device("cpu"))

    def test_ssv2_composition_preserves_native_tokens_and_protocol_readout(self):
        from downstream.ssv2 import FrozenFrameAverageClassifier
        for adaptation in ("frozen", "attentive", "finetune"):
            backbone = self.model(adaptation == "finetune")
            model = FrozenFrameAverageClassifier(backbone, 3, adaptation=adaptation)
            self.assertEqual(torch.count_nonzero(model.classifier.weight), 0)
            clip = torch.randn(2, 4, 3, 32, 32)
            tokens = backbone.video_tokens(clip)
            expected = (tokens if adaptation == "attentive" else
                        torch.nn.functional.normalize(tokens.float().mean(1), dim=-1))
            seen = []
            target = model.reader if adaptation == "attentive" else model.classifier
            hook = target.register_forward_pre_hook(lambda m, a: seen.append(a[0]))
            model(clip)
            hook.remove()
            torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
            self.assertEqual(seen[0].requires_grad, adaptation == "finetune")

    @unittest.skipUnless(__import__('importlib').util.find_spec('scipy')
                         and __import__('importlib').util.find_spec('h5py')
                         and __import__('importlib').util.find_spec('av')
                         and __import__('importlib').util.find_spec('pycocotools'),
                         "Full downstream dependencies required")
    def test_all_four_real_task_clis_accept_frozen_attentive_and_finetune(self):
        import json
        import tempfile
        from pathlib import Path
        from scipy.io import savemat
        from downstream import ade20k, nyuv2, coco, ssv2
        from tests.test_downstream_ade20k import tiny_ade, smoke_config as ade_config
        from tests.test_downstream_nyuv2 import tiny_nyuv2, smoke_config as nyu_config
        from tests.test_downstream_coco import tiny_coco, smoke_config as coco_config
        from tests.test_downstream_ssv2 import tiny_ssv2, smoke_config as video_config
        for module, fixture, config in ((ade20k, tiny_ade, ade_config),
                                        (nyuv2, tiny_nyuv2, nyu_config),
                                        (coco, tiny_coco, coco_config),
                                        (ssv2, tiny_ssv2, video_config)):
            for adaptation in ("frozen", "attentive", "finetune"):
                with self.subTest(task=module.TASK, adaptation=adaptation), tempfile.TemporaryDirectory() as d:
                    root = Path(d) / "data"
                    fixture(root)
                    if module is nyuv2:
                        savemat(root / "labeled/splits.mat", {"trainNdxs": [[1], [2], [3]], "testNdxs": [[4], [5], [6]]})
                    cfg = config(root)
                    cfg.update(backbone=self.spec(), profile="capture_basic5_components", adaptation=adaptation)
                    if module is coco:
                        cfg["detector"]["anchor_sizes"] = [8, 16, 32, 64]
                    path = Path(d) / "config.json"
                    path.write_text(json.dumps(cfg))
                    out = Path(d) / "out"
                    self.assertEqual(module.main(["--config", str(path), "--out", str(out)]), 0)
                    result = json.loads((out / "run_manifest.json").read_text())
                    self.assertEqual(result["status"], "ok")
                    from downstream import contract
                    self.assertTrue(contract.verify(out, path, 0)[0])

    def test_invalid_architecture_geometry_and_fixture_refused(self):
        for update in ({"arch": "unknown"}, {"patch_size": 8}, {"img_size": 224}):
            with self.assertRaises(ValueError):
                sb.build_frozen_backbone(self.spec() | update, torch.device("cpu"))
        partial = self.spec()
        partial.pop("num_heads")
        with self.assertRaises(ValueError):
            sb.build_frozen_backbone(partial, torch.device("cpu"))

    def test_image_shape_and_token_grid_refuse_invalid_inputs(self):
        from unittest import mock
        model = self.model()
        for shape in ((1, 2, 32, 32), (1, 3, 0, 32), (1, 3, 1, 32, 32)):
            with self.assertRaises(ValueError):
                model.forward_features(torch.randn(*shape))
        with mock.patch.object(model.encoder, "forward", return_value=torch.zeros(1, 5, 48)):
            with self.assertRaisesRegex(RuntimeError, "patch grid"):
                model.forward_features(torch.zeros(1, 3, 32, 32))


class TestInfrastructure(unittest.TestCase):
    @needs_checkout
    @unittest.skipUnless(HAVE, "author encoder dependencies required")
    def test_method_smoke_runs_without_checkout_or_workflows(self):
        from tests.test_repository_scan import without_git
        result = without_git(
            "import pathlib, tempfile, unittest, shutil\n"
            "from tests import test_ci\n"
            "assert shutil.which('git') is None\n"
            "with tempfile.TemporaryDirectory() as directory:\n"
            " test_ci.WORKFLOWS = pathlib.Path(directory) / 'absent'\n"
            " assert not test_ci.WORKFLOWS.exists()\n"
            " suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_method_vjepa2_1')\n"
            " result = unittest.TextTestRunner(verbosity=2).run(suite)\n"
            " assert result.testsRun - len(result.skipped) >= 9\n"
            " raise SystemExit(0 if result.wasSuccessful() else 1)\n")
        self.assertEqual(result.returncode, 0, result.stderr[-5000:])

    def test_author_namespaces_are_isolated_and_decoys_survive(self):
        import sys
        import types
        from pathlib import Path
        from unittest import mock
        from provider_support import prepare_upstream
        root = Path('/tmp/pinned-fixture/author').resolve()
        foreign = str(root.parent / 'foreign')
        decoy = str(root.parent) + '-decoy/keep'
        namespace = '_author_namespace'
        modules = {namespace: types.ModuleType(namespace),
                   namespace + '.child': types.ModuleType(namespace + '.child'),
                   namespace + '_decoy': types.ModuleType(namespace + '_decoy')}
        with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, 'path',
                [foreign, decoy]):
            prepare_upstream(root, (namespace,))
            self.assertNotIn(namespace, sys.modules)
            self.assertNotIn(namespace + '.child', sys.modules)
            self.assertIn(namespace + '_decoy', sys.modules)
            self.assertEqual(sys.path, [str(root), decoy])

    @needs_checkout
    def test_ci_explicitly_runs_this_contract_with_author_dependencies(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML:
            self.skipTest('PyYAML required')
        steps = parsed()['tests.yml']['jobs']['downstream']['steps']
        commands = [s['run'] for s in steps if s.get('name') == 'Run Basic5 component contracts with downstream dependencies']
        self.assertEqual(len(commands), 1)
        self.assertTrue(_runs_finetune_tests(commands[0], module='tests.test_method_vjepa2_1'))
        from tests.test_method_requirements import declared_packages
        from pathlib import Path
        self.assertIn('einops', declared_packages(Path(__file__).resolve().parents[1] / 'downstream/requirements.lock.txt'))
