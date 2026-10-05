"""Explicit source FT profiles, not certification of historical table recipes."""

import copy
import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF

    from downstream import imagenet, optimization
    from tests import test_basic5_imagenet as image_helpers

    HAVE = True
except ImportError:
    HAVE = False

PROFILES = {
    "clip_hf": "captured_bicubic_random_python_v1",
    "siglip2_g": "captured_bicubic_random_python_v1",
    "cradiov4_h": "captured_bicubic_random_python_v1",
    "cosmos3_super_vm": "captured_bicubic_half_python_v1",
    "dinov3_hf": "captured_bilinear_plain_torch_v1",
    "raev2_k7": "captured_bilinear_raw_zero_torch_v1",
    "vjepa2_1": "captured_bilinear_normalized_zero_torch_v1",
    "vggt_omega": "captured_bicubic_unit_mixup_v1",
}


@unittest.skipUnless(HAVE, "downstream dependencies required")
class Profiles(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self, kind="vjepa2_1", root=Path("/unused")):
        cfg = image_helpers.TestImageNet().config(root)
        cfg.update(
            adaptation="finetune",
            optimizer_profile="basic5_finetune_v1",
            finetune_recipe=PROFILES[kind],
        )
        cfg["backbone"]["kind"] = kind
        return cfg

    def module(self):
        return importlib.import_module("downstream.imagenet_finetune")

    def test_explicit_source_recipes_enable_all_eight_classifiers(self):
        for kind in PROFILES:
            with self.subTest(kind=kind):
                cfg = self.config(kind)
                imagenet.validate_config(cfg)
                report = optimization.resolve_optimization(cfg, imagenet.TASK)
                self.assertEqual(report["optimizer"], "AdamW")
                self.assertEqual(report["reference_batch"], 1024)
                self.assertEqual(report["base_lr"], 0.0005)
                self.assertEqual(report["weight_decay"], 0.05)
                self.assertEqual(report["layer_decay"], 0.75)
                self.assertAlmostEqual(report["lr"], 0.0005 * 2 / 1024)

    def test_wrong_missing_and_frozen_recipe_selections_are_refused(self):
        cfg = self.config()
        bads = [
            cfg | {"finetune_recipe": "typo"},
            cfg | {"finetune_recipe": PROFILES["dinov3_hf"]},
            cfg | {"adaptation": "frozen", "optimizer_profile": "basic5_frozen_v1"},
            cfg | {"optimizer_profile": "basic5_frozen_v1"},
            cfg | {"backbone": dict(cfg["backbone"], kind="sam3_trunk")},
            cfg | {"backbone": dict(cfg["backbone"], kind="unknown")},
        ]
        absent = copy.deepcopy(cfg)
        del absent["finetune_recipe"]
        bads.append(absent)
        released = copy.deepcopy(cfg)
        released["backbone"]["arch"] = "released"
        bads.append(released)
        for bad in bads:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                imagenet.validate_config(bad)

    def test_reference_schedule_requires_1024_and_100_epochs(self):
        cfg = self.config()
        cfg["scheduler_profile"] = optimization.REFERENCE_SCHEDULE
        with self.assertRaises(ValueError):
            imagenet.validate_config(cfg)
        cfg["probe"].update(batch_size=1024, epochs=100)
        imagenet.validate_config(cfg)
        report = optimization.resolve_optimization(cfg, imagenet.TASK)
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=report["lr"])
        scheduler = optimization.build_task_scheduler(opt, report, 2)
        values = [opt.param_groups[0]["lr"]]
        for _ in range(200):
            opt.step()
            scheduler.step()
            values.append(opt.param_groups[0]["lr"])
        self.assertAlmostEqual(values[0], 1e-6)
        self.assertAlmostEqual(values[10], 0.0005)
        self.assertAlmostEqual(values[-1], 1e-6)

    def test_recipes_disclose_unsmoothed_targets_and_unverified_table_identity(self):
        for kind, profile in PROFILES.items():
            report = self.module().resolve(self.config(kind))
            self.assertEqual(report["profile"], profile)
            self.assertEqual(report["label_smoothing"], 0.0)
            self.assertEqual(report["documented_label_smoothing"], 0.1)
            self.assertTrue(report["protocol_conflict"])
            self.assertFalse(report["historical_run_verified"])
            self.assertEqual(report["gradient_clip_norm"], 1.0)

    def test_batch_targets_preserve_probabilities_and_cutmix_actual_area(self):
        m = self.module()
        x = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
        y = torch.tensor([0, 1])
        perm = mock.patch("torch.randperm", return_value=torch.tensor([1, 0]))
        recipe = m.resolve(self.config("siglip2_g"))
        with (
            perm,
            mock.patch("random.random", return_value=0.8),
            mock.patch("numpy.random.beta", return_value=0.5),
            mock.patch("random.randint", side_effect=[0, 0]),
        ):
            mixed, targets = m.mix(x.clone(), y, 3, recipe)
        self.assertEqual(mixed[0].sum().item(), 3.0)
        torch.testing.assert_close(
            targets,
            torch.tensor([[15 / 16, 1 / 16, 0], [1 / 16, 15 / 16, 0]]),
            rtol=0,
            atol=0,
        )
        for kind in PROFILES:
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            mixed, targets = m.mix(x.clone(), y, 3, m.resolve(self.config(kind)))
            self.assertEqual(mixed.shape, x.shape)
            self.assertTrue(torch.isfinite(mixed).all())
            torch.testing.assert_close(targets.sum(1), torch.ones(2))
            self.assertTrue(torch.equal(targets[:, 2], torch.zeros(2)))

    def test_torch_cutmix_and_mixup_only_have_distinct_sampling(self):
        m = self.module()
        x = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
        y = torch.tensor([0, 1])
        with (
            mock.patch("torch.randperm", return_value=torch.tensor([1, 0])),
            mock.patch(
                "torch.rand", side_effect=[torch.tensor([0.9]), torch.tensor([0.2])]
            ),
            mock.patch(
                "torch.distributions.Beta.sample", return_value=torch.tensor(0.5)
            ),
            mock.patch(
                "torch.randint", side_effect=[torch.tensor([0]), torch.tensor([0])]
            ),
        ):
            mixed, targets = m.mix(x, y, 3, m.resolve(self.config("dinov3_hf")))
        self.assertEqual(mixed[0].sum().item(), 3.0)
        torch.testing.assert_close(
            targets,
            torch.tensor([[15 / 16, 1 / 16, 0], [1 / 16, 15 / 16, 0]]),
            rtol=0,
            atol=0,
        )
        with (
            mock.patch("torch.randperm", return_value=torch.tensor([1, 0])),
            mock.patch(
                "torch.distributions.Beta.sample", return_value=torch.tensor(0.25)
            ),
            mock.patch(
                "torch.rand",
                side_effect=AssertionError("mixup-only must not select CutMix"),
            ),
        ):
            mixed, targets = m.mix(x, y, 3, m.resolve(self.config("vggt_omega")))
        torch.testing.assert_close(
            mixed[0], torch.full_like(x[0], 0.75), rtol=0, atol=0
        )
        torch.testing.assert_close(
            targets, torch.tensor([[0.25, 0.75, 0], [0.75, 0.25, 0]]), rtol=0, atol=0
        )

    def test_transform_order_distinguishes_erasing_domains_and_evaluation(self):
        m = self.module()
        image = Image.fromarray(
            np.random.RandomState(4).randint(0, 256, (39, 57, 3), dtype="uint8")
        )
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        for kind, value in (
            ("siglip2_g", "random"),
            ("cosmos3_super_vm", 0.5),
            ("raev2_k7", 0),
            ("vjepa2_1", 0),
        ):
            recipe = m.resolve(self.config(kind))
            observed = []

            def erase(self, tensor, observed=observed):
                observed.append((self.value, tensor.clone()))
                return torch.zeros_like(tensor)

            red = Image.new("RGB", (32, 32), (255, 0, 0))
            with (
                mock.patch("torchvision.transforms.RandomErasing.forward", erase),
                mock.patch(
                    "torchvision.transforms.RandAugment.forward", return_value=red
                ),
            ):
                result = m.transform(image, True, 32, recipe)
                evaluation = m.transform(image, False, 32, recipe)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][0], value)
            expected_input = TF.to_tensor(red)
            if kind == "vjepa2_1":
                expected_input = TF.normalize(
                    expected_input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            torch.testing.assert_close(observed[0][1], expected_input, rtol=0, atol=0)
            expected = (
                torch.zeros_like(result)
                if kind == "vjepa2_1"
                else (-mean / std).expand_as(result)
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(evaluation, result))
        recipe = m.resolve(self.config("dinov3_hf"))
        with (
            mock.patch(
                "torchvision.transforms.RandAugment.forward",
                side_effect=AssertionError("unexpected RA"),
            ),
            mock.patch(
                "torchvision.transforms.RandomErasing.forward",
                side_effect=AssertionError("unexpected RE"),
            ),
        ):
            self.assertEqual(m.transform(image, True, 32, recipe).shape, (3, 32, 32))

    def test_both_geometry_profiles_preserve_crop_flip_and_validation_interpolation(
        self,
    ):
        m = self.module()
        image = Image.fromarray(
            np.random.RandomState(18).randint(0, 256, (39, 57, 3), dtype="uint8")
        )
        for kind, interpolation in (
            ("dinov3_hf", Image.Resampling.BILINEAR),
            ("siglip2_g", Image.Resampling.BICUBIC),
        ):
            recipe = m.resolve(self.config(kind))
            for flip in (False, True):
                with (
                    mock.patch(
                        "torchvision.transforms.RandomResizedCrop.get_params",
                        return_value=(2, 3, 10, 15),
                    ),
                    mock.patch(
                        "torch.rand", return_value=torch.tensor([0.2 if flip else 0.9])
                    ),
                    mock.patch(
                        "torchvision.transforms.RandAugment.forward",
                        side_effect=lambda x: x,
                    ),
                    mock.patch(
                        "torchvision.transforms.RandomErasing.forward",
                        side_effect=lambda x: x,
                    ),
                ):
                    actual = m.transform(image, True, 32, recipe)
                expected = image.crop((3, 2, 18, 12)).resize((32, 32), interpolation)
                if flip:
                    expected = expected.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                expected = TF.normalize(
                    TF.to_tensor(expected), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected = TF.center_crop(
                TF.to_tensor(image.resize((int(256 * 57 / 39), 256), interpolation)),
                [32, 32],
            )
            expected = TF.normalize(
                expected, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
            torch.testing.assert_close(
                m.transform(image, False, 32, recipe), expected, rtol=0, atol=0
            )

    def test_unit_repair_is_explicit_and_applies_before_normalization(self):
        m = self.module()
        recipe = m.resolve(self.config("vggt_omega"))
        self.assertTrue(recipe["legacy_unit_repair"])
        image = Image.new("RGB", (32, 32), (128, 128, 128))
        negative = torch.full((3, 32, 32), -0.5)
        with mock.patch(
            "torchvision.transforms.RandomErasing.forward", return_value=negative
        ):
            actual = m.transform(image, True, 32, recipe)
        expected = TF.normalize(
            torch.full_like(negative, 0.25),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for value, expected_value in ((0.6, 0.6), (1.5, 1.0), (-0.00001, 0.0)):
            with mock.patch(
                "torchvision.transforms.RandomErasing.forward",
                return_value=torch.full_like(negative, value),
            ):
                actual = m.transform(image, True, 32, recipe)
            expected = TF.normalize(
                torch.full_like(negative, expected_value),
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dataset_uses_the_selected_source_transform(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image_helpers.TestImageNet().fixture(root)
            for kind in PROFILES:
                recipe = self.module().resolve(self.config(kind))
                dataset = imagenet.ImageNetImages(
                    root, "train", 32, finetune_recipe=recipe
                )
                source = Image.open(dataset.samples[0][0]).convert("RGB")
                torch.manual_seed(9)
                expected = self.module().transform(source, True, 32, recipe)
                torch.manual_seed(9)
                actual, label = dataset[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(label, 0)

    def test_runner_uses_soft_targets_from_mixing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image_helpers.TestImageNet().fixture(root / "data")
            cfg = self.config(root=root / "data")

            def mix(images, labels, num_classes, recipe):
                targets = images.new_zeros((len(labels), num_classes))
                targets[:, 500] = 0.75
                targets[:, 999] = 0.25
                return images, targets

            with mock.patch("downstream.imagenet_finetune.mix", side_effect=mix):
                imagenet.run(cfg, root / "out")
            state = torch.load(
                root / "out/finetune_model.pt", weights_only=True, map_location="cpu"
            )["model"]
            bias = state["classifier.bias"]
            self.assertGreater(bias[500].item(), 0)
            self.assertGreater(bias[999].item(), 0)
            self.assertLess(bias[0].item(), 0)

    def test_runner_clips_large_gradients_and_preserves_small_gradients(self):
        for factor in (0.000001, 1000.0):
            with self.subTest(factor=factor), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                image_helpers.TestImageNet().fixture(root / "data")
                cfg = self.config(root=root / "data")
                cfg["probe"]["max_steps_per_epoch"] = 1
                models, raw = [], []
                cls = imagenet.ImageClassifier

                def amplify(gradient, factor=factor, raw=raw):
                    gradient = gradient * factor
                    raw.append(gradient.detach().clone().flatten())
                    return gradient

                def build(*args, cls=cls, models=models, amplify=amplify, **kwargs):
                    model = cls(*args, **kwargs)
                    models.append(model)
                    for parameter in model.parameters():
                        if parameter.requires_grad:
                            parameter.register_hook(amplify)
                    return model

                with (
                    mock.patch.object(imagenet, "ImageClassifier", side_effect=build),
                    mock.patch(
                        "torch.nn.utils.clip_grad_norm_",
                        wraps=torch.nn.utils.clip_grad_norm_,
                    ) as clipping,
                ):
                    imagenet.run(cfg, root / "out")
                self.assertEqual(clipping.call_count, 1)
                before = torch.linalg.vector_norm(torch.cat(raw)).item()
                after = torch.linalg.vector_norm(
                    torch.cat(
                        [
                            p.grad.flatten()
                            for p in models[0].parameters()
                            if p.grad is not None
                        ]
                    )
                ).item()
                if factor > 1:
                    self.assertGreater(before, 1)
                    self.assertAlmostEqual(after, 1.0, places=5)
                else:
                    self.assertGreater(before, 0)
                    self.assertLess(before, 1)
                    self.assertAlmostEqual(after, before, delta=before * 1e-5)

    def test_all_eight_real_ft_routes_update_encoder_and_head_and_preserve_contract(
        self,
    ):
        from tests import (
            test_basic5_imagenet,
            test_method_hf_basic5,
            test_method_patch_basic5,
        )

        if not all(
            m.HAVE
            for m in (
                test_method_hf_basic5,
                test_method_patch_basic5,
                test_basic5_imagenet,
            )
        ):
            self.skipTest(
                "full downstream model dependencies required for eight-route integration"
            )
        from downstream import contract
        from tests.test_method_cradiov4_h import TestRadio
        from tests.test_method_hf_basic5 import TestVisionFamilies
        from tests.test_method_patch_basic5 import TestPatchFamilies
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega

        for kind, profile in PROFILES.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                owner = None
                if kind in ("clip_hf", "siglip2_g", "dinov3_hf"):
                    owner = TestVisionFamilies()
                elif kind == "cosmos3_super_vm":
                    owner = TestPatchFamilies()
                elif kind == "raev2_k7":
                    owner = TestK7Backbone()
                elif kind == "cradiov4_h":
                    owner = TestRadio()
                elif kind == "vggt_omega":
                    owner = TestOmega()
                if owner:
                    owner.setUp()
                try:
                    cfg = self.config(kind, Path(d) / "data")
                    if owner:
                        spec, _ = (
                            owner.fixture(kind)
                            if kind
                            in ("clip_hf", "siglip2_g", "dinov3_hf", "cosmos3_super_vm")
                            else owner.fixture()
                        )
                        cfg["backbone"] = spec
                    image_helpers.TestImageNet().fixture(Path(cfg["data_root"]))
                    cfg["probe"].update(epochs=2)
                    out, config = Path(d) / "out", Path(d) / "config.json"
                    config.write_text(json.dumps(cfg))
                    models, before = [], []
                    cls = imagenet.ImageClassifier

                    def build(*args, cls=cls, models=models, before=before, **kwargs):
                        model = cls(*args, **kwargs)
                        models.append(model)
                        before.append(copy.deepcopy(model.state_dict()))
                        return model

                    with mock.patch.object(
                        imagenet, "ImageClassifier", side_effect=build
                    ):
                        self.assertEqual(
                            imagenet.main(["--config", str(config), "--out", str(out)]),
                            0,
                        )
                    model = models[0]
                    self.assertTrue(
                        any(
                            not torch.equal(v, before[0]["backbone." + k])
                            for k, v in model.backbone.state_dict().items()
                        )
                    )
                    self.assertFalse(
                        torch.equal(
                            model.classifier.weight, before[0]["classifier.weight"]
                        )
                    )
                    self.assertTrue(
                        all(
                            p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()
                        )
                    )
                    norm = torch.linalg.vector_norm(
                        torch.cat(
                            [
                                p.grad.flatten()
                                for p in model.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                    self.assertGreater(norm.item(), 0.0)
                    self.assertLessEqual(norm.item(), 1.000001)
                    self.assertIsNone(model.reader)
                    result = json.loads((out / "results.json").read_text())
                    self.assertEqual(result["finetune_recipe"]["profile"], profile)
                    self.assertEqual(result["final"]["images"], 2)
                    self.assertFalse(result["canonical_eligible"])
                    self.assertFalse(result["record_value"])
                    self.assertTrue(contract.verify(out, config, 0)[0])
                    checkpoint = torch.load(
                        out / "finetune_model.pt", map_location="cpu", weights_only=True
                    )
                    self.assertEqual(
                        checkpoint["finetune_recipe"], result["finetune_recipe"]
                    )
                    self.assertEqual(checkpoint["epochs_completed"], 2)
                    self.assertEqual(set(checkpoint["model"]), set(model.state_dict()))
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(
                            checkpoint["model"][name], value.cpu(), rtol=0, atol=0
                        )
                finally:
                    if owner:
                        owner.doCleanups()

    def test_missing_model_dependency_skips_only_the_eight_route_integration(self):
        import subprocess
        import sys

        script = """
import sys, unittest
sys.modules['transformers'] = None
from tests.test_method_imagenet_finetune import HAVE, Profiles
assert HAVE
result = unittest.TestResult()
Profiles('test_all_eight_real_ft_routes_update_encoder_and_head_and_preserve_contract').run(result)
assert result.testsRun == 1
assert not result.errors and not result.failures, (result.errors, result.failures)
assert len(result.skipped) == 1, result.skipped
"""
        run = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


class Delivery(unittest.TestCase):
    def test_discovery_does_not_repeat_imported_test_cases(self):
        import sys

        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        for group in suite:
            for case in group:
                self.assertTrue(case.id().startswith(__name__ + "."), case.id())

    @unittest.skipUnless(HAVE, "downstream dependencies required")
    @unittest.skipUnless(
        (ROOT / "docs/examples/imagenet_finetune.json").is_file(), "example unavailable"
    )
    def test_documented_example_is_executable_and_explicit(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / "docs/examples/imagenet_finetune.json").read_text())
        imagenet.validate_config(cfg)
        self.assertEqual(cfg["adaptation"], "finetune")
        self.assertEqual(cfg["probe"]["epochs"], 100)
        self.assertEqual(cfg["probe"]["batch_size"], 1024)
        self.assertEqual(cfg["scheduler_profile"], optimization.REFERENCE_SCHEDULE)
        self.assertEqual(cfg["backbone"]["arch"], "released")

    @unittest.skipUnless(
        (ROOT / ".github/workflows/tests.yml").is_file(), "CI definition unavailable"
    )
    def test_ci_executes_new_tests(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        steps = parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
        self.assertTrue(
            any(
                _runs_finetune_tests(
                    s.get("run", ""), module="tests.test_method_imagenet_finetune"
                )
                for s in steps
            )
        )
