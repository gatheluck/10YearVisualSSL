"""Source-selected LP/AP geometry, with explicit legacy and FT boundaries."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from PIL import Image
    from torchvision import transforms as T

    from downstream import imagenet
    from tests import test_basic5_imagenet as helpers

    HAVE = True
except ImportError:
    HAVE = False

PROFILE = "captured_provider_v1"
EXPECTED = {
    "clip_hf": "bicubic",
    "siglip2_g": "bicubic",
    "cradiov4_h": "bicubic",
    "cosmos3_super_vm": "bicubic",
    "dinov3_hf": "bilinear",
    "raev2_k7": "bilinear",
    "vjepa2_1": "bilinear",
    "vggt_omega": "bicubic",
}


@unittest.skipUnless(HAVE, "image dependencies required")
class Preprocessing(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self, root=Path("/unused"), kind="vjepa2_1", adaptation="frozen"):
        cfg = helpers.TestImageNet().config(root)
        cfg.update(preprocessing_profile=PROFILE, adaptation=adaptation)
        cfg["backbone"]["kind"] = kind
        if adaptation == "attentive":
            from downstream.spatial_backbones import attentive_profile

            if attentive_profile(kind) is not None:
                cfg["reader_profile"] = attentive_profile(kind)
        return cfg

    def test_all_sixteen_explicit_selections_and_metadata(self):
        for kind, interpolation in EXPECTED.items():
            for adaptation in ("frozen", "attentive"):
                with self.subTest(kind=kind, adaptation=adaptation):
                    cfg = self.config(kind=kind, adaptation=adaptation)
                    imagenet.validate_config(cfg)
                    report = imagenet.resolve_preprocessing(cfg)
                    self.assertEqual(report["profile"], PROFILE)
                    self.assertEqual(report["interpolation"], interpolation)
                    self.assertFalse(report["historical_run_verified"])

    def test_invalid_profile_provider_adaptation_and_released_size_are_refused(self):
        cfg = self.config()
        cases = [
            cfg | {"preprocessing_profile": "typo"},
            cfg | {"preprocessing_profile": None},
            cfg | {"preprocessing_profile": True},
            cfg | {"backbone": dict(cfg["backbone"], kind="sam3_trunk")},
            cfg | {"backbone": dict(cfg["backbone"], kind="unknown")},
            cfg
            | {
                "adaptation": "finetune",
                "optimizer_profile": "basic5_finetune_v1",
                "finetune_recipe": "captured_bilinear_normalized_zero_torch_v1",
            },
            cfg | {"backbone": dict(cfg["backbone"], arch="released")},
        ]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                imagenet.validate_config(bad)
        cfg["backbone"]["arch"] = "released"
        cfg["probe"]["image_size"] = 224
        imagenet.validate_config(cfg)

    def test_training_and_validation_match_source_geometry_and_rng(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helpers.TestImageNet().fixture(root)
            for kind, mode in EXPECTED.items():
                recipe = imagenet.resolve_preprocessing(self.config(kind=kind))
                interpolation = (
                    T.InterpolationMode.BICUBIC
                    if mode == "bicubic"
                    else T.InterpolationMode.BILINEAR
                )
                for train in (False, True):
                    ds = imagenet.ImageNetImages(
                        root, "train" if train else "val", 224, preprocessing=recipe
                    )
                    with Image.open(ds.samples[0][0]) as image:
                        image = image.convert("RGB")
                    transform = (
                        T.Compose(
                            [
                                T.RandomResizedCrop(
                                    224,
                                    scale=(0.08, 1.0),
                                    ratio=(0.75, 4.0 / 3.0),
                                    interpolation=interpolation,
                                ),
                                T.RandomHorizontalFlip(0.5),
                                T.ToTensor(),
                            ]
                        )
                        if train
                        else T.Compose(
                            [
                                T.Resize(256, interpolation=interpolation),
                                T.CenterCrop(224),
                                T.ToTensor(),
                            ]
                        )
                    )
                    for seed in (0, 3, 7):
                        with self.subTest(kind=kind, train=train, seed=seed):
                            torch.manual_seed(seed)
                            expected = T.functional.normalize(
                                transform(image),
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225],
                            )
                            rng = torch.get_rng_state()
                            torch.manual_seed(seed)
                            actual, label = ds[0]
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                            self.assertEqual(label, 0)
                    if not train and mode == "bicubic":
                        legacy, _ = imagenet.ImageNetImages(root, "val", 224)[0]
                        self.assertFalse(torch.equal(actual, legacy))

    def test_legacy_and_finetune_remain_separate(self):
        cfg = self.config()
        del cfg["preprocessing_profile"]
        imagenet.validate_config(cfg)
        self.assertIsNone(imagenet.resolve_preprocessing(cfg))
        cfg.update(
            adaptation="finetune",
            optimizer_profile="basic5_finetune_v1",
            finetune_recipe="captured_bilinear_normalized_zero_torch_v1",
        )
        imagenet.validate_config(cfg)
        self.assertIsNone(imagenet.resolve_preprocessing(cfg))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helpers.TestImageNet().fixture(root)
            with self.assertRaises(ValueError):
                imagenet.ImageNetImages(
                    root,
                    "val",
                    224,
                    preprocessing={"interpolation": "bicubic"},
                    finetune_recipe={"interpolation": "bilinear"},
                )

    def test_all_sixteen_real_routes_use_profile_and_keep_frozen_encoder(self):
        from tests import test_method_hf_basic5 as hf
        from tests import test_method_patch_basic5 as patch

        if not (helpers.HAVE and hf.HAVE and patch.HAVE):
            self.skipTest("full encoder dependencies required for real routes")
        from downstream import contract
        from tests.test_method_cradiov4_h import TestRadio
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega

        for kind, mode in EXPECTED.items():
            for adaptation in ("frozen", "attentive"):
                with (
                    self.subTest(kind=kind, adaptation=adaptation),
                    tempfile.TemporaryDirectory() as d,
                ):
                    owner = (
                        hf.TestVisionFamilies()
                        if kind in ("clip_hf", "siglip2_g", "dinov3_hf")
                        else patch.TestPatchFamilies()
                        if kind == "cosmos3_super_vm"
                        else TestRadio()
                        if kind == "cradiov4_h"
                        else TestK7Backbone()
                        if kind == "raev2_k7"
                        else TestOmega()
                        if kind == "vggt_omega"
                        else None
                    )
                    if owner:
                        owner.setUp()
                    try:
                        root, out, path = (
                            Path(d) / "data",
                            Path(d) / "out",
                            Path(d) / "config.json",
                        )
                        helpers.TestImageNet().fixture(root)
                        cfg = self.config(root, kind, adaptation)
                        if owner:
                            cfg["backbone"], _ = (
                                owner.fixture(kind)
                                if kind
                                in (
                                    "clip_hf",
                                    "siglip2_g",
                                    "dinov3_hf",
                                    "cosmos3_super_vm",
                                )
                                else owner.fixture()
                            )
                        path.write_text(json.dumps(cfg))
                        models, before, datasets = [], [], []
                        classifier, dataset = (
                            imagenet.ImageClassifier,
                            imagenet.ImageNetImages,
                        )

                        def build(
                            *args,
                            classifier=classifier,
                            models=models,
                            before=before,
                            **kwargs,
                        ):
                            model = classifier(*args, **kwargs)
                            models.append(model)
                            before.append(copy.deepcopy(model.state_dict()))
                            return model

                        def data(*args, dataset=dataset, datasets=datasets, **kwargs):
                            ds = dataset(*args, **kwargs)
                            datasets.append(ds)
                            return ds

                        with (
                            mock.patch.object(imagenet, "ImageClassifier", build),
                            mock.patch.object(imagenet, "ImageNetImages", data),
                        ):
                            self.assertEqual(
                                imagenet.main(
                                    ["--config", str(path), "--out", str(out)]
                                ),
                                0,
                            )
                        report = json.loads((out / "results.json").read_text())
                        self.assertEqual(report["preprocessing"]["interpolation"], mode)
                        self.assertEqual(report["preprocessing"]["profile"], PROFILE)
                        self.assertFalse(
                            report["preprocessing"]["historical_run_verified"]
                        )
                        self.assertFalse(report["canonical_eligible"])
                        self.assertFalse(report["record_value"])
                        self.assertTrue(contract.verify(out, path, 0)[0])
                        self.assertEqual(len(datasets), 2)
                        for ds in datasets:
                            self.assertEqual(ds.preprocessing["interpolation"], mode)
                        model = models[0]
                        for name, value in model.backbone.state_dict().items():
                            torch.testing.assert_close(
                                value, before[0]["backbone." + name], rtol=0, atol=0
                            )
                        self.assertFalse(
                            torch.equal(
                                model.classifier.weight, before[0]["classifier.weight"]
                            )
                        )
                    finally:
                        if owner:
                            owner.tearDown()


class Delivery(unittest.TestCase):
    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml").is_file(),
        "CI unavailable",
    )
    def test_ci_runs_preprocessing_suite(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        self.assertTrue(
            any(
                _runs_finetune_tests(
                    s.get("run", ""), module="tests.test_method_imagenet_preprocessing"
                )
                for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            )
        )

    @unittest.skipUnless(HAVE, "image dependencies required")
    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "docs").is_dir(), "docs unavailable"
    )
    def test_documented_example_selects_real_provider_geometry(self):
        path = Path(__file__).resolve().parents[1] / "docs/examples/imagenet_probe.json"
        self.assertTrue(path.is_file(), "LP/AP example is missing")
        cfg = json.loads(path.read_text())
        imagenet.validate_config(cfg)
        self.assertEqual(cfg["preprocessing_profile"], PROFILE)
        self.assertEqual(cfg["adaptation"], "attentive")
        self.assertEqual(cfg["probe"]["image_size"], 224)
        self.assertEqual(cfg["backbone"]["arch"], "released")
        self.assertEqual(
            imagenet.resolve_preprocessing(cfg)["interpolation"], "bicubic"
        )

    @unittest.skipUnless(HAVE, "image dependencies required")
    def test_partial_dependencies_preserve_geometry_checks(self):
        import subprocess
        import sys

        for missing in (
            ("timm",),
            ("einops",),
            ("transformers",),
            ("timm", "einops", "transformers"),
        ):
            with self.subTest(missing=missing):
                script = """
import importlib, json, sys, unittest
for name in json.loads(sys.argv[1]):
    sys.modules[name] = None
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('dependency hiding did not work')
from tests.test_method_imagenet_preprocessing import HAVE, Preprocessing
assert HAVE, 'image checks must remain available'
names = ['test_all_sixteen_explicit_selections_and_metadata',
         'test_training_and_validation_match_source_geometry_and_rng',
         'test_all_sixteen_real_routes_use_profile_and_keep_frozen_encoder']
result = unittest.TestResult()
unittest.TestSuite(Preprocessing(n) for n in names).run(result)
assert result.testsRun == 3
assert not result.errors and not result.failures, (result.errors, result.failures)
assert len(result.skipped) == 1, result.skipped
assert result.skipped[0][0]._testMethodName == names[-1]
"""
                result = subprocess.run(
                    [sys.executable, "-c", script, json.dumps(missing)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
