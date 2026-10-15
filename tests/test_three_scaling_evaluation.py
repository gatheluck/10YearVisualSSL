"""Frozen scaling providers preserve source readouts across BasicFive."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import timm  # noqa: F401
    import torch
    from torch import nn
    from torch.nn import functional as F

    HAVE = True
except ImportError:
    HAVE = False

KINDS = ("colorization_scaling", "rotation_scaling", "split_brain_scaling")


class TestCI(unittest.TestCase):
    def test_downstream_job_runs_contracts(self):
        from tests._checkout import is_checkout
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if not HAVE_YAML or not is_checkout():
            self.skipTest("checkout and PyYAML required")
        command = next(
            s["run"]
            for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if s.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        self.assertTrue(
            _runs_finetune_tests(command, module="tests.test_three_scaling_evaluation")
        )


@unittest.skipUnless(HAVE, "torch and timm required")
class TestEvaluation(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def provider(self, kind):
        from downstream import spatial_backbones as sb

        providers = sb.discover_providers()
        self.assertIn(
            kind, providers, "dedicated frozen scaling integration is missing"
        )
        return sb._load_provider(providers[kind])

    def fixture(self, kind):
        provider = self.provider(kind)
        # Patch geometry, not feature behavior: real attention and task heads run.
        patch = mock.patch.object(
            provider, "PRESETS", {"vit_base_patch16_224": (16, 2, 2)}
        )
        patch.start()
        self.addCleanup(patch.stop)
        encoder = provider.build_encoder("vit_base_patch16_224").eval()
        export = {
            "frozen_encoder_version": 1,
            "kind": kind,
            "arch": "vit_base_patch16_224",
            "state_dict": encoder.state_dict(),
        }
        path = self.root / (kind + ".pt")
        torch.save(export, path)
        spec = {
            "kind": kind,
            "arch": "vit_base_patch16_224",
            "encoder": str(path),
            "img_size": 224,
            "patch_size": 16,
        }
        return provider, encoder, export, spec

    def oracle_tokens(self, model, x):
        from timm.layers import resample_abs_pos_embed

        m = copy.deepcopy(model).eval()
        grid = tuple(s // 16 for s in x.shape[-2:])
        m.pos_embed = nn.Parameter(
            resample_abs_pos_embed(
                m.pos_embed,
                grid,
                old_size=(14, 14),
                num_prefix_tokens=1,
                interpolation="bicubic",
                antialias=True,
            )
        )
        if hasattr(m.patch_embed, "img_size"):
            m.patch_embed.img_size = x.shape[-2:]
        with torch.no_grad():
            if hasattr(m, "forward_features"):
                return m.forward_features(x)
            return m(x)

    def test_documented_configuration_validates(self):
        from tests._checkout import is_checkout

        if not is_checkout():
            self.skipTest("checkout required for documentation")
        from downstream import imagenet

        path = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/three_scaling_linear.json"
        )
        cfg = json.loads(path.read_text())
        for kind in KINDS:
            cfg["backbone"]["kind"] = kind
            imagenet.validate_config(cfg)

    def test_native_geometry_and_normalization_are_distinct(self):
        for kind in KINDS:
            p = self.provider(kind)
            self.assertEqual(
                p.PRESETS,
                {
                    "vit_base_patch16_224": (768, 12, 12),
                    "vit_large_patch16_224": (1024, 24, 16),
                },
            )
            with mock.patch.object(p, "PRESETS", {"vit_base_patch16_224": (16, 2, 2)}):
                encoder = p.build_encoder("vit_base_patch16_224")
            branches = (
                list(encoder) if isinstance(encoder, nn.ModuleList) else [encoder]
            )
            self.assertEqual(len(branches), 2 if kind == "split_brain_scaling" else 1)
            for branch in branches:
                norm, cls_token = branch.norm, branch.cls_token
                assert isinstance(norm, nn.LayerNorm)
                assert isinstance(cls_token, torch.Tensor)
                self.assertEqual(
                    norm.eps, 1e-6 if kind == "split_brain_scaling" else 1e-5
                )
                self.assertEqual(cls_token.shape[-1], 8 if len(branches) == 2 else 16)
            self.assertEqual(p.SUPPORTED_ADAPTATIONS, ("frozen",))

    def test_readouts_frozen_state_and_rectangular_positions(self):
        for kind in KINDS:
            p, encoder, _, spec = self.fixture(kind)
            bb = p.build(spec)
            raw = torch.rand(2, 3, 32, 48)
            normalized = (
                raw - raw.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            ) / raw.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            # Independently measured standard D65 primary/white/black values
            if kind == "rotation_scaling":
                expected = self.oracle_tokens(encoder, normalized)
            else:
                from downstream.frozen_vit import lab_inputs

                l, ab = lab_inputs(raw)
                if kind == "colorization_scaling":
                    from downstream.frozen_vit import lightness_input

                    expected = self.oracle_tokens(encoder, lightness_input(raw))
                else:
                    expected = torch.cat(
                        [
                            self.oracle_tokens(encoder[0], l),
                            self.oracle_tokens(encoder[1], ab),
                        ],
                        -1,
                    )
            torch.testing.assert_close(
                bb.classification_features(normalized, adaptation="frozen"),
                F.normalize(expected[:, 0], dim=-1),
                atol=2e-6,
                rtol=2e-6,
            )
            torch.testing.assert_close(
                bb.forward_features(normalized),
                expected[:, 1:].transpose(1, 2).reshape(2, 16, 2, 3),
                atol=2e-6,
                rtol=2e-6,
            )
            clip = normalized.reshape(1, 2, 3, 32, 48)
            torch.testing.assert_close(
                bb.classification_features(clip, adaptation="frozen", video=True),
                expected[:, 0].mean(0, keepdim=True),
                atol=2e-6,
                rtol=2e-6,
            )
            detection_input = normalized if kind == "rotation_scaling" else raw
            torch.testing.assert_close(
                bb.forward_detection_features(detection_input),
                bb.forward_features(normalized),
                atol=2e-6,
                rtol=2e-6,
            )
            bb.train()
            self.assertFalse(bb.training)
            self.assertTrue(all(not v.requires_grad for v in bb.parameters()))
            with self.assertRaises(ValueError):
                bb.classification_features(normalized, adaptation="finetune")
            with self.assertRaises(ValueError):
                bb.forward_features(raw[:, :, :31])
            with self.assertRaises(ValueError):
                bb.classification_features(clip[:, :0], adaptation="frozen", video=True)

    def test_lab_primary_colors_and_boundary_values(self):
        # First assert the feature exists, so absent implementation is a meaningful RED.
        self.provider(KINDS[0])
        from downstream.frozen_vit import lab_inputs, lightness_input

        rgb = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).T[None, :, :, None]
        l, ab = lab_inputs(rgb)
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [53.2408, 80.0925, 67.2032],
                [87.7347, -86.1827, 83.1793],
                [32.2970, 79.1875, -107.8602],
            ]
        )
        torch.testing.assert_close(
            (l[0, 0, :, 0] * 50 + 50), expected[:, 0], atol=0.001, rtol=0
        )
        torch.testing.assert_close(
            ab[0, :, :, 0].T * 128, expected[:, 1:], atol=0.001, rtol=0
        )
        torch.testing.assert_close(
            lightness_input(rgb)[0, 0, :, 0] * 100, expected[:, 0], atol=0.001, rtol=0
        )
        torch.testing.assert_close(
            lightness_input(rgb * 3 - 1),
            lightness_input((rgb * 3 - 1).clamp(0, 1)),
            atol=0,
            rtol=0,
        )

    def test_export_rejects_wrong_identity_and_corrupt_tensors(self):
        for kind in KINDS:
            p, _, export, spec = self.fixture(kind)
            for case in (
                "kind",
                "version",
                "arch",
                "missing",
                "extra",
                "nan",
                "shape",
                "dtype",
                "override",
            ):
                bad = copy.deepcopy(export)
                options = dict(spec)
                key = next(iter(bad["state_dict"]))
                if case == "kind":
                    bad["kind"] = "other"
                if case == "version":
                    bad["frozen_encoder_version"] = True
                if case == "arch":
                    bad["arch"] = "unknown"
                if case == "missing":
                    del bad["state_dict"][key]
                if case == "extra":
                    bad["state_dict"]["decoder.weight"] = torch.zeros(1)
                if case == "nan":
                    bad["state_dict"][key].fill_(float("nan"))
                if case == "shape":
                    bad["state_dict"][key] = torch.zeros(3)
                if case == "dtype":
                    bad["state_dict"][key] = bad["state_dict"][key].double()
                if case == "override":
                    options["embed_dim"] = 16
                torch.save(bad, spec["encoder"])
                with self.subTest(kind=kind, case=case), self.assertRaises(ValueError):
                    p.build(options)

    def test_native_import_explicit_identity_and_readonly_source(self):
        for kind in KINDS:
            p, encoder, _, spec = self.fixture(kind)
            from downstream.frozen_vit import import_native

            # Provider declares exact source prefixes; no best-effort key matching.
            states = list(encoder) if isinstance(encoder, nn.ModuleList) else [encoder]
            raw = {
                "module." + prefix + k: v
                for prefix, branch in zip(p.NATIVE_PREFIXES, states)
                for k, v in branch.state_dict().items()
            }
            raw["module." + p.NATIVE_DISCARDED_PREFIXES[0] + "weight"] = torch.zeros(1)
            import numpy as np

            blob = {
                "rng_state": {"numpy": np.random.get_state()},
                "state_dict": raw,
                "epoch": 99,
                "private_path": "/private/not-for-export",
            }
            source = self.root / (kind + "-native.pt")
            torch.save(blob, source)
            before = source.read_bytes()
            args = {
                "provider": p,
                "source": source,
                "sha256": hashlib.sha256(before).hexdigest(),
                "arch": spec["arch"],
                "state_key": "state_dict",
                "epoch_key": "epoch",
                "epoch_value": 99,
                "module_prefix": True,
                "out": self.root / (kind + "-out"),
            }
            from downstream import frozen_vit
            from downstream import spatial_backbones as sb

            with mock.patch.object(sb, "_load_provider", return_value=p):
                self.assertEqual(
                    frozen_vit.main(
                        [
                            "--kind",
                            kind,
                            "--source",
                            str(source),
                            "--sha256",
                            args["sha256"],
                            "--arch",
                            spec["arch"],
                            "--state-key",
                            "state_dict",
                            "--epoch-key",
                            "epoch",
                            "--epoch-value",
                            "99",
                            "--module-prefix",
                            "--out",
                            str(args["out"]),
                        ]
                    ),
                    0,
                )
            record = json.loads((args["out"] / "export.json").read_text())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(record["epoch_value"], 99)
            self.assertNotIn("/private", json.dumps(record))
            result = torch.load(args["out"] / "encoder.pt", weights_only=True)
            self.assertEqual(
                set(result), {"frozen_encoder_version", "kind", "arch", "state_dict"}
            )
            for key, value in result["state_dict"].items():
                torch.testing.assert_close(
                    value, encoder.state_dict()[key], atol=0, rtol=0
                )
            with self.assertRaises(FileExistsError):
                import_native(**args)
            for case in ("sha256", "epoch_value", "state_key", "module_prefix"):
                out = self.root / (kind + "-" + case)
                options = dict(args, out=out)
                options[case] = {
                    "sha256": "0" * 64,
                    "epoch_value": 100,
                    "state_key": "model",
                    "module_prefix": False,
                }[case]
                with self.subTest(case=case), self.assertRaises(ValueError):
                    import_native(**options)
                self.assertFalse(out.exists())

    def test_hash_is_verified_before_deserialization(self):
        from downstream.frozen_vit import import_native

        p, _, _, spec = self.fixture(KINDS[0])
        source = self.root / "invalid-native.pt"
        source.write_bytes(b"untrusted bytes must not reach the loader")
        with mock.patch(
            "downstream.native_checkpoint.load_native",
            side_effect=AssertionError("deserialized before hash verification"),
        ) as loader:
            with self.assertRaisesRegex(ValueError, "sha256"):
                import_native(
                    provider=p,
                    source=source,
                    sha256="0" * 64,
                    arch=spec["arch"],
                    state_key="state_dict",
                    epoch_key="epoch",
                    epoch_value=1,
                    module_prefix=False,
                    out=self.root / "must-not-exist",
                )
            loader.assert_not_called()
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_native_import_rejects_unexpected_encoder_parameters(self):
        from downstream.frozen_vit import import_native

        for kind in KINDS:
            p, encoder, _, spec = self.fixture(kind)
            states = list(encoder) if isinstance(encoder, nn.ModuleList) else [encoder]
            raw = {
                prefix + k: v
                for prefix, branch in zip(p.NATIVE_PREFIXES, states)
                for k, v in branch.state_dict().items()
            }
            raw[p.NATIVE_PREFIXES[0] + "unexpected.weight"] = torch.zeros(1)
            source = self.root / (kind + "-extra.pt")
            torch.save({"model": raw, "epoch": 1}, source)
            with self.assertRaises(ValueError):
                import_native(
                    provider=p,
                    source=source,
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    arch=spec["arch"],
                    state_key="model",
                    epoch_key="epoch",
                    epoch_value=1,
                    module_prefix=False,
                    out=self.root / (kind + "-bad"),
                )

    def test_task_dependencies_are_isolated(self):
        import subprocess
        import sys

        try:
            import av  # noqa: F401
            import h5py  # noqa: F401
            import pycocotools  # noqa: F401
            import scipy.io  # noqa: F401
        except ImportError:
            self.skipTest(
                "dependency-isolation controls require downstream environment"
            )

        # Each real task remains executable when an unrelated reader is absent.
        # The complete downstream environment must execute all fifteen entries.
        for missing, skipped in (
            ("", 0),
            ("scipy", 1),
            ("h5py", 1),
            ("av", 1),
            ("pycocotools", 1),
            ("scipy,h5py,av,pycocotools", 3),
        ):
            with self.subTest(missing=missing):
                script = r"""
import importlib.util
import sys
import unittest
missing, expected = sys.argv[1], int(sys.argv[2])
for reader in filter(None, missing.split(",")):
    sys.modules[reader] = None
    assert importlib.util.find_spec(reader) is None
    try:
        __import__(reader)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("dependency isolation was not effective")
from tests.test_three_scaling_evaluation import TestEvaluation
names = ["test_real_task_entries_" + task
         for task in ("imagenet", "ade20k", "coco", "nyuv2", "ssv2")]
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.TestSuite(TestEvaluation(n) for n in names))
assert result.wasSuccessful(), (result.errors, result.failures)
assert result.testsRun == 5, result.testsRun
assert len(result.skipped) == expected, result.skipped
"""
                result = subprocess.run(
                    [sys.executable, "-c", script, missing, str(skipped)],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_task_entries_imagenet(self):
        from downstream import imagenet
        from tests.test_basic5_imagenet import TestImageNet

        self._task_entries(imagenet, TestImageNet().fixture, TestImageNet().config)

    def test_real_task_entries_ade20k(self):
        from downstream import ade20k
        from tests.test_downstream_ade20k import smoke_config, tiny_ade

        self._task_entries(ade20k, tiny_ade, smoke_config)

    def test_real_task_entries_coco(self):
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("COCO entries require pycocotools")
        from downstream import coco
        from tests.test_downstream_coco import smoke_config, tiny_coco

        self._task_entries(coco, tiny_coco, smoke_config)

    def test_real_task_entries_nyuv2(self):
        try:
            import h5py  # noqa: F401
            from scipy.io import savemat
        except ImportError:
            self.skipTest("NYUv2 entries require h5py and scipy")
        from downstream import nyuv2
        from tests.test_downstream_nyuv2 import smoke_config, tiny_nyuv2

        def fixture(root):
            tiny_nyuv2(root)
            savemat(
                root / "labeled/splits.mat",
                {"trainNdxs": [[1], [2], [3]], "testNdxs": [[4], [5], [6]]},
            )

        self._task_entries(nyuv2, fixture, smoke_config)

    def test_real_task_entries_ssv2(self):
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("SSv2 entries require av")
        from downstream import ssv2
        from tests.test_downstream_ssv2 import smoke_config, tiny_ssv2

        self._task_entries(ssv2, tiny_ssv2, smoke_config)

    def _task_entries(self, module, fixture, factory):
        from downstream import contract
        from downstream import spatial_backbones as sb

        for kind in KINDS:
            p, _, _, spec = self.fixture(kind)
            original = sb._load_provider
            with mock.patch.object(
                sb,
                "_load_provider",
                side_effect=lambda path, p=p, kind=kind, original=original: (
                    p if sb._provider_kind(path) == kind else original(path)
                ),
            ):
                root = self.root / kind / module.TASK / "data"
                fixture(root)
                cfg = factory(root)
                detection = module.TASK == "coco_detection"
                classification = module.TASK == "imagenet_classification"
                if not classification:
                    cfg.update(
                        profile="capture_basic5_components",
                        adaptation="frozen",
                        optimizer_profile="basic5_frozen_v1",
                    )
                    settings = cfg["detector" if detection else "probe"]
                    settings.update(
                        lr="protocol",
                        batch_size=2,
                        max_train_samples=0,
                        max_val_samples=0,
                        max_steps_per_epoch=0,
                    )
                    if detection:
                        settings["anchor_sizes"] = [8, 16, 32, 64]
                cfg["backbone"] = spec
                if detection:
                    cfg["detector_profile"] = "captured_native_detection_v1"
                if classification:
                    cfg["preprocessing_profile"] = "captured_provider_v1"
                if module.TASK == "ssv2_video":
                    cfg["video_recipe"] = "captured_provider_video_v1"
                cfg["detector" if detection else "probe"].update(
                    epochs=1, max_val_samples=2
                )
                path, out = root.parent / "config.json", root.parent / "out"
                path.write_text(json.dumps(cfg))
                with self.subTest(kind=kind, task=module.TASK):
                    self.assertEqual(
                        module.main(["--config", str(path), "--out", str(out)]), 0
                    )
                    self.assertTrue(contract.verify(out, path, 0)[0])
                    result = json.loads((out / "results.json").read_text())
                    self.assertFalse(result["canonical_eligible"])
                    self.assertFalse(result["record_value"])
