"""Provider-owned video recipes and portable LP/AP/FT execution."""

import copy
import importlib.util
import itertools
import json
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAVE = all(
    importlib.util.find_spec(n) for n in ("torch", "torchvision", "numpy", "PIL", "av")
)
REAL = HAVE and all(importlib.util.find_spec(n) for n in ("timm", "einops"))
if HAVE:
    import numpy as np
    import torch

    from downstream import optimization, ssv2
    from tests import test_downstream_ssv2 as fixtures
    from tests import test_method_imagenet_finetune as image_ft


@unittest.skipUnless(HAVE, "video dependencies required")
class Video(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self, root=Path("/unused"), adaptation="frozen", kind="vjepa2_1"):
        cfg = fixtures.smoke_config(root)
        cfg.update(
            profile="capture_basic5_components",
            adaptation=adaptation,
            optimizer_profile="basic5_finetune_v1"
            if adaptation == "finetune"
            else "basic5_frozen_v1",
            video_recipe="captured_provider_video_v1",
        )
        from tests.test_method_vjepa2_1_training import TestFinetuneOptimization

        cfg["backbone"] = TestFinetuneOptimization().config()["backbone"]
        cfg["backbone"]["kind"] = kind
        cfg["probe"].update(
            lr="protocol",
            batch_size=2,
            num_frames=2,
            image_size=32,
            max_steps_per_epoch=0,
            max_train_samples=0,
            max_val_samples=0,
        )
        if adaptation == "attentive":
            from downstream.spatial_backbones import attentive_profile

            if attentive_profile(kind):
                cfg["reader_profile"] = attentive_profile(kind)
        return cfg

    def test_twenty_four_video_recipes_and_source_metadata(self):
        for kind in image_ft.PROFILES:
            for adaptation in ("frozen", "attentive", "finetune"):
                with self.subTest(kind=kind, adaptation=adaptation):
                    cfg = self.config(adaptation=adaptation, kind=kind)
                    ssv2.validate_config(cfg)
                    from downstream.video_recipe import resolve

                    report = resolve(cfg)
                    self.assertEqual(
                        report["temporal_rng"],
                        "torch"
                        if kind in ("dinov3_hf", "raev2_k7", "vjepa2_1")
                        else "python",
                    )
                    self.assertEqual(
                        report["interpolation"],
                        "bilinear"
                        if kind in ("dinov3_hf", "raev2_k7", "vjepa2_1")
                        else "bicubic",
                    )
                    self.assertFalse(report["horizontal_flip"])
                    self.assertEqual(report["label_smoothing"], 0.0)
                    self.assertFalse(report["historical_run_verified"])
                    self.assertEqual(
                        report["batch_mixing"] != "none", adaptation == "finetune"
                    )
        for changes in (
            {"video_recipe": "typo"},
            {"profile": "legacy"},
            {"task": "imagenet_classification"},
        ):
            with self.assertRaises((ValueError, ssv2.ConfigError)):
                ssv2.validate_config(self.config() | changes)
        with self.assertRaises(ValueError):
            resolve(self.config() | {"task": "imagenet_classification"})
        unknown = self.config()
        unknown["backbone"]["kind"] = "uninspected_provider"
        with self.assertRaises(ValueError):
            resolve(unknown)
        cfg = self.config()
        cfg["backbone"]["arch"] = "released"
        with self.assertRaises((ValueError, ssv2.ConfigError)):
            ssv2.validate_config(cfg)

    def test_temporal_rng_and_clip_wide_mixing_match_reference_operations(self):
        from downstream import video_recipe

        for kind in image_ft.PROFILES:
            cfg = self.config(adaptation="finetune", kind=kind)
            recipe = video_recipe.resolve(cfg)
            for seed in (3, 8):
                ssv2.make_deterministic(seed)
                indices = video_recipe.sample_indices(19, 7, True, recipe)
                ssv2.make_deterministic(seed)
                bounds = [round(i * 19 / 7) for i in range(8)]
                expected = []
                for a, b in itertools.pairwise(bounds):
                    expected.append(
                        random.randint(a, b - 1)
                        if recipe["temporal_rng"] == "python"
                        else int(torch.randint(a, b, (1,)))
                    )
                self.assertEqual(indices, expected)
                self.assertEqual(
                    video_recipe.sample_indices(19, 7, False, recipe),
                    [1, 3, 6, 9, 12, 14, 17],
                )
                self.assertEqual(
                    video_recipe.sample_indices(2, 7, False, recipe),
                    [0, 0, 1, 1, 1, 1, 1],
                )
                clips = torch.arange(4 * 3 * 3 * 6 * 8, dtype=torch.float32).reshape(
                    4, 3, 3, 6, 8
                )
                labels = torch.tensor([0, 1, 2, 3])
                original = clips.clone()
                ssv2.make_deterministic(seed)
                actual, targets = video_recipe.mix(clips, labels, 174, recipe)
                from downstream.imagenet_finetune import mix

                ssv2.make_deterministic(seed)
                expected, soft = mix(clips.reshape(4, 9, 6, 8), labels, 174, recipe)
                torch.testing.assert_close(
                    actual, expected.reshape_as(clips), rtol=0, atol=0
                )
                torch.testing.assert_close(targets, soft, rtol=0, atol=0)
                torch.testing.assert_close(clips, original, rtol=0, atol=0)

    def test_execution_has_effective_batch_and_fifty_epoch_horizon(self):
        for kind in image_ft.PROFILES:
            for adaptation in ("frozen", "attentive", "finetune"):
                cfg = self.config(adaptation=adaptation, kind=kind)
                cfg["execution"] = {
                    "profile": "captured_video_v1",
                    "accumulation_steps": 2,
                    "precision": "fp32",
                    "tail_policy": "discard",
                }
                ssv2.validate_config(cfg)
                report = optimization.resolve_optimization(cfg, ssv2.TASK)
                self.assertEqual(report["effective_batch"], 4)
                cfg["scheduler_profile"] = "basic5_reference_schedule_v1"
                cfg["probe"]["batch_size"] = 128
                report = optimization.resolve_optimization(cfg, ssv2.TASK)
                self.assertEqual(report["epochs"], 50)
                self.assertEqual(report["effective_batch"], 256)
                bad = copy.deepcopy(cfg)
                bad["probe"]["epochs"] = 51
                with self.assertRaises((ValueError, ssv2.ConfigError)):
                    ssv2.validate_config(bad)

    def test_clip_augmentation_and_validation_keep_provider_semantics(self):
        from PIL import Image
        from torchvision import transforms as T
        from torchvision.transforms import functional as TF

        from downstream import video_recipe

        base = Image.fromarray(
            np.random.RandomState(8).randint(0, 256, (80, 96, 3), dtype=np.uint8)
        )
        for kind in image_ft.PROFILES:
            for adaptation in ("frozen", "attentive", "finetune"):
                recipe = video_recipe.resolve(
                    self.config(adaptation=adaptation, kind=kind)
                )
                frames = [base.copy() for _ in range(4)]
                ssv2.make_deterministic(3)
                output = video_recipe.transform(frames, False, 32, recipe)
                interpolation = (
                    T.InterpolationMode.BILINEAR
                    if kind in ("dinov3_hf", "raev2_k7", "vjepa2_1")
                    else T.InterpolationMode.BICUBIC
                )
                expected = TF.normalize(
                    TF.to_tensor(
                        TF.center_crop(TF.resize(base, [256], interpolation), [32] * 2)
                    ),
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                )
                torch.testing.assert_close(
                    output, torch.stack([expected] * 4), rtol=0, atol=0
                )
                different = 0
                for seed in range(12):
                    ssv2.make_deterministic(seed)
                    output = video_recipe.transform(frames, True, 32, recipe)
                    self.assertEqual(output.shape, (4, 3, 32, 32))
                    self.assertTrue(torch.isfinite(output).all())
                    different += any(
                        not torch.equal(output[0], frame) for frame in output[1:]
                    )
                if kind == "vjepa2_1" and adaptation == "finetune":
                    self.assertGreater(different, 0)
                else:
                    self.assertEqual(different, 0)

    def test_actual_video_lp_ap_ft_resume_and_exports(self):
        if not REAL:
            self.skipTest("real encoder dependencies required")
        from tests.test_method_imagenet_execution import Execution

        for adaptation in ("frozen", "attentive", "finetune"):
            with (
                self.subTest(adaptation=adaptation),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                data = fixtures.tiny_ssv2(root / "data")
                # Five clips exercise a discarded epoch tail with physical batch one.
                entries = json.loads((data / "labels/train.json").read_text())
                (data / "labels/train.json").write_text(json.dumps((entries * 3)[:5]))
                cfg = self.config(data, adaptation)
                cfg["probe"].update(batch_size=1, epochs=3)
                cfg["execution"] = {
                    "profile": "captured_video_v1",
                    "accumulation_steps": 2,
                    "precision": "fp32",
                    "tail_policy": "discard",
                }
                for name, epochs, resume in (
                    ("full", 3, None),
                    ("first", 1, None),
                    ("continued", 3, root / "first/resume.pt"),
                ):
                    c = copy.deepcopy(cfg)
                    c["probe"]["epochs"] = epochs
                    if resume:
                        c["resume"] = str(resume)
                    path = root / (name + ".json")
                    path.write_text(json.dumps(c))
                    self.assertEqual(
                        ssv2.main(["--config", str(path), "--out", str(root / name)]), 0
                    )
                full = torch.load(root / "full/resume.pt", weights_only=True)
                resumed = torch.load(root / "continued/resume.pt", weights_only=True)
                Execution().equal_tree(full, resumed)
                self.assertEqual(full["format"], "ssv2_epoch_v1")
                self.assertEqual(full["runtime"]["updates"], 6)
                self.assertEqual(full["runtime"]["schedule_horizon"], 250)
                self.assertEqual(full["runtime"]["discarded_microbatches"], 3)
                self.assertEqual(
                    any(k.startswith("backbone.") for k in full["model"]),
                    adaptation == "finetune",
                )
                report = json.loads((root / "continued/results.json").read_text())
                self.assertEqual(report["final"]["videos"], 2)
                self.assertFalse(report["record_value"])
                self.assertFalse(report["canonical_eligible"])
                self.assertEqual(
                    report["video_recipe"]["profile"], "captured_provider_video_v1"
                )
                exported = (
                    root
                    / "continued"
                    / ("finetune_model.pt" if adaptation == "finetune" else "probe.pt")
                )
                self.assertTrue(exported.is_file())
                original = (root / "first/resume.pt").read_bytes()
                with self.assertRaises(FileExistsError):
                    ssv2.run(cfg, root / "first")
                self.assertEqual((root / "first/resume.pt").read_bytes(), original)

    def test_invalid_video_recipe_inputs_fail_before_data_access(self):
        for field, value in (
            ("num_frames", 0),
            ("num_frames", True),
            ("image_size", 0),
            ("num_workers", -1),
            ("max_train_samples", -1),
            ("max_val_samples", True),
            ("epochs", 51),
            ("max_steps_per_epoch", 1),
        ):
            cfg = self.config()
            cfg["probe"][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises((ValueError, ssv2.ConfigError)),
            ):
                ssv2.validate_config(cfg)
        cfg = self.config()
        cfg.pop("optimizer_profile")
        with self.assertRaises((ValueError, ssv2.ConfigError)):
            ssv2.validate_config(cfg)
        cfg = self.config()
        cfg.pop("adaptation")
        with self.assertRaises((ValueError, ssv2.ConfigError)):
            ssv2.validate_config(cfg)

    def fixture(self, root):
        fixtures.tiny_ssv2(root)
        path = root / "labels/train.json"
        entries = json.loads(path.read_text())
        path.write_text(json.dumps((entries * 3)[:5]))

    def test_all_real_model_routes_keep_expected_training_scope(self):
        from tests import test_method_hf_basic5 as hf
        from tests import test_method_patch_basic5 as patch

        if not (REAL and hf.HAVE and patch.HAVE):
            self.skipTest("full encoder dependencies required")
        from tests.test_method_cradiov4_h import TestRadio
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega

        for kind in image_ft.PROFILES:
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
                for adaptation in ("frozen", "attentive", "finetune"):
                    with (
                        self.subTest(kind=kind, adaptation=adaptation),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        root = Path(tmp)
                        data = root / "data"
                        self.fixture(data)
                        cfg = self.config(data, adaptation, kind)
                        cfg["probe"]["batch_size"] = 1
                        cfg["execution"] = {
                            "profile": "captured_video_v1",
                            "accumulation_steps": 2,
                            "precision": "fp32",
                            "tail_policy": "discard",
                        }
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
                        if cfg["backbone"].get("arch") == "released":
                            cfg["probe"].update(image_size=224, num_frames=16)
                        original = ssv2.FrozenFrameAverageClassifier
                        models = []
                        states = []

                        def build(
                            *args,
                            original=original,
                            models=models,
                            states=states,
                            **kwargs,
                        ):
                            model = original(*args, **kwargs)
                            models.append(model)
                            states.append(copy.deepcopy(model.state_dict()))
                            return model

                        with mock.patch.object(
                            ssv2, "FrozenFrameAverageClassifier", build
                        ):
                            ssv2.run(cfg, root / "out")
                        model = models[0]
                        initial = states[0]
                        changed = any(
                            not torch.equal(v, initial["backbone." + k])
                            for k, v in model.backbone.state_dict().items()
                        )
                        self.assertEqual(changed, adaptation == "finetune")
                        self.assertFalse(
                            torch.equal(
                                model.classifier.weight, initial["classifier.weight"]
                            )
                        )
                        self.assertTrue(all(p.grad is None for p in model.parameters()))
                        report = json.loads((root / "out/results.json").read_text())
                        self.assertEqual(report["execution"]["updates"], 2)
                        self.assertEqual(
                            report["execution"]["discarded_microbatches"], 1
                        )
            finally:
                if owner:
                    owner.doCleanups()

    def test_two_rank_video_resume_and_unpadded_validation(self):
        if not REAL:
            self.skipTest("real encoder dependencies required")
        from tests.test_method_imagenet_execution import Execution

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "data")
            val = root / "data/labels/validation.json"
            entries = json.loads(val.read_text())
            val.write_text(json.dumps((entries * 2)[:3]))
            for adaptation in ("frozen", "attentive", "finetune"):
                cfg = self.config(root / "data", adaptation)
                cfg["probe"].update(batch_size=1, epochs=3)
                cfg["execution"] = {
                    "profile": "captured_video_v1",
                    "accumulation_steps": 2,
                    "precision": "fp32",
                    "tail_policy": "discard",
                }
                (root / (adaptation + ".json")).write_text(json.dumps(cfg))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--master-addr=127.0.0.1",
                    f"--master-port={port}",
                    "--nproc-per-node=2",
                    "--module",
                    "tests." + Path(__file__).stem,
                    "--worker",
                    tmp,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            try:
                output, _ = proc.communicate(timeout=150)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                output, _ = proc.communicate(timeout=10)
                self.fail("worker timeout: " + output[-10000:])
            self.assertEqual(proc.returncode, 0, output[-10000:])
            for adaptation in ("frozen", "attentive", "finetune"):
                full = torch.load(
                    root / (adaptation + "-full/resume.pt"), weights_only=True
                )
                resumed = torch.load(
                    root / (adaptation + "-continued/resume.pt"), weights_only=True
                )
                Execution().equal_tree(full, resumed)
                self.assertEqual(len(full["rng"]), 2)
                self.assertEqual(full["runtime"]["updates"], 3)
                self.assertEqual(full["runtime"]["discarded_microbatches"], 3)
                report = json.loads(
                    (root / (adaptation + "-continued/results.json")).read_text()
                )
                self.assertEqual(report["final"]["videos"], 3)
                self.assertEqual(report["execution"]["effective_batch"], 4)

    def test_invalid_execution_and_collectives_are_refused(self):
        from downstream import video_execution

        cfg = self.config()
        cfg["execution"] = {
            "profile": "captured_video_v1",
            "accumulation_steps": 2,
            "precision": "fp32",
            "tail_policy": "discard",
        }
        for key, value in (
            ("profile", "wrong"),
            ("accumulation_steps", True),
            ("accumulation_steps", 0),
            ("precision", "fp16"),
            ("tail_policy", "flush"),
            ("unexpected", 0),
        ):
            bad = copy.deepcopy(cfg)
            bad["execution"][key] = value
            with (
                self.subTest(key=key),
                self.assertRaises((ValueError, ssv2.ConfigError)),
            ):
                ssv2.validate_config(bad)
        for value in ("", None, False):
            with self.assertRaises((ValueError, ssv2.ConfigError)):
                ssv2.validate_config(cfg | {"resume": value})
        for field in ("video_recipe", "execution"):
            bad = cfg | {"resume": "/checkpoint.pt"}
            bad.pop(field)
            with self.assertRaises(ssv2.ConfigError):
                ssv2.validate_config(bad)
        with mock.patch.dict(
            os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
        ):
            self.assertEqual(video_execution.resolve(cfg)["effective_batch"], 8)
            cfg["backbone"]["kind"] = "vggt_omega"
            with self.assertRaisesRegex(ValueError, "distributed"):
                video_execution.resolve(cfg)
        for count, segments in ((0, 3), (2, 0), (True, 3)):
            from downstream import video_recipe

            with self.assertRaises(ValueError):
                video_recipe.sample_indices(count, segments, True, {})
        recipe = video_recipe.resolve(self.config())
        with self.assertRaisesRegex(ValueError, "FT"):
            video_recipe.mix(
                torch.zeros(2, 2, 3, 4, 4),
                torch.zeros(2, dtype=torch.long),
                174,
                recipe,
            )
        with self.assertRaisesRegex(ValueError, "clips"):
            video_recipe.mix(
                torch.zeros(2, 3, 4, 4), torch.zeros(2, dtype=torch.long), 174, recipe
            )

    def test_resume_identity_and_invalid_labels_fail_closed(self):
        if not REAL:
            self.skipTest("real encoder dependencies required")
        from downstream import video_execution

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "data")
            cfg = self.config(root / "data", "finetune")
            cfg["execution"] = {
                "profile": "captured_video_v1",
                "accumulation_steps": 2,
                "precision": "fp32",
                "tail_policy": "discard",
            }
            ssv2.run(cfg, root / "first")
            source = root / "first/resume.pt"
            original = source.read_bytes()
            cfg["resume"] = str(source)
            cfg["probe"]["epochs"] = 2
            for section, key, value in (
                ("probe", "num_frames", 3),
                ("probe", "max_train_samples", 4),
                ("execution", "accumulation_steps", 1),
                ("execution", "precision", "bf16"),
            ):
                bad = copy.deepcopy(cfg)
                bad[section][key] = value
                with (
                    self.subTest(key=key),
                    self.assertRaises((ValueError, RuntimeError)),
                ):
                    ssv2.run(bad, root / key)
            annotation = root / "data/labels/train.json"
            saved = annotation.read_bytes()
            entries = json.loads(saved)
            annotation.write_text(json.dumps(entries[1:] + entries[:1]))
            with self.assertRaisesRegex(ValueError, "identity"):
                ssv2.run(cfg, root / "changed-order")
            annotation.write_bytes(saved)
            video = next((root / "data").rglob("*.webm"))
            video.write_bytes(video.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "identity"):
                ssv2.run(cfg, root / "changed-bytes")
            self.assertEqual(source.read_bytes(), original)
            labels = root / "data/labels/labels.json"
            values = json.loads(labels.read_text())
            values[next(iter(values))] = "174"
            labels.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "class mapping"):
                video_execution.data(cfg, ssv2)


class Delivery(unittest.TestCase):
    def test_ci_runs_video_execution(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if not HAVE_YAML:
            self.skipTest("CI YAML parser required")
        command = next(
            s["run"]
            for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if s.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        self.assertTrue(
            _runs_finetune_tests(command, module="tests.test_method_ssv2_execution")
        )

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "docs").is_dir(), "docs unavailable"
    )
    def test_documented_example_selects_video_profile(self):
        path = Path(__file__).resolve().parents[1] / "docs/examples/ssv2_execution.json"
        self.assertTrue(path.is_file())
        cfg = json.loads(path.read_text())
        self.assertEqual(cfg["video_recipe"], "captured_provider_video_v1")
        self.assertEqual(cfg["execution"]["profile"], "captured_video_v1")
        self.assertEqual(cfg["probe"]["epochs"], 50)
        self.assertEqual(cfg["probe"]["num_frames"], 16)
        if HAVE:
            ssv2.validate_config(cfg)


def worker(root):
    root = Path(root)
    from downstream import extended_distributed as distributed

    torch.set_num_threads(1)
    with distributed.session("cpu"):
        for adaptation in ("frozen", "attentive", "finetune"):
            base = json.loads((root / (adaptation + ".json")).read_text())
            for name, epochs, resume in (
                ("full", 3, None),
                ("first", 1, None),
                ("continued", 3, root / (adaptation + "-first/resume.pt")),
            ):
                cfg = copy.deepcopy(base)
                cfg["probe"]["epochs"] = epochs
                if resume:
                    cfg["resume"] = str(resume)
                rc = distributed.cli(
                    json.dumps(cfg).encode(),
                    root / (adaptation + "-" + name),
                    ssv2.run,
                    ssv2.TASK,
                )
                if rc:
                    raise RuntimeError("video worker failed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        unittest.main()
