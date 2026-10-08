"""Dense epoch continuation must reproduce uninterrupted portable training."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import test_method_dense_execution as fixtures

HAVE = fixtures.HAVE and all(
    importlib.util.find_spec(name) for name in ("timm", "einops")
)


@unittest.skipUnless(HAVE, "dense runner and tiny encoder dependencies required")
class Continuation(unittest.TestCase):
    def test_reference_schedule_and_worker_rng_survive_resume(self):
        import torch

        from downstream import ade20k, optimization

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            cfg, patch = self.setup_case(
                ade20k,
                fixtures.ade_config,
                lambda root: fixtures.tiny_ade(root, per=8),
                base / "data",
                "frozen",
            )
            cfg["scheduler_profile"] = optimization.REFERENCE_SCHEDULE
            cfg["execution"]["accumulation_steps"] = 8
            cfg["probe"].update(epochs=3, num_workers=1)
            with patch:
                ade20k.run(cfg, base / "full")
                short = copy.deepcopy(cfg)
                short["probe"]["epochs"] = 1
                ade20k.run(short, base / "first")
                ade20k.run(
                    dict(cfg, resume=str(base / "first/resume.pt")), base / "second"
                )
            full = torch.load(base / "full/resume.pt", weights_only=True)
            actual = torch.load(base / "second/resume.pt", weights_only=True)
            self.assert_state_equal(full, actual)
            self.assertEqual(full["scheduler"]["last_epoch"], 3)
            self.assertNotEqual(
                full["scheduler"]["_last_lr"], full["scheduler"]["base_lrs"]
            )

    def assert_state_equal(self, left, right):
        import torch

        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def setup_case(self, api, factory, fixture, root, adaptation):
        from scipy.io import savemat

        from downstream import spatial_backbones

        fixture(root)
        if api is fixtures.nyuv2:
            savemat(
                root / "labeled/splits.mat",
                {"trainNdxs": [[1], [2], [3]], "testNdxs": [[4], [5], [6]]},
            )
        cfg = fixtures.Dense().config(api, factory, root, adaptation=adaptation)
        spec = dict(cfg["backbone"], kind="vit")
        name = (
            "build_trainable_backbone"
            if adaptation == "finetune"
            else "build_frozen_backbone"
        )
        builder = getattr(spatial_backbones, name)

        def build(_spec, device):
            value = builder(spec, device)
            if adaptation == "finetune":
                value.finetune_group_policy = lambda: (
                    1,
                    {
                        n: (0, False)
                        for n, p in value.named_parameters()
                        if p.requires_grad
                    },
                )
            return value

        return cfg, mock.patch.object(api, name, build)

    def test_all_three_tasks_and_adaptations_resume_exactly(self):
        import torch

        for api, factory, fixture in fixtures.Dense().cases():
            for adaptation in ("frozen", "attentive", "finetune"):
                with (
                    self.subTest(task=api.TASK, adaptation=adaptation),
                    tempfile.TemporaryDirectory() as d,
                ):
                    base = Path(d)
                    cfg, patch = self.setup_case(
                        api, factory, fixture, base / "data", adaptation
                    )
                    key = "detector" if api is fixtures.coco else "probe"
                    cfg[key]["epochs"] = 2
                    for name in ("full", "first", "second"):
                        (base / name).mkdir()
                    with patch:
                        expected = api.run(cfg, base / "full")
                        self.assertTrue(
                            (base / "full/resume.pt").is_file(),
                            "execution must publish an epoch checkpoint",
                        )
                        short = copy.deepcopy(cfg)
                        short[key]["epochs"] = 1
                        api.run(short, base / "first")
                        resumed = dict(cfg, resume=str(base / "first/resume.pt"))
                        actual = api.run(resumed, base / "second")
                    self.assertEqual(expected, actual)
                    read = lambda name, base=base: torch.load(
                        base / name / "resume.pt", weights_only=True
                    )
                    full, continued = read("full"), read("second")
                    self.assert_state_equal(full, continued)
                    self.assertEqual(full["epoch"], 2)
                    prefix = "backbone.body." if api is fixtures.coco else "backbone."
                    self.assertEqual(
                        any(k.startswith(prefix) for k in full["model"]),
                        adaptation == "finetune",
                    )
                    if api is fixtures.coco:
                        self.assertTrue(
                            any(k.startswith("backbone.fpn.") for k in full["model"])
                        )
                    report = json.loads((base / "second/results.json").read_text())
                    self.assertEqual(report["continuation"]["start_epoch"], 1)
                    self.assertFalse(report["record_value"])
                    self.assertFalse(report["continuation"]["legacy_native_compatible"])

    def test_existing_output_is_preserved_by_runner_and_cli(self):
        for api, factory, _ in fixtures.Dense().cases():
            with self.subTest(task=api.TASK), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                cfg = fixtures.Dense().config(api, factory)
                out = root / "out"
                out.mkdir()
                sentinel = out / "run_manifest.json"
                sentinel.write_text("original")
                config = root / "cfg.json"
                config.write_text(json.dumps(cfg))
                with self.assertRaisesRegex(FileExistsError, "empty"):
                    api.run(cfg, out)
                self.assertNotEqual(
                    api.main(["--config", str(config), "--out", str(out)]), 0
                )
                self.assertEqual(sentinel.read_text(), "original")
                self.assertEqual(
                    sorted(p.name for p in out.iterdir()), ["run_manifest.json"]
                )

    def test_resume_requires_execution_and_nonempty_path(self):
        for api, factory, _ in fixtures.Dense().cases():
            cfg = fixtures.Dense().config(api, factory)
            api.validate_config(dict(cfg, resume="checkpoint.pt"))
            for value in (None, "", " ", 123):
                with (
                    self.subTest(task=api.TASK, value=value),
                    self.assertRaises(api.ConfigError),
                ):
                    api.validate_config(dict(cfg, resume=value))
            cfg.pop("execution")
            with self.assertRaisesRegex(api.ConfigError, "execution"):
                api.validate_config(dict(cfg, resume="checkpoint.pt"))

    def test_resume_rejects_changed_inputs_config_encoder_and_bad_checkpoints(self):
        import torch

        from downstream import ade20k

        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            cfg, patch = self.setup_case(
                ade20k, fixtures.ade_config, fixtures.tiny_ade, base / "data", "frozen"
            )
            cfg["probe"]["epochs"] = 1
            with patch:
                ade20k.run(cfg, base / "first")
                blob = torch.load(base / "first/resume.pt", weights_only=True)
                resumed = copy.deepcopy(cfg)
                resumed["probe"]["epochs"] = 2
                resumed["resume"] = str(base / "first/resume.pt")
                for index, (field, value) in enumerate(
                    (
                        ("format", "native"),
                        ("epoch", 3),
                        ("identity", "wrong"),
                        ("rng", []),
                        ("model", {}),
                        ("optimizer", {}),
                        ("runtime", {}),
                        ("scheduler", {}),
                    )
                ):
                    changed = dict(blob, **{field: value})
                    checkpoint = base / f"bad-{index}.pt"
                    torch.save(changed, checkpoint)
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        ade20k.run(
                            dict(resumed, resume=str(checkpoint)),
                            base / f"bad-out-{index}",
                        )
                with self.assertRaisesRegex(ValueError, "identity"):
                    ade20k.run(dict(resumed, seed=cfg["seed"] + 1), base / "seed")
                original_builder = ade20k.build_frozen_backbone

                def altered_encoder(spec, device):
                    model = original_builder(spec, device)
                    with torch.no_grad():
                        next(model.parameters()).add_(1)
                    return model

                with (
                    mock.patch.object(ade20k, "build_frozen_backbone", altered_encoder),
                    self.assertRaisesRegex(ValueError, "identity"),
                ):
                    ade20k.run(resumed, base / "encoder")
                with self.assertRaisesRegex(ValueError, "identity"):
                    altered = copy.deepcopy(resumed)
                    altered["probe"]["image_size"] += 16
                    ade20k.run(altered, base / "size")
                for index, pattern in enumerate(
                    (
                        "images/training/*.jpg",
                        "images/validation/*.jpg",
                        "annotations/training/*.png",
                        "annotations/validation/*.png",
                    )
                ):
                    path = next((base / "data").glob(pattern))
                    original = path.read_bytes()
                    path.write_bytes(original + b"changed")
                    with (
                        self.subTest(input=pattern),
                        self.assertRaisesRegex(ValueError, "identity"),
                    ):
                        ade20k.run(resumed, base / f"input-{index}")
                    path.write_bytes(original)
                # An unchanged checkpoint at the same target remains an evaluation-only restart.
                equal = dict(cfg, resume=resumed["resume"])
                expected = json.loads((base / "first/results.json").read_text())[
                    "final"
                ]
                self.assertEqual(ade20k.run(equal, base / "equal"), expected)
                self.assert_state_equal(
                    blob, torch.load(base / "equal/resume.pt", weights_only=True)
                )

    def test_membership_tracks_selected_order_and_all_native_data(self):
        from torch.utils.data import Subset

        from downstream import ade20k, coco, dense_resume, nyuv2

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for api, factory, fixture in fixtures.Dense().cases():
                data = root / api.TASK
                cfg, _ = self.setup_case(api, factory, fixture, data, "frozen")
                if api is ade20k:
                    train = ade20k.ADE20kSegmentation(data, "training", 32)
                    val = ade20k.ADE20kSegmentation(data, "validation", 32)
                    changed = train.images[0]
                elif api is coco:
                    train = coco.CocoDetectionForFRCNN(
                        str(data / "images/train2017"),
                        str(data / "annotations/instances_train2017.json"),
                    )
                    val = coco.CocoDetectionForFRCNN(
                        str(data / "images/val2017"),
                        str(data / "annotations/instances_val2017.json"),
                    )
                    changed = data / "annotations/instances_train2017.json"
                else:
                    train = nyuv2.NYUv2Depth(
                        data / "labeled/nyu_depth_v2_labeled.mat", [0, 1, 2], 32
                    )
                    val = nyuv2.NYUv2Depth(
                        data / "labeled/nyu_depth_v2_labeled.mat", [3, 4, 5], 32
                    )
                    changed = data / "labeled/splits.mat"
                before = dense_resume.membership(cfg, train, val)
                reorder = dense_resume.membership(
                    cfg, Subset(train, list(reversed(range(len(train))))), val
                )
                self.assertNotEqual(before["train"], reorder["train"])
                self.assertEqual(before["val"], reorder["val"])
                original = changed.read_bytes()
                changed.write_bytes(original + b"changed")
                self.assertNotEqual(before, dense_resume.membership(cfg, train, val))
                changed.write_bytes(original)
                self.assertEqual(before, dense_resume.membership(cfg, train, val))
                self.assertNotIn(str(root), json.dumps(before))


class Delivery(unittest.TestCase):
    def test_ci_executes_dense_continuation(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed

        if not HAVE_YAML or not WORKFLOWS.is_dir():
            self.skipTest("workflow source and YAML required")
        commands = [
            step["run"]
            for step in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
            if step.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        ]
        self.assertEqual(len(commands), 1)
        self.assertTrue(
            _runs_finetune_tests(commands[0], module="tests.test_dense_continuation")
        )
