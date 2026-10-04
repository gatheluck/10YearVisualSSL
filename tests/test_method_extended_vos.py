"""First-mask video segmentation: independent metrics, geometry and execution."""

import importlib.util
import unittest

HAVE = all(
    importlib.util.find_spec(n) for n in ("torch", "torchvision", "numpy", "PIL")
)


@unittest.skipUnless(HAVE, "downstream dependencies required")
class TestComponents(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.extended_vos"),
            "portable DAVIS execution is missing",
        )
        import torch

        from downstream import extended_vos

        torch.set_num_threads(1)
        self.api = extended_vos

    def test_mask_head_occupancy_correlation_and_updates(self):
        import torch
        from torch.nn import functional as F

        torch.manual_seed(7)
        head = self.api.MaskHead(3, width=5)
        a, b = torch.randn(2, 3, 2, 2), torch.randn(2, 3, 3, 3)
        first = torch.zeros(2, 1, 8, 8)
        first[:, :, 1, 1] = 1
        ta, sb = F.normalize(head.proj(a), dim=1), F.normalize(head.proj(b), dim=1)
        weights = F.interpolate(first, size=(2, 2), mode="area")
        desc = (ta * weights).sum((-1, -2), keepdim=True) / weights.sum(
            (-1, -2), keepdim=True
        )
        corr = sb * desc
        expected = F.interpolate(
            head.mask(corr), size=(8, 8), mode="bilinear", align_corners=False
        )
        got, score = head(a, b, first)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        torch.testing.assert_close(score, corr.sum(1), rtol=0, atol=0)
        self.api.mask_loss(got, first).backward()
        self.assertGreater(head.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.mask.weight.grad.abs().sum().item(), 0)
        for bad in (first * 0, first + 2, first * float("nan"), first[:, 0]):
            with self.assertRaises(ValueError):
                head(a, b, bad)

    def test_pair_freezing_normalization_and_adapter_gradient(self):
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torchvision.transforms.functional import normalize

        class Body(nn.Module):
            out_channels = 4

            def __init__(self):
                super().__init__()
                self.body = nn.Conv2d(3, 4, 1)
                self.inputs = []

            def forward_features(self, x):
                self.inputs.append(x.detach().clone())
                return self.body(F.adaptive_avg_pool2d(x, (2, 2)))

        for profile in (None, "captured_single_block_v1", "captured_cross_self_v1"):
            body = Body()
            model = self.api.Probe(body, profile)
            model.train()
            x = torch.rand(1, 3, 16, 16, requires_grad=True)
            mask = torch.ones(1, 1, 16, 16)
            prediction, _ = model(x, x, mask)
            self.api.mask_loss(prediction, mask).backward()
            self.assertFalse(body.training)
            self.assertTrue(
                all(not p.requires_grad and p.grad is None for p in body.parameters())
            )
            self.assertIsNone(x.grad)
            torch.testing.assert_close(
                body.inputs[0],
                normalize(x.detach(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                rtol=0,
                atol=0,
            )
            if profile:
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.adapter.parameters()
                    )
                )

    def test_captured_void_evaluation_is_explicitly_background(self):
        import numpy as np

        target = np.zeros((64, 64), np.uint8)
        target[10, 10] = 1
        target[10, 11] = 255
        pred = target.copy()
        pred[10, 11] = 1
        score = self.api.VideoScore()
        score.add(
            "void",
            [target] * 3,
            [
                np.where(target == 255, 0, target),
                pred,
                np.where(target == 255, 0, target),
            ],
        )
        self.assertEqual(score.result()["J"], 50.0)

    def test_bce_masks_void_without_diluting_gradient(self):
        import torch

        x = torch.tensor([[[[0.0, 1.0, -2.0]]]], requires_grad=True)
        y = torch.tensor([[[[1.0, 255.0, 0.0]]]])
        loss = self.api.mask_loss(x, y)
        expected = (
            torch.nn.functional.softplus(-x[0, 0, 0, 0])
            + torch.nn.functional.softplus(x[0, 0, 0, 2])
        ) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(x.grad[0, 0, 0, 1], 0)
        for bad in (y * 0 + 255, y * 0 + 0.5, y * float("nan"), y[:, 0]):
            with self.assertRaises(ValueError):
                self.api.mask_loss(x, bad)
        with self.assertRaises(ValueError):
            self.api.mask_loss(x * float("nan"), y)

    def test_boundary_iou_and_empty_masks(self):
        import numpy as np

        g = np.zeros((64, 64), bool)
        g[20:25, 20:25] = True
        p = np.roll(g, 1, axis=1)
        j, f = self.api.frame_score(g, p)
        self.assertAlmostEqual(j, 2 / 3)
        self.assertEqual(f, 1.0)
        large = np.zeros((256, 256), bool)
        large[80:130, 80:130] = True
        self.assertEqual(self.api.frame_score(large, np.roll(large, 3, axis=1))[1], 1.0)
        self.assertEqual(self.api.frame_score(g * False, g * False), (1.0, 1.0))
        self.assertEqual(self.api.frame_score(g, g * False), (0.0, 0.0))
        for bad in (g.astype(float) + 0.5, g[0], g[:0]):
            with self.assertRaises(ValueError):
                self.api.frame_score(g, bad)

    def test_object_macro_reduction_excludes_both_endpoint_frames(self):
        import numpy as np

        first = np.zeros((64, 64), np.uint8)
        first[5:20, 5:20] = 1
        two = first.copy()
        two[35:50, 35:50] = 4
        score = self.api.VideoScore()
        score.add("one", [first] * 3, [first, first * 0, first * 0])
        score.add("two", [two] * 4, [two * 0, two, two, two * 0])
        result = score.result()
        self.assertAlmostEqual(result["J"], 200 / 3)
        self.assertAlmostEqual(result["F"], 200 / 3)
        self.assertAlmostEqual(result["J_and_F"], 200 / 3)
        self.assertEqual(result["objects"], 3)
        self.assertEqual(result["scored_frames"], 3)
        with self.assertRaises(ValueError):
            score.add("one", [first] * 3, [first] * 3)
        for pred in (
            [first] * 2,
            [first] * 4,
            [first, first.astype(float), first],
            [first, first + 8, first],
        ):
            with self.assertRaises(ValueError):
                self.api.VideoScore().add("bad", [first] * 3, pred)
        with self.assertRaises(ValueError):
            self.api.VideoScore().result()

    def test_stream_only_uses_first_mask_and_has_deterministic_background_ties(self):
        import numpy as np
        import torch

        frames = [np.zeros((7, 9, 3), np.uint8) + i for i in range(4)]
        first = np.zeros((7, 9), np.uint8)
        first[1:3, 1:3] = 8
        first[4:6, 5:7] = 2
        first[0, 0] = 255
        calls = []

        def fwd(a, b, m):
            calls.append((a.clone(), b.clone(), m.clone()))
            return torch.ones(1, 1, 224, 224), torch.zeros(1, 14, 14)

        masks = list(self.api.mask_stream(frames, first, fwd))
        self.assertEqual(len(masks), 4)
        self.assertEqual(masks[0][0, 0], 0)
        self.assertTrue(all((m == 2).all() for m in masks[1:]))
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(torch.equal(c[0], calls[0][0]) for c in calls))
        zero = lambda a, b, m: (torch.zeros(1, 1, 224, 224), None)
        self.assertTrue(
            all(
                not m.any() for m in list(self.api.mask_stream(frames, first, zero))[1:]
            )
        )
        for bad in (
            lambda a, b, m: torch.ones(1, 4),
            lambda a, b, m: (torch.ones(1, 1, 223, 224), None),
            lambda a, b, m: (torch.ones(1, 1, 224, 224) * float("nan"), None),
        ):
            with self.assertRaises(ValueError):
                list(self.api.mask_stream(frames, first, bad))


@unittest.skipUnless(HAVE, "downstream dependencies required")
class TestData(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.assertIsNotNone(
            importlib.util.find_spec("downstream.vos_data"),
            "native sequence membership is missing",
        )
        from downstream import vos_data

        self.api = vos_data
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        import numpy as np
        from PIL import Image

        for split, name, base in [("train", "learn", 20), ("val", "held", 100)]:
            path = self.root / "ImageSets/2017" / f"{split}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + "\n")
            for i in range(3):
                a = self.root / "JPEGImages/480p" / name / f"{i:05d}.jpg"
                a.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((8, 12, 3), base + i, np.uint8)).save(a)
                mask = np.zeros((8, 12), np.uint8)
                mask[1:3, 1:3] = 2
                mask[4:6, 7:9] = 7
                mask[0, 0] = 255
                m = self.root / "Annotations/480p" / name / f"{i:05d}.png"
                m.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(m)
        return self.api.build_membership(self.root)

    def save_load(self, obj):
        import json

        path = self.root / "samples.json"
        path.write_text(json.dumps(obj))
        return self.api.load_data(self.root, path, "davis2017")

    def test_native_split_pairs_geometry_void_and_membership_identity(self):
        import numpy as np
        import torch
        from PIL import Image

        obj = self.fixture()
        train, val, meta = self.save_load(obj)
        self.assertEqual((len(train), len(val)), (6, 1))
        a, b, occupancy, target = train[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(a.shape, (3, 224, 224))
        self.assertEqual(target.shape, (1, 224, 224))
        self.assertTrue((target == 255).any())
        raw = np.array(Image.open(self.root / obj["train"][0]["masks"][0]["path"]))
        self.assertAlmostEqual(
            occupancy.mean().item(), float((raw == 2).mean()), places=3
        )
        self.assertEqual(set(target.unique().tolist()), {0.0, 1.0, 255.0})
        self.assertFalse(meta["official_membership_authenticated"])
        changed = self.root / obj["train"][0]["masks"][1]["path"]
        raw[3, 3] = 2
        Image.fromarray(raw).save(changed)
        _, _, after = self.save_load(obj)
        self.assertNotEqual(meta["assets_sha256"], after["assets_sha256"])

    def test_unknown_objects_palette_shapes_and_missing_frames_rejected(self):
        import numpy as np
        from PIL import Image

        obj = self.fixture()
        p = self.root / obj["train"][0]["masks"][1]["path"]
        original = p.read_bytes()
        for array in (
            np.full((8, 12), 3, np.uint8),
            np.zeros((9, 12), np.uint8),
            np.zeros((8, 12, 3), np.uint8),
        ):
            Image.fromarray(array).save(p)
            with self.assertRaises(ValueError):
                self.save_load(obj)
        p.write_bytes(original)
        p.unlink()
        with self.assertRaises(ValueError):
            self.api.build_membership(self.root)

    def test_cross_split_names_and_images_must_be_independent(self):
        import copy

        obj = self.fixture()
        dup = copy.deepcopy(obj)
        dup["validation"][0]["name"] = "learn"
        with self.assertRaises(ValueError):
            self.save_load(dup)
        src = self.root / obj["train"][0]["frames"][0]["path"]
        dst = self.root / obj["validation"][0]["frames"][0]["path"]
        dst.write_bytes(src.read_bytes())
        with self.assertRaises(ValueError):
            self.save_load(obj)

    def test_schema_path_escape_and_incomplete_sequence_rejected(self):
        import copy

        obj = self.fixture()
        changes = [
            lambda x: x.update(schema_version=True),
            lambda x: x.update(split_evidence=""),
            lambda x: x["train"][0]["frames"][0].update(path="../escape.jpg"),
            lambda x: x["train"][0]["frames"].pop(),
            lambda x: x["train"][0]["masks"].__setitem__(1, x["train"][0]["masks"][0]),
        ]
        for change in changes:
            bad = copy.deepcopy(obj)
            change(bad)
            with self.assertRaises(ValueError):
                self.save_load(bad)
        (self.root / "ImageSets/2017/train.txt").write_text("learn\nlearn\n")
        with self.assertRaises(ValueError):
            self.api.build_membership(self.root)

    def test_full_resolution_fallback_requires_both_directories(self):
        self.fixture()
        for parent in ("JPEGImages", "Annotations"):
            (self.root / parent / "480p").rename(self.root / parent / "Full-Resolution")
        other = self.api.build_membership(self.root)
        self.assertIn("Full-Resolution", other["train"][0]["frames"][0]["path"])
        self.save_load(other)
        (self.root / "JPEGImages/480p").mkdir()
        self.assertEqual(self.api.build_membership(self.root), other)

    def test_native_cli_preserves_existing_manifest(self):
        import json

        expected = self.fixture()
        out = self.root / "native.json"
        self.assertEqual(
            self.api.main(["--data-root", str(self.root), "--out", str(out)]), 0
        )
        self.assertEqual(json.loads(out.read_text()), expected)
        before = out.read_bytes()
        with self.assertRaises(FileExistsError):
            self.api.main(["--data-root", str(self.root), "--out", str(out)])
        self.assertEqual(out.read_bytes(), before)


@unittest.skipUnless(
    HAVE and importlib.util.find_spec("transformers"), "encoder dependencies required"
)
class TestExecution(unittest.TestCase):
    fixture = TestData.fixture
    save_load = TestData.save_load

    def setUp(self):
        TestData.setUp(self)
        from downstream import extended_vos

        self.assertTrue(
            callable(getattr(extended_vos, "run", None)),
            "DAVIS training/evaluation integration missing",
        )
        self.runner = extended_vos

    def prepare(self, adaptation="frozen"):
        from tests.test_method_extended_structured import TestExecution as Structured

        Structured.encoder(self)
        self.save_load(self.fixture())
        cfg = Structured.config(self, "davis2017", adaptation)
        cfg.update(task="extended_vos", transform_profile="captured_first_mask_v1")
        cfg["probe"]["batch_size"] = 2
        return cfg

    def test_lp_ap_training_continuation_and_frozen_export(self):
        import copy
        import json

        import torch

        from tests.test_method_extended_detection_training import TestDetectionTraining

        for adaptation in ("frozen", "attentive"):
            cfg = self.prepare(adaptation)
            full = copy.deepcopy(cfg)
            full["probe"]["epochs"] = 2
            out = self.root / (adaptation + "-full")
            out.mkdir()
            self.runner.run(full, out)
            first = self.root / (adaptation + "-first")
            first.mkdir()
            self.runner.run(cfg, first)
            resumed = copy.deepcopy(full)
            resumed["resume"] = str(first / "resume.pt")
            last = self.root / (adaptation + "-resume")
            last.mkdir()
            self.runner.run(resumed, last)
            one = torch.load(out / "resume.pt", weights_only=True)
            two = torch.load(last / "resume.pt", weights_only=True)
            for key in ("model", "optimizer", "scheduler", "runtime", "rng"):
                TestDetectionTraining().equal_tree(one[key], two[key])
            self.assertFalse(any(k.startswith("backbone.") for k in one["model"]))
            report = json.loads((out / "results.json").read_text())
            self.assertFalse(report["canonical_eligible"])
            self.assertFalse(report["record_value"])
            self.assertEqual(report["final"]["objects"], 2)
            self.assertEqual(report["final"]["scored_frames"], 1)
            self.assertEqual(report["execution"]["updates"], 6)
            a = torch.load(first / "probe.pt", weights_only=True)
            self.assertTrue(any(not torch.equal(a[k], one["model"][k]) for k in a))
            for key in ("J", "F", "J_and_F"):
                self.assertTrue(0 <= report["final"][key] <= 100)

    def test_exact_sequence_population_and_target_independence(self):
        import copy

        import torch

        self.prepare()
        obj = self.fixture()
        extra = copy.deepcopy(obj["validation"][0])
        extra["name"] = "second-held"
        obj["validation"].append(extra)
        _, val, _ = self.save_load(obj)
        fwd = lambda a, b, m: (torch.zeros(1, 1, 224, 224), None)
        result = self.runner.evaluate(val, range(len(val)), fwd, "cpu")
        self.assertEqual(result["J_and_F"], 0)
        self.assertEqual(result["sequences"], 2)
        for indices in ([], [0], [0, 0], [0, 1, 0], [2], [-1]):
            with self.assertRaises(ValueError):
                self.runner.evaluate(val, indices, fwd, "cpu")

    def test_recipe_expansion_and_invalid_configuration(self):
        cfg = self.prepare()
        recipe = self.runner.recipe("davis2017", "frozen")
        self.assertEqual(recipe["epochs"], 30)
        self.assertEqual(recipe["warmup"], "1 epoch from 1e-6")
        self.assertEqual(recipe["schedule"], "cosine to 0")
        self.assertEqual(recipe["weight_decay"], 0.0001)
        self.assertEqual(
            self.runner.recipe("davis2017", "attentive")["weight_decay"], 0.05
        )
        self.runner.validate_config(cfg)
        for change in (
            {"dataset": "lasot"},
            {"transform_profile": "crop"},
            {"adaptation": "finetune"},
            {"reader_profile": "guessed"},
            {"seed": 1},
        ):
            with self.assertRaises(ValueError):
                self.runner.validate_config(dict(cfg, **change))

    def test_cli_manifest_and_output_protection(self):
        import json

        cfg = self.prepare()
        p = self.root / "run.json"
        p.write_text(json.dumps(cfg))
        out = self.root / "out"
        self.assertEqual(self.runner.main(["--config", str(p), "--out", str(out)]), 0)
        from downstream import contract

        self.assertEqual(contract.verify(out, p, 0), (True, []))
        before = (out / "results.json").read_bytes()
        self.assertNotEqual(
            self.runner.main(["--config", str(p), "--out", str(out)]), 0
        )
        self.assertEqual((out / "results.json").read_bytes(), before)

    @unittest.skipUnless(
        all(importlib.util.find_spec(n) for n in ("timm", "einops")),
        "five-provider fixture dependencies required",
    )
    def test_all_five_providers_lp_ap_frozen_gradients_and_normalization(self):
        import torch

        from downstream import spatial_backbones as sb
        from tests.test_method_hf_basic5 import TestVisionFamilies
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega

        owners = [TestVisionFamilies(), TestK7Backbone(), TestOmega()]
        for owner in owners:
            owner.setUp()
            self.addCleanup(owner.doCleanups)
        specs = [owners[0].fixture(k)[0] for k in ("dinov3_hf", "siglip2_g")] + [
            owners[1].fixture()[0],
            owners[2].fixture()[0],
        ]
        specs.append(
            {
                "kind": "vjepa2_1",
                "arch": "vit_giant_xformers",
                "encoder": "",
                "img_size": 384,
                "patch_size": 16,
                "embed_dim": 48,
                "depth": 12,
                "num_heads": 4,
            }
        )
        for spec in specs:
            provider = sb._load_provider(sb.discover_providers()[spec["kind"]])
            for ap in (False, True):
                with self.subTest(kind=spec["kind"], ap=ap):
                    backbone = sb.build_frozen_backbone(spec, torch.device("cpu"))
                    model = self.runner.Probe(
                        backbone, provider.EXTENDED_DENSE_READER if ap else None
                    )
                    model.train()
                    self.assertFalse(backbone.training)
                    inputs = []
                    h = backbone.register_forward_pre_hook(
                        lambda _m, args, seen=inputs: seen.append(args[0])
                    )
                    # Providers use forward_features directly: inspect that exact boundary.
                    from unittest.mock import patch

                    x = torch.rand(1, 3, 224, 224)
                    m = torch.ones(1, 1, 224, 224)
                    original = backbone.forward_features

                    def capture(value, seen=inputs, call=original):
                        seen.append(value.detach())
                        return call(value)

                    with patch.object(
                        backbone, "forward_features", side_effect=capture
                    ):
                        pred, _ = model(x, x, m)
                    h.remove()
                    self.assertEqual(pred.shape, (1, 1, 224, 224))
                    from torchvision.transforms.functional import normalize

                    torch.testing.assert_close(
                        inputs[0],
                        normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        rtol=0,
                        atol=0,
                    )
                    self.runner.mask_loss(pred, m).backward()
                    self.assertTrue(
                        all(
                            p.grad is None and not p.requires_grad
                            for p in backbone.parameters()
                        )
                    )
                    self.assertGreater(
                        model.head.mask.weight.grad.abs().sum().item(), 0
                    )
                    if ap:
                        self.assertTrue(
                            any(
                                p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.adapter.parameters()
                            )
                        )

    def test_two_rank_worker_resume(self):
        import json
        import os
        import socket
        import subprocess
        import sys

        import torch

        from tests.test_method_extended_detection_training import TestDetectionTraining

        cfg = self.prepare("attentive")
        cfg["probe"].update(batch_size=1, num_workers=1)
        cfg["execution"] = {
            "accumulation_steps": 2,
            "precision": "fp32",
            "tail_policy": "discard",
        }
        for name, epochs, resume in [
            ("first", 1, None),
            ("full", 2, None),
            ("resumed", 2, str(self.root / "first/resume.pt")),
        ]:
            cfg["probe"]["epochs"] = epochs
            if resume:
                cfg["resume"] = resume
            path = self.root / "cfg.json"
            path.write_text(json.dumps(cfg))
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            result = subprocess.run(
                check=False,
                args=[
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--master-addr=127.0.0.1",
                    "--master-port=" + str(port),
                    "--nproc-per-node=2",
                    "--module",
                    "downstream.extended_vos",
                    "--config",
                    str(path),
                    "--out",
                    str(self.root / name),
                ],
                capture_output=True,
                text=True,
                timeout=180,
                env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        a = torch.load(self.root / "full/resume.pt", weights_only=True)
        b = torch.load(self.root / "resumed/resume.pt", weights_only=True)
        TestDetectionTraining().equal_tree(a, b)
        self.assertEqual(len(b["rng"]), 2)
        self.assertEqual(b["runtime"]["discarded_microbatches"], 2)
        report = json.loads((self.root / "resumed/results.json").read_text())
        self.assertEqual(report["final"]["sequences"], 1)


class TestDelivery(unittest.TestCase):
    def test_guides_examples_and_ci_deliver_all_three_paths(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for name in ("EXTENDED_VOS.md", "EXTENDED_SEMANTIC_INPUTS.md"):
            path = root / "docs" / name
            self.assertTrue(path.is_file(), name + " is missing")
            self.assertIn("(SUBMISSION_SCOPE.md)", path.read_text())
        for name, dataset, task, profile in [
            ("extended_vos", "davis2017", "extended_vos", "captured_first_mask_v1"),
            (
                "extended_context",
                "pascal_context",
                "extended_semantic_segmentation",
                "captured_fixed224_v1",
            ),
            (
                "extended_mapillary",
                "mapillary_vistas",
                "extended_semantic_segmentation",
                "captured_fixed224_v1",
            ),
        ]:
            cfg = json.loads((root / "docs/examples" / f"{name}.json").read_text())
            self.assertEqual(
                (cfg["dataset"], cfg["task"], cfg["transform_profile"]),
                (dataset, task, profile),
            )
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if HAVE_YAML:
            for module in (
                "tests.test_method_extended_vos",
                "tests.test_context_membership",
                "tests.test_mapillary_membership",
            ):
                self.assertTrue(
                    any(
                        _runs_finetune_tests(step.get("run", ""), module=module)
                        for workflow in parsed().values()
                        for step in workflow.get("jobs", {})
                        .get("downstream", {})
                        .get("steps", [])
                    )
                )
