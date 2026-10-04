"""Extended depth must preserve its own loss, geometry and split semantics."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(
    importlib.util.find_spec(x) for x in ("torch", "torchvision", "h5py", "scipy")
)


@unittest.skipUnless(HAVE, "downstream depth dependencies required")
class TestDepth(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.extended_depth"),
            "Extended depth execution is missing",
        )
        import torch

        from downstream import extended_depth

        torch.set_num_threads(1)
        self.api = extended_depth

    def fixture(self, root, channels_first=False):
        import h5py
        import numpy as np
        from scipy.io import savemat

        images = np.arange(4 * 3 * 5 * 7, dtype=np.uint8).reshape(4, 3, 5, 7)
        depths = np.linspace(0.05, 11, 4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7)
        with h5py.File(root / "depth.mat", "w") as f:
            f["images"] = images.transpose(1, 2, 3, 0) if channels_first else images
            f["depths"] = depths.transpose(1, 2, 0) if channels_first else depths
        savemat(root / "splits.mat", {"trainNdxs": [[3], [1]], "testNdxs": [[4], [2]]})
        return images, depths

    def test_official_indices_geometry_and_paired_flip(self):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        for layout in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                images, depths = self.fixture(root, layout)
                train, val, meta = self.api.load_data(
                    root / "depth.mat", root / "splits.mat"
                )
                self.assertEqual(train.indices, [2, 0])
                self.assertEqual(val.indices, [3, 1])
                self.assertFalse(meta["official_membership_verified"])
                image, depth, valid = val[0]
                expected = TF.to_tensor(
                    Image.fromarray(images[3].transpose(1, 2, 0)).resize(
                        (224, 224), Image.Resampling.BICUBIC
                    )
                )
                target = torch.from_numpy(
                    np.array(
                        Image.fromarray(depths[3]).resize(
                            (224, 224), Image.Resampling.BILINEAR
                        )
                    )
                )
                torch.testing.assert_close(image, expected, rtol=0, atol=0)
                torch.testing.assert_close(depth, target, rtol=0, atol=0)
                self.assertTrue(torch.equal(valid, (target > 0.1) & (target < 10)))
                with patch("random.random", return_value=0):
                    flipped = train[0]
                with patch("random.random", return_value=1):
                    plain = train[0]
                for a, b in zip(flipped, plain):
                    torch.testing.assert_close(a, b.flip(-1), rtol=0, atol=0)

    def test_rejects_invalid_membership_and_geometry(self):
        import h5py
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            for train, test in [
                ([1, 1], [2, 3, 4]),
                ([1, 2], [2, 3]),
                ([0, 1], [2, 3]),
                ([1.5, 2], [3, 4]),
                ([1, 2], [3, 5]),
                ([], [1, 2, 3, 4]),
                ([1], [2, 3]),
            ]:
                savemat(root / "splits.mat", {"trainNdxs": train, "testNdxs": test})
                with self.assertRaises(ValueError):
                    self.api.load_data(root / "depth.mat", root / "splits.mat")
            self.fixture(root)
            with h5py.File(root / "depth.mat", "a") as f:
                del f["depths"]
                f["depths"] = [1, 2, 3, 4]
            with self.assertRaises(ValueError):
                self.api.load_data(root / "depth.mat", root / "splits.mat")

            import numpy as np

            for image_shape, depth_shape in [
                ((4, 3, 5), (4, 5, 7)),
                ((0, 3, 5, 7), (0, 5, 7)),
                ((4, 2, 5, 7), (4, 5, 7)),
                ((4, 3, 0, 7), (4, 0, 7)),
            ]:
                with h5py.File(root / "depth.mat", "w") as f:
                    f["images"] = np.zeros(image_shape)
                    f["depths"] = np.zeros(depth_shape)
                with self.assertRaises(ValueError):
                    self.api.load_data(root / "depth.mat", root / "splits.mat")

    def test_loss_is_full_log_variance_and_rejects_empty_or_nonfinite(self):
        import torch

        pred = torch.tensor([[[2.0, 8.0, 99.0]]], requires_grad=True)
        target = torch.tensor([[[1.0, 2.0, 0.0]]])
        valid = target > 0
        logs = torch.log(pred[valid]) - torch.log(target[valid])
        expected = logs.square().mean() - logs.mean().square()
        result = self.api.depth_loss(pred, target, valid)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertNotEqual(
            result.item(), (logs.square().mean() - 0.5 * logs.mean().square()).item()
        )
        result.backward()
        self.assertEqual(pred.grad[0, 0, 2], 0)
        for p, t, m in [
            (pred, target, torch.zeros_like(valid)),
            (pred * float("nan"), target, valid),
            (pred, target, valid.float()),
            (pred, target[:, :, :1], valid),
            (pred, torch.zeros_like(target), valid),
            (pred, target * float("nan"), valid),
            (pred, torch.full_like(target, float("inf")), valid),
            (pred.flatten(), target.flatten(), valid.flatten()),
            (pred[:0], target[:0], valid[:0]),
        ]:
            with self.assertRaises(ValueError):
                self.api.depth_loss(p, t, m)

    def test_float_rgb_invalid_values_and_input_hashes(self):
        import h5py
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            _, _, before = self.api.load_data(root / "depth.mat", root / "splits.mat")
            with h5py.File(root / "depth.mat", "a") as f:
                del f["images"]
                f["images"] = np.full((4, 3, 5, 7), 0.5, dtype=np.float32)
            train, _, after = self.api.load_data(
                root / "depth.mat", root / "splits.mat"
            )
            self.assertNotEqual(before["data_sha256"], after["data_sha256"])
            self.assertEqual(before["split_sha256"], after["split_sha256"])
            torch.testing.assert_close(
                train[0][0], torch.full((3, 224, 224), 127 / 255)
            )
            for value in (-1, 256, float("nan")):
                with h5py.File(root / "depth.mat", "a") as f:
                    f["images"][2, 0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "RGB"):
                    train[0]

    def test_metrics_pool_pixels_without_alignment_and_reject_empty(self):
        import torch

        score = self.api.DepthScore()
        score.update(
            torch.tensor([[[2.0]]]),
            torch.tensor([[[1.0]]]),
            torch.ones(1, 1, 1, dtype=torch.bool),
        )
        score.update(
            torch.tensor([[[2.0, 2.0, 200.0]]]),
            torch.tensor([[[2.0, 2.0, 100.0]]]),
            torch.ones(1, 1, 3, dtype=torch.bool),
        )
        result = score.result()
        self.assertEqual(result["rmse"], 0.5)
        self.assertEqual(result["absrel"], 0.25)
        self.assertEqual(result["delta1"], 75)
        self.assertEqual(result["pixels"], 4)
        score = self.api.DepthScore()
        score.update(
            torch.ones(1, 1, 1),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "valid"):
            score.result()
        with self.assertRaises(ValueError):
            score.update(
                torch.full((1, 1, 1), float("nan")),
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1, dtype=torch.bool),
            )

    def test_positive_mapping_precedes_resize_and_encoder_stays_frozen(self):
        import torch
        from torch import nn
        from torch.nn import functional as F

        class Body(nn.Module):
            out_channels = 8

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 1)

            def forward_features(self, x):
                return self.conv(x)

        for reader in (None, "captured_single_block_v1", "captured_cross_self_v1"):
            torch.manual_seed(2)
            model = self.api.DepthProbe(Body(), reader)
            x = torch.rand(2, 3, 2, 3)
            model.head.weight.data.mul_(100)
            with torch.no_grad():
                from torchvision.transforms.functional import normalize

                features = model.backbone.forward_features(
                    normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ).float()
                if model.adapter is not None:
                    features = model.adapter(features)
                raw = model.head(features)
                expected = F.interpolate(
                    F.softplus(raw) + 0.001,
                    size=(5, 7),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            torch.testing.assert_close(model(x, (5, 7)), expected, rtol=0, atol=0)
            before = copy.deepcopy(model.state_dict())
            opt = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad], lr=0.1
            )
            for _ in range(3):
                model.train()
                opt.zero_grad()
                y = model(x, (5, 7))
                self.api.depth_loss(
                    y,
                    torch.linspace(1, 3, y.numel()).reshape_as(y),
                    torch.ones_like(y, dtype=torch.bool),
                ).backward()
                opt.step()
            self.assertFalse(model.backbone.training)
            self.assertFalse(torch.equal(before["head.weight"], model.head.weight))
            self.assertTrue(
                all(
                    p.grad is None and not p.requires_grad
                    for p in model.backbone.parameters()
                )
            )
            if reader:
                self.assertFalse(
                    torch.equal(
                        before["adapter.output_projection.weight"],
                        model.adapter.output_projection.weight,
                    )
                )
        with self.assertRaises(ValueError):
            self.api.DepthProbe(Body(), "unknown")

    def test_raw_rgb_is_normalized_once_for_provider_contract(self):
        import torch
        from torch import nn

        class Body(nn.Module):
            out_channels = 3

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))

            def forward_features(self, x):
                self.seen = x.clone()
                return x * self.weight

        body = Body()
        self.api.DepthProbe(body, None)(torch.zeros(1, 3, 2, 2), (2, 2))
        expected = (
            torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
            .view(1, 3, 1, 1)
            .expand(1, 3, 2, 2)
        )
        torch.testing.assert_close(body.seen, expected)

    def config(self, root, adaptation="frozen"):
        return {
            "task": "extended_depth",
            "profile": "capture_extended_components",
            "dataset": "nyuv2",
            "seed": 0,
            "device": "cpu",
            "data_root": str(root / "depth.mat"),
            "samples": str(root / "splits.mat"),
            "transform_profile": "captured_nyu_fixed224_v1",
            "adaptation": adaptation,
            "reader_profile": "captured_single_block_v1"
            if adaptation == "attentive"
            else None,
            "backbone": {
                "kind": "siglip2_g",
                "arch": "fixture",
                "encoder": str(root / "encoder"),
                "img_size": 32,
                "patch_size": 16,
            },
            "probe": {"epochs": 1, "batch_size": 2, "num_workers": 0},
        }

    def test_cli_training_resume_matches_uninterrupted_and_protects_outputs(self):
        import subprocess
        import sys

        import torch

        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest("real encoder CLI fixture requires transformers")
        from downstream import contract

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b))
                for k in a:
                    equal(a[k], b[k])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                self.assertEqual(a, b)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            SiglipVisionModel(
                SiglipVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=32,
                    patch_size=16,
                )
            ).save_pretrained(root / "encoder")
            for adaptation in ("frozen", "attentive"):
                cfg = self.config(root, adaptation)
                for bad in (
                    {"dataset": "kitti"},
                    {"seed": 1},
                    {"transform_profile": "guess"},
                    {"reader_profile": "unknown"},
                    {"probe": {"epochs": 31, "batch_size": 2, "num_workers": 0}},
                ):
                    with self.assertRaises(ValueError):
                        self.api.validate_config(dict(cfg, **bad))
                one = root / (adaptation + "1")
                one.mkdir()
                self.api.run(cfg, one)
                cfg["probe"]["epochs"] = 2
                full = root / (adaptation + "full")
                full.mkdir()
                self.api.run(cfg, full)
                resumed = root / (adaptation + "resumed")
                cfg["resume"] = str(one / "resume.pt")
                config = root / "config.json"
                config.write_text(json.dumps(cfg))
                cmd = [
                    sys.executable,
                    "-m",
                    "downstream.extended_depth",
                    "--config",
                    str(config),
                    "--out",
                    str(resumed),
                ]
                proc = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=120
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(contract.verify(resumed, config, 0), (True, []))
                equal(
                    torch.load(full / "resume.pt", weights_only=True),
                    torch.load(resumed / "resume.pt", weights_only=True),
                )
                report = json.loads((resumed / "results.json").read_text())
                self.assertEqual(report["final"]["epochs"], 2)
                self.assertGreater(report["final"]["pixels"], 0)
                self.assertFalse(report["record_value"])
                self.assertFalse(report["canonical_eligible"])
                self.assertFalse(
                    any(
                        k.startswith("backbone.")
                        for k in torch.load(resumed / "probe.pt", weights_only=True)
                    )
                )
                self.assertNotEqual(
                    subprocess.run(
                        cmd, check=False, capture_output=True, timeout=120
                    ).returncode,
                    0,
                )

    def test_recipe_selects_only_verified_dataset_and_weight_decay(self):
        for adaptation in ("frozen", "attentive"):
            spec = self.api.recipe("nyuv2", adaptation)
            self.assertEqual(spec["epochs"], 30)
            self.assertEqual(
                spec["weight_decay"], 0.05 if adaptation == "attentive" else 0.0001
            )
            self.assertEqual(spec["betas"], [0.9, 0.999])
        for dataset, adaptation in [
            ("sun_rgbd", "frozen"),
            ("kitti", "frozen"),
            ("nyuv2", "finetune"),
        ]:
            with self.assertRaises(ValueError):
                self.api.recipe(dataset, adaptation)

    def test_ten_actual_provider_training_routes(self):
        try:
            import einops
            import timm
            from transformers import DINOv3ViTModel, SiglipVisionModel
        except ImportError:
            self.skipTest(
                "five-provider fixtures require timm, einops and transformers"
            )
        self.assertTrue(all((einops, timm, DINOv3ViTModel, SiglipVisionModel)))
        import torch

        from downstream import spatial_backbones as sb
        from tests.test_method_hf_basic5 import TestVisionFamilies
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega

        owners = [TestVisionFamilies(), TestK7Backbone(), TestOmega()]
        for owner in owners:
            owner.setUp()
            self.addCleanup(owner.doCleanups)
        specs = [owners[0].fixture(k)[0] for k in ("dinov3_hf", "siglip2_g")]
        specs += [owners[1].fixture()[0], owners[2].fixture()[0]]
        specs += [
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
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            for spec in specs:
                for adaptation in ("frozen", "attentive"):
                    with self.subTest(kind=spec["kind"], adaptation=adaptation):
                        cfg = self.config(root, adaptation)
                        cfg["backbone"] = spec
                        provider = sb._load_provider(
                            sb.discover_providers()[spec["kind"]]
                        )
                        cfg["reader_profile"] = (
                            provider.EXTENDED_DENSE_READER
                            if adaptation == "attentive"
                            else None
                        )
                        out = root / (spec["kind"] + adaptation)
                        out.mkdir()
                        self.api.run(cfg, out)
                        report = json.loads((out / "results.json").read_text())
                        self.assertEqual(report["execution"]["updates"], 1)
                        self.assertGreater(report["final"]["pixels"], 0)
                        self.assertEqual(report["metric_alignment"], "none")
                        self.assertFalse(report["canonical_eligible"])
                        saved = torch.load(out / "probe.pt", weights_only=True)
                        self.assertTrue(saved)
                        self.assertFalse(any(k.startswith("backbone.") for k in saved))

    def test_two_rank_epoch_resume_is_exact(self):
        import os
        import socket
        import subprocess
        import sys

        import torch

        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest("distributed encoder fixture requires transformers")
        from tests.test_method_extended_detection_training import TestDetectionTraining

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            SiglipVisionModel(
                SiglipVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=32,
                    patch_size=16,
                )
            ).save_pretrained(root / "encoder")
            cfg = self.config(root, "attentive")
            cfg["probe"]["batch_size"] = 1
            cfg["probe"]["num_workers"] = 1
            for name, epochs, resume in [
                ("one", 1, None),
                ("full", 2, None),
                ("resumed", 2, str(root / "one/resume.pt")),
            ]:
                cfg["probe"]["epochs"] = epochs
                if resume:
                    cfg["resume"] = resume
                path = root / "cfg.json"
                path.write_text(json.dumps(cfg))
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--master-addr=127.0.0.1",
                        "--master-port=" + str(port),
                        "--nproc-per-node=2",
                        "--module",
                        "downstream.extended_depth",
                        "--config",
                        str(path),
                        "--out",
                        str(root / name),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            a = torch.load(root / "full/resume.pt", weights_only=True)
            b = torch.load(root / "resumed/resume.pt", weights_only=True)
            TestDetectionTraining().equal_tree(a, b)
            self.assertEqual(len(b["rng"]), 2)
            report = json.loads((root / "resumed/results.json").read_text())
            self.assertEqual(report["final"]["images"], 2)


class TestDelivery(unittest.TestCase):
    def test_document_example_and_required_ci_route(self):
        root = Path(__file__).resolve().parents[1]
        doc = root / "docs/EXTENDED_DEPTH.md"
        example = root / "docs/examples/extended_depth.json"
        self.assertTrue(doc.is_file(), "depth guide missing")
        cfg = json.loads(example.read_text())
        self.assertEqual(cfg["task"], "extended_depth")
        if HAVE:
            from downstream.extended_depth import validate_config

            validate_config(cfg)
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed

        if HAVE_YAML and WORKFLOWS.exists():
            self.assertTrue(
                any(
                    _runs_finetune_tests(
                        step.get("run", ""), module="tests.test_method_extended_depth"
                    )
                    for workflow in parsed().values()
                    for step in workflow.get("jobs", {})
                    .get("downstream", {})
                    .get("steps", [])
                )
            )
