"""Flow coordinates, valid support, source heads and executable continuation."""

import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(n) for n in ("torch", "torchvision", "h5py"))


@unittest.skipUnless(HAVE, "downstream flow dependencies required")
class TestFlow(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.extended_flow"),
            "Extended flow execution is missing",
        )
        import torch

        from downstream import extended_flow

        torch.set_num_threads(1)
        self.api = extended_flow

    @staticmethod
    def flo(array):
        return (
            struct.pack("<fii", 202021.25, array.shape[1], array.shape[0])
            + array.astype("<f4").tobytes()
        )

    def test_native_decoding_and_strict_invalid_targets(self):
        import h5py
        import numpy as np

        flow = np.ones((2, 3, 2), dtype=np.float32)
        flow[0, 0] = 1e10
        a, valid = self.api.decode_flo(self.flo(flow))
        np.testing.assert_array_equal(a, flow)
        self.assertEqual(valid.sum(), 5)
        for raw in (
            b"",
            self.flo(flow)[:-1],
            self.flo(flow) + b"x",
            b"xxxx" + self.flo(flow)[4:],
            self.flo(flow * 1e10),
        ):
            with self.assertRaises(ValueError):
                self.api.decode_flo(raw)
        for extra in (False, True):
            raw = io.BytesIO()
            with h5py.File(raw, "w") as f:
                f["flow"] = np.ones((4, 6, 2), dtype=np.float32)
                f["valid"] = np.ones((4, 6), dtype=np.uint8)
                if extra:
                    f["occlusion"] = np.zeros((4, 6))
            if extra:
                with self.assertRaises(ValueError):
                    self.api.decode_flo5(raw.getvalue())
            else:
                a, m = self.api.decode_flo5(raw.getvalue())
                self.assertEqual(a.shape, (4, 6, 2))
                self.assertTrue(m.all())

    def test_resize_uses_image_pixels_and_requires_full_valid_support(self):
        import numpy as np
        import torch

        f = np.ones((4, 6, 2), dtype=np.float32)
        f[..., 1] = 2
        v = np.ones((4, 6), dtype=bool)
        t, m = self.api.resize_flow(f, v, size=(8, 12), image_hw=(2, 3))
        torch.testing.assert_close(t[0], torch.full((8, 12), 4.0), rtol=0, atol=0)
        torch.testing.assert_close(t[1], torch.full((8, 12), 8.0), rtol=0, atol=0)
        self.assertTrue(m.all())
        f[1, 1] = float("nan")
        v[1, 1] = False
        t, m = self.api.resize_flow(f, v, size=(8, 12), image_hw=(2, 3))
        self.assertFalse(m[2:4, 2:4].any())
        self.assertEqual(int(m.sum()), 80)
        self.assertTrue(torch.isfinite(t).all())
        self.assertEqual(torch.count_nonzero(t[:, ~m]), 0)
        for a, b in [
            (f, np.zeros_like(v)),
            (f, np.ones((1, 1))),
            (np.zeros(f.shape, dtype="int32"), v),
        ]:
            with self.assertRaises(ValueError):
                self.api.resize_flow(a, b)
        with self.assertRaises(ValueError):
            self.api.resize_flow(f, v, image_hw=(0, 3))

    def test_l1_loss_and_global_metrics_keep_coordinate_units_separate(self):
        import torch

        target = torch.ones((2, 2, 1, 2))
        valid = torch.tensor([[[True, False]], [[True, True]]])
        pred = target.clone()
        pred[:, 0] += 3
        pred[:, 1] += 4
        pred.requires_grad_()
        loss = self.api.flow_loss(pred, target, valid)
        self.assertEqual(loss.item(), 3.5)
        loss.backward()
        self.assertEqual(pred.grad[0, :, 0, 1].abs().sum(), 0)
        score = self.api.FlowScore()
        score.update(
            pred.detach(),
            target,
            valid,
            {
                "rendering": ["clean", "final"],
                "native_hw": torch.tensor([[2, 4], [1, 2]]),
            },
        )
        r = score.result()
        self.assertEqual(r["epe"], 5)
        self.assertEqual(r["epe_clean"], 5)
        self.assertEqual(r["pixels"], 3)
        self.assertEqual(r["outlier_gt1px_at224_pct"], 100)
        self.assertAlmostEqual(r["epe_native_pixels_on_224_grid"], 20 / 3)
        boundary = self.api.FlowScore()
        boundary.update(
            target + torch.tensor([1.0, 0.0]).view(1, 2, 1, 1),
            target,
            valid,
            {"rendering": ["", ""], "native_hw": torch.tensor([[1, 2], [1, 2]])},
        )
        self.assertEqual(boundary.result()["outlier_gt1px_at224_pct"], 0)
        unequal = self.api.FlowScore()
        changed = pred.detach().clone()
        changed[1] = target[1] + 2 * (changed[1] - target[1])
        unequal.update(
            changed,
            target,
            valid,
            {
                "rendering": ["clean", "final"],
                "native_hw": torch.tensor([[2, 4], [1, 2]]),
            },
        )
        self.assertAlmostEqual(unequal.result()["epe"], 25 / 3)
        self.assertEqual(unequal.result()["epe_native_pixels_on_224_grid"], 10)
        for p, t, v in [
            (pred, target, valid.float()),
            (pred * float("nan"), target, valid),
            (pred, target, valid & False),
            (pred[:1], target, valid),
        ]:
            with self.assertRaises(ValueError):
                self.api.flow_loss(p, t, v)
        with self.assertRaises(ValueError):
            self.api.FlowScore().result()
        empty = self.api.FlowScore()
        empty.update(
            target,
            target * 0,
            valid,
            {"rendering": ["", ""], "native_hw": torch.tensor([[1, 2], [1, 2]])},
        )
        with self.assertRaises(ValueError):
            empty.result()

    def test_head_initialization_difference_order_freeze_and_updates(self):
        import torch
        from torch import nn
        from torch.nn import functional as F

        from downstream.attention import SpatialAdapter
        from downstream.captured_readers import (
            CROSS_SELF,
            SINGLE_BLOCK,
            SingleBlockSpatialAdapter,
        )

        class Body(nn.Module):
            out_channels = 8

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 1)

            def forward_features(self, x):
                return self.conv(F.adaptive_avg_pool2d(x, (2, 2)))

        for profile in (None, SINGLE_BLOCK, CROSS_SELF):
            torch.manual_seed(3)
            body = Body()
            torch.manual_seed(8)
            model = self.api.FlowProbe(body, profile)
            torch.manual_seed(8)
            adapter = (
                (
                    SingleBlockSpatialAdapter
                    if profile == SINGLE_BLOCK
                    else SpatialAdapter
                )(8)
                if profile
                else None
            )
            proj = nn.Conv2d(8, 128, 1)
            head = nn.Conv2d(128, 2, 1)
            x = torch.rand(2, 3, 8, 8)
            y = torch.rand_like(x)

            def feat(z, body=body, adapter=adapter):
                z = (
                    z - z.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                ) / z.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                out = body.forward_features(z).detach()
                return adapter(out) if adapter else out

            expected = F.interpolate(
                head(proj(feat(y)) - proj(feat(x))),
                size=(5, 7),
                mode="bilinear",
                align_corners=False,
            )
            actual = model(x, y, (5, 7))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            model.train()
            self.assertFalse(body.training)
            self.assertTrue(all(not p.requires_grad for p in body.parameters()))
            before = {n: p.clone() for n, p in model.named_parameters()}
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            actual.square().mean().backward()
            opt.step()
            self.assertTrue(all(p.grad is None for p in body.parameters()))
            self.assertFalse(
                torch.equal(before["head.flow.weight"], model.head.flow.weight)
            )
            if profile:
                self.assertTrue(
                    any(
                        not torch.equal(before[n], p)
                        for n, p in model.named_parameters()
                        if n.startswith("adapter.")
                    )
                )

    def fixture(self, root):
        import numpy as np
        from PIL import Image

        rows = []
        for i in range(4):
            Image.fromarray(np.full((4, 6, 3), 20 + i * 30, dtype=np.uint8)).save(
                root / f"a{i}.png"
            )
            Image.fromarray(np.full((4, 6, 3), 80 + i * 20, dtype=np.uint8)).save(
                root / f"b{i}.png"
            )
            (root / f"f{i}.flo").write_bytes(
                self.flo(np.ones((4, 6, 2), dtype=np.float32) * (i + 1))
            )
            rows.append(
                {
                    "a": f"a{i}.png",
                    "b": f"b{i}.png",
                    "flow": f"f{i}.flo",
                    "scene": str(i),
                    "rendering": "",
                }
            )
        data = {
            "schema_version": 1,
            "dataset": "middlebury_flow",
            "split_evidence": "explicit reduced fixture",
            "train": rows[:2],
            "validation": rows[2:],
        }
        (root / "samples.json").write_text(json.dumps(data))
        return data

    def test_membership_data_hashes_and_invalid_splits(self):
        import copy

        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.fixture(root)
            train, val, meta = self.api.load_data(
                root, root / "samples.json", "middlebury_flow"
            )
            a, _b, t, _m = train[0]
            self.assertEqual(a.shape, (3, 224, 224))
            self.assertEqual(t.shape, (2, 224, 224))
            torch.testing.assert_close(
                t[0], torch.full((224, 224), 224 / 6), rtol=1e-6, atol=0
            )
            self.assertEqual(val[0][-1]["native_hw"].tolist(), [4, 6])
            self.assertFalse(meta["official_membership_verified"])
            (root / "a0.png").write_bytes((root / "a1.png").read_bytes())
            self.assertNotEqual(
                meta,
                self.api.load_data(root, root / "samples.json", "middlebury_flow")[2],
            )
            for key, value in [
                ("a", "../escape.png"),
                ("scene", "0"),
                ("flow", "f0.flo"),
            ]:
                bad = copy.deepcopy(original)
                bad["validation"][0][key] = value
                (root / "samples.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    self.api.load_data(root, root / "samples.json", "middlebury_flow")

    def test_native_sintel_and_archive_middlebury_membership(self):
        import zipfile

        import numpy as np
        from PIL import Image

        self.assertTrue(
            callable(getattr(self.api, "build_manifest", None)),
            "native flow split conversion is missing",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scene in ("z", "a"):
                (root / "flow" / scene).mkdir(parents=True)
                (root / "flow" / scene / "frame_0001.flo").write_bytes(
                    self.flo(np.ones((4, 6, 2), dtype=np.float32))
                )
                for tag in ("clean", "final"):
                    (root / tag / scene).mkdir(parents=True)
                    for i in (1, 2):
                        Image.new("RGB", (6, 4), (i * 40, 20, 30)).save(
                            root / tag / scene / f"frame_{i:04}.png"
                        )
            data = self.api.build_manifest(root, "mpi_sintel")
            self.assertEqual([r["scene"] for r in data["train"]], ["a", "a"])
            self.assertEqual(
                [r["rendering"] for r in data["validation"]], ["clean", "final"]
            )
            p = root / "membership.json"
            p.write_text(json.dumps(data))
            self.assertEqual(len(self.api.load_data(root, p, "mpi_sintel")[0]), 2)
            (root / "flow" / "z" / "frame_0001.flo").unlink()
            with self.assertRaises(ValueError):
                self.api.build_manifest(root, "mpi_sintel")
            archives = root / "archives"
            archives.mkdir()
            image = io.BytesIO()
            Image.new("RGB", (6, 4), (10, 20, 30)).save(image, format="PNG")
            with zipfile.ZipFile(archives / "other-gt-flow.zip", "w") as z:
                for scene in ("z", "a"):
                    z.writestr(
                        f"gt/{scene}/flow10.flo",
                        self.flo(np.ones((4, 6, 2), dtype=np.float32)),
                    )
            with zipfile.ZipFile(archives / "other-color-allframes.zip", "w") as z:
                for scene in ("z", "a"):
                    for i in (10, 11):
                        z.writestr(f"color/{scene}/frame{i}.png", image.getvalue())
            data = self.api.build_manifest(root, "middlebury_flow")
            p.write_text(json.dumps(data))
            train, val, _ = self.api.load_data(root, p, "middlebury_flow")
            self.assertEqual((len(train), len(val)), (1, 1))
            self.assertEqual(val[0][0].shape, (3, 224, 224))
            with zipfile.ZipFile(archives / "other-color-allframes.zip", "w") as z:
                z.writestr("color/a/frame10.png", image.getvalue())
            with self.assertRaises(ValueError):
                self.api.build_manifest(root, "middlebury_flow")

    @staticmethod
    def config(root, adaptation="frozen"):
        return {
            "task": "extended_flow",
            "profile": "capture_extended_components",
            "dataset": "middlebury_flow",
            "seed": 0,
            "device": "cpu",
            "data_root": str(root),
            "samples": str(root / "samples.json"),
            "transform_profile": "captured_flow_input_pixels_v2",
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

    def test_training_cli_resume_and_contract(self):
        import subprocess
        import sys

        import torch

        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest("real encoder fixture requires transformers")
        from downstream import contract
        from tests.test_method_extended_detection_training import TestDetectionTraining

        self.assertTrue(
            callable(getattr(self.api, "run", None)), "flow execution is missing"
        )
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
                    {"dataset": "blinkvision"},
                    {"transform_profile": "guess"},
                    {"seed": 1},
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
                cfg["resume"] = str(one / "resume.pt")
                path = root / "config.json"
                path.write_text(json.dumps(cfg))
                out = root / (adaptation + "resumed")
                cmd = [
                    sys.executable,
                    "-m",
                    "downstream.extended_flow",
                    "--config",
                    str(path),
                    "--out",
                    str(out),
                ]
                proc = subprocess.run(
                    check=False, args=cmd, capture_output=True, text=True, timeout=120
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(contract.verify(out, path, 0), (True, []))
                TestDetectionTraining().equal_tree(
                    torch.load(full / "resume.pt", weights_only=True),
                    torch.load(out / "resume.pt", weights_only=True),
                )
                report = json.loads((out / "results.json").read_text())
                self.assertFalse(report["canonical_eligible"])
                self.assertFalse(report["record_value"])
                self.assertTrue(report["source_protocol_conflicts"])
                self.assertEqual(report["final"]["images"], 2)
                self.assertFalse(
                    any(
                        k.startswith("backbone.")
                        for k in torch.load(out / "probe.pt", weights_only=True)
                    )
                )
                self.assertNotEqual(
                    subprocess.run(
                        check=False, args=cmd, capture_output=True, timeout=120
                    ).returncode,
                    0,
                )

    def test_cross_split_rgb_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            data["validation"][0]["a"] = "a0.png"
            (root / "samples.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.api.load_data(root, root / "samples.json", "middlebury_flow")

    def test_spring_double_grid_data_and_bad_geometry(self):
        import h5py
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            data["dataset"] = "spring"
            for row in data["train"] + data["validation"]:
                row["flow"] += "5"
                with h5py.File(root / row["flow"], "w") as f:
                    f["flow"] = np.ones((8, 12, 2), dtype=np.float32)
            (root / "samples.json").write_text(json.dumps(data))
            train, _, _ = self.api.load_data(root, root / "samples.json", "spring")
            torch.testing.assert_close(
                train[0][2][0], torch.full((224, 224), 224 / 6), rtol=1e-6, atol=0
            )
            with h5py.File(root / data["train"][0]["flow"], "w") as f:
                f["flow"] = np.ones((7, 12, 2), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "geometry"):
                self.api.load_data(root, root / "samples.json", "spring")

    def test_ten_actual_provider_training_routes(self):
        try:
            import einops
            import timm
            from transformers import DINOv3ViTModel, SiglipVisionModel
        except ImportError:
            self.skipTest("five-provider fixtures require downstream dependencies")
        self.assertTrue(all((einops, timm, DINOv3ViTModel, SiglipVisionModel)))
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
                        self.assertEqual(report["final"]["pixels"], 2 * 224 * 224)

    def test_two_rank_worker_resume(self):
        import os
        import socket
        import subprocess
        import sys

        import torch

        try:
            from transformers import SiglipVisionConfig, SiglipVisionModel
        except ImportError:
            self.skipTest("distributed fixture requires transformers")
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
            cfg["probe"].update(batch_size=1, num_workers=1)
            for name, epochs, resume in (
                ("one", 1, None),
                ("full", 2, None),
                ("resumed", 2, str(root / "one/resume.pt")),
            ):
                cfg["probe"]["epochs"] = epochs
                if resume:
                    cfg["resume"] = resume
                path = root / "cfg.json"
                path.write_text(json.dumps(cfg))
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                p = subprocess.run(
                    check=False,
                    args=[
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--master-addr=127.0.0.1",
                        "--master-port=" + str(port),
                        "--nproc-per-node=2",
                        "--module",
                        "downstream.extended_flow",
                        "--config",
                        str(path),
                        "--out",
                        str(root / name),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                )
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            a = torch.load(root / "full/resume.pt", weights_only=True)
            b = torch.load(root / "resumed/resume.pt", weights_only=True)
            TestDetectionTraining().equal_tree(a, b)
            self.assertEqual(len(b["rng"]), 2)
            self.assertEqual(
                json.loads((root / "resumed/results.json").read_text())["final"][
                    "images"
                ],
                2,
            )

    def test_fail_closed_input_and_recipe_guards(self):
        import copy
        import zipfile

        import numpy as np
        import torch
        from torch import nn

        for dataset, adaptation in (("unknown", "frozen"), ("mpi_sintel", "finetune")):
            with self.assertRaises(ValueError):
                self.api.recipe(dataset, adaptation)
        for adaptation, decay in (("frozen", 0.0001), ("attentive", 0.05)):
            r = self.api.recipe("mpi_sintel", adaptation)
            self.assertEqual(
                (
                    r["epochs"],
                    r["base_lr"],
                    r["lr_reference_effective_batch"],
                    r["weight_decay"],
                ),
                (30, 0.001, 16, decay),
            )
            self.assertEqual(
                (r["warmup"], r["schedule"]), ("1 epoch from 1e-6", "cosine to 0")
            )
        with self.assertRaises(ValueError):
            self.api.FlowProbe(nn.Identity(), "guess")
        f = np.ones((2, 3, 2), dtype=np.float32)
        for mask in (np.ones((2, 3)) * 2, np.ones((2, 3)) * np.nan):
            with self.assertRaises(ValueError):
                self.api.resize_flow(f, mask)
        for bad in (f[0], f[:, :, :1], f[:0]):
            with self.assertRaises(ValueError):
                self.api.resize_flow(bad, np.ones(bad.shape[:2]))
        target = torch.ones((1, 2, 2, 3))
        valid = torch.ones((1, 2, 3), dtype=torch.bool)
        for meta in (
            {"rendering": ["guess"], "native_hw": torch.tensor([[2, 3]])},
            {"rendering": [""], "native_hw": torch.tensor([[0, 3]])},
            {"rendering": [""], "native_hw": torch.tensor([[float("nan"), 3]])},
        ):
            with self.assertRaises(ValueError):
                self.api.FlowScore().update(target, target, valid, meta)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.fixture(root)
            for key, value in (
                ("schema_version", True),
                ("dataset", "spring"),
                ("split_evidence", ""),
                ("train", []),
            ):
                bad = copy.deepcopy(original)
                bad[key] = value
                (root / "samples.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    self.api.load_data(root, root / "samples.json", "middlebury_flow")
            for bad_row in (
                {"scene": ""},
                {"rendering": "clean"},
                {"extra": 0},
                {"b": "a0.png"},
                {"a": "missing.png"},
            ):
                bad = copy.deepcopy(original)
                bad["train"][0].update(bad_row)
                (root / "samples.json").write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    self.api.load_data(root, root / "samples.json", "middlebury_flow")
            bad = copy.deepcopy(original)
            bad["train"].append(bad["train"][0])
            (root / "samples.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                self.api.load_data(root, root / "samples.json", "middlebury_flow")
            for ref in (
                {"archive": "x.zip"},
                {"archive": "x.zip", "member": "../x"},
                str(root / "a0.png"),
            ):
                with self.assertRaises(ValueError):
                    self.api._read(root, ref)
            with zipfile.ZipFile(root / "x.zip", "w") as z:
                z.writestr("a", "x")
            with self.assertRaises(ValueError):
                self.api._read(root, {"archive": "x.zip", "member": "missing"})
            for row in original["train"]:
                (root / row["flow"]).write_bytes(
                    self.flo(np.zeros((4, 6, 2), dtype=np.float32))
                )
            (root / "samples.json").write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "zero"):
                self.api.load_data(root, root / "samples.json", "middlebury_flow")


class TestDelivery(unittest.TestCase):
    def test_example_guide_and_ci_route(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(
            (root / "docs/EXTENDED_FLOW.md").is_file(), "flow guide missing"
        )
        cfg = json.loads((root / "docs/examples/extended_flow.json").read_text())
        self.assertEqual(cfg["task"], "extended_flow")
        if HAVE:
            from downstream.extended_flow import validate_config

            validate_config(cfg)
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed

        if HAVE_YAML and WORKFLOWS.exists():
            self.assertTrue(
                any(
                    _runs_finetune_tests(
                        step.get("run", ""), module="tests.test_method_extended_flow"
                    )
                    for workflow in parsed().values()
                    for step in workflow.get("jobs", {})
                    .get("downstream", {})
                    .get("steps", [])
                ),
                "flow suite missing from required downstream CI",
            )
