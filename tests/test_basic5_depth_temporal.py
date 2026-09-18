"""Captured depth and temporal behavior, including real runner integration."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAVE = all(importlib.util.find_spec(m) for m in
           ("torch", "torchvision", "h5py", "scipy", "timm", "av"))
if HAVE:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from downstream import nyuv2, ssv2
    from tests.test_downstream_nyuv2 import tiny_nyuv2, smoke_config
    from tests.test_downstream_ssv2 import tiny_ssv2, smoke_config as video_config

PROFILE = "capture_basic5_components"


@unittest.skipUnless(HAVE, "Depth/video integration needs downstream dependencies")
class TestDepthComponents(unittest.TestCase):
    def test_head_initialization_and_upsampling_before_softplus(self):
        self.assertTrue(hasattr(nyuv2, "Depth1x1Head"), "minimal depth head missing")
        torch.manual_seed(31)
        head = nyuv2.Depth1x1Head(3)
        torch.manual_seed(31)
        conv = torch.nn.Conv2d(3, 1, 1)
        torch.nn.init.normal_(conv.weight, std=.01)
        torch.nn.init.zeros_(conv.bias)
        torch.testing.assert_close(head.conv.weight, conv.weight, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 4)
        with torch.no_grad():
            head.conv.weight.fill_(2)
        x = (torch.arange(12.)-5.5).reshape(1, 3, 2, 2).requires_grad_()
        actual = head(x, (5, 7))
        expected = F.softplus(F.interpolate(head.conv(x), (5, 7),
                             mode="bilinear", align_corners=False)) + .001
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_silog_formula_mask_and_empty_gradient(self):
        self.assertTrue(hasattr(nyuv2, "silog_loss"), "captured loss missing")
        pred = torch.tensor([1., math.e, 99.], requires_grad=True)
        target = torch.tensor([1., 1., float("nan")])
        valid = torch.tensor([1, 1, 0])
        loss = nyuv2.silog_loss(pred, target, valid)
        self.assertAlmostEqual(float(loss), .375, places=6)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([-.25, .75/math.e, 0.]))
        empty = nyuv2.silog_loss(pred, target, torch.zeros(3))
        self.assertEqual(float(empty), 0)
        self.assertTrue(empty.requires_grad)
        clamped = nyuv2.silog_loss(torch.tensor([0.]), torch.ones(1), torch.ones(1))
        self.assertAlmostEqual(float(clamped), .5 * math.log(1e-6)**2, places=4)

    def test_depth_metrics_threshold_units_and_float32(self):
        self.assertTrue(hasattr(nyuv2, "depth_metrics"), "five depth metrics missing")
        pred = torch.tensor([1., 1.25, 1.5625, 1.953125, 50.])
        target = torch.ones(5)
        result = nyuv2.depth_metrics(pred, target, torch.tensor([1, 1, 1, 1, 0]))
        self.assertEqual([result[k] for k in ("delta1", "delta2", "delta3")],
                         [25., 50., 75.])
        self.assertAlmostEqual(result["abs_rel"], float((pred[:4]-1).mean()))
        self.assertAlmostEqual(result["rmse"], float(((pred[:4]-1)**2).mean().sqrt()))
        low = nyuv2.depth_metrics(pred.half(), target.half(), torch.ones(5))
        full = nyuv2.depth_metrics(pred.half().float(), target, torch.ones(5))
        self.assertEqual(low, full)
        self.assertTrue(all(v == 0 for v in nyuv2.depth_metrics(
            pred, target, torch.zeros(5)).values()))

    def test_official_split_ids_are_not_positional_and_formats_agree(self):
        self.assertTrue(hasattr(nyuv2, "load_official_splits"), "official split loader missing")
        import h5py
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "splits.mat"
            data = {"trainNdxs": np.array([[6], [2], [4]]),
                    "testNdxs": np.array([[1], [5], [3]])}
            savemat(path, data)
            self.assertEqual(nyuv2.load_official_splits(path, 6),
                             ([5, 1, 3], [0, 4, 2]))
            with h5py.File(path, "w") as f:
                for key, ids in data.items():
                    refs = f.create_dataset(key, (len(ids), 1), dtype=h5py.ref_dtype)
                    for i, value in enumerate(ids.reshape(-1)):
                        refs[i, 0] = f.create_dataset(f"{key}_{i}", data=[[value]]).ref
            self.assertEqual(nyuv2.load_official_splits(path, 6),
                             ([5, 1, 3], [0, 4, 2]))
            with h5py.File(path, "w") as f:
                for k, v in data.items():
                    f[k] = v
            self.assertEqual(nyuv2.load_official_splits(path, 6),
                             ([5, 1, 3], [0, 4, 2]))

    def test_bad_splits_are_refused_without_fallback(self):
        self.assertTrue(hasattr(nyuv2, "load_official_splits"), "official split loader missing")
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "split.mat"
            with self.assertRaises(FileNotFoundError):
                nyuv2.load_official_splits(p, 4)
            for train, test in (([1, 1], [3, 4]), ([1, 2], [2, 4]),
                                ([0, 2], [3, 4]), ([1, 5], [3, 4]),
                                ([1.5, 2], [3, 4]), ([1, 2], [3]),
                                ([], [1, 2, 3, 4])):
                savemat(p, {"trainNdxs": train, "testNdxs": test})
                with self.subTest(train=train, test=test), self.assertRaises(ValueError):
                    nyuv2.load_official_splits(p, 4)

    def test_valid_depth_bounds_and_joint_flip(self):
        import h5py
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.mat"
            with h5py.File(p, "w") as f:
                f["images"] = np.arange(3*4*4, dtype=np.uint8).reshape(1, 3, 4, 4)
                depths = np.ones((1, 4, 4), dtype=np.float32)
                depths[0, :, 0] = [.09, .1, 10., 10.1]
                f["depths"] = depths
            ds = nyuv2.NYUv2Depth(p, [0], 4, profile=PROFILE, train=True)
            with mock.patch.object(torch, "rand", return_value=torch.tensor([1.])):
                image, depth, valid = ds[0]
            self.assertEqual(valid[0, 0].tolist(), [0., 1., 1., 0.])
            with mock.patch.object(torch, "rand", return_value=torch.tensor([0.])):
                flipped = ds[0]
            for before, after in zip((image, depth, valid), flipped):
                torch.testing.assert_close(after, before.flip(-1), rtol=0, atol=0)
            ds.file.close()

    def test_depth_runner_uses_components_and_records_incomplete_recipe(self):
        from scipy.io import savemat
        from downstream import contract
        with tempfile.TemporaryDirectory() as d:
            root = tiny_nyuv2(Path(d) / "data")
            savemat(root / "labeled/splits.mat",
                    {"trainNdxs": [6, 2, 4], "testNdxs": [1, 5, 3]})
            cfg = smoke_config(root)
            cfg["profile"] = PROFILE
            cfg["probe"].update(max_train_samples=0, max_val_samples=0,
                                max_steps_per_epoch=0)
            out = Path(d) / "out"
            cfg_path = Path(d) / "config.json"
            cfg_path.write_text(json.dumps(cfg))
            with mock.patch.object(nyuv2, "silog_loss", wraps=nyuv2.silog_loss) as loss:
                self.assertEqual(nyuv2.main(["--config", str(cfg_path), "--out", str(out)]), 0)
            self.assertGreater(loss.call_count, 0)
            self.assertTrue(contract.verify(out, cfg_path, 0)[0])
            result = json.loads((out / "results.json").read_text())
            self.assertEqual(result["profile"], PROFILE)
            self.assertFalse(result["record_value"])
            self.assertFalse(result["subset_or_smoke"])
            self.assertFalse(result["canonical_eligible"])
            self.assertEqual(result["split_sha256"], contract.sha256_file(root / "labeled/splits.mat"))
            for key in ("delta1", "delta2", "delta3"):
                self.assertIn("nyuv2_" + key, json.loads((out / "metrics.json").read_text())["metrics"])

    def test_captured_evaluation_is_mean_of_batch_metrics(self):
        model = torch.nn.Identity()
        # Batch RMSEs are 0 and 3. Global pixel RMSE would be sqrt(3).
        loader = [(torch.ones(2, 1, 1, 1), torch.ones(2, 1, 1, 1), torch.ones(2, 1, 1, 1)),
                  (torch.ones(1, 1, 1, 1)*4, torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))]
        metrics = nyuv2.evaluate(model, loader, "cpu", profile=PROFILE)
        self.assertEqual(metrics["rmse"], 1.5)
        self.assertEqual(metrics["valid_pixels"], 3)
        self.assertEqual(metrics["delta1"], 50.)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            nyuv2.evaluate(model, [], "cpu", profile=PROFILE)
        bad = [(torch.full((1,), float("nan")), torch.ones(1), torch.ones(1))]
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            nyuv2.evaluate(model, bad, "cpu", profile=PROFILE)

    def test_captured_image_resize_matches_pil_reference(self):
        import h5py
        from PIL import Image
        from torchvision.transforms import functional as TF, InterpolationMode
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"tiny.mat"
            image = np.random.RandomState(1).randint(0, 256, (3, 35, 27), dtype=np.uint8)
            with h5py.File(p, "w") as f:
                f["images"] = image[None]
                f["depths"] = np.ones((1, 35, 27), dtype=np.float32)
            ds = nyuv2.NYUv2Depth(p, [0], 16, profile=PROFILE)
            actual = ds[0][0]
            expected = TF.to_tensor(TF.resize(Image.fromarray(image.transpose(2, 1, 0)),
                [16, 16], interpolation=InterpolationMode.BILINEAR))
            expected = (expected - nyuv2.IMAGENET_MEAN) / nyuv2.IMAGENET_STD
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            ds.file.close()

    def test_profile_typo_is_refused_in_both_runners(self):
        for mod, cfg in ((nyuv2, smoke_config(Path("/missing"))),
                         (ssv2, video_config(Path("/missing")))):
            cfg["profile"] = "capture_basic5_component"
            with self.assertRaisesRegex(mod.ConfigError, "profile"):
                mod.validate_config(cfg)

    def test_depth_gpu_update_keeps_backbone_frozen(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from downstream.spatial_backbones import build_frozen_backbone
        cfg = smoke_config(Path("/missing"))
        backbone = build_frozen_backbone(cfg["backbone"], torch.device("cuda"))
        model = nyuv2.FrozenDepthModel(backbone, profile=PROFILE).cuda().train()
        before = {k: v.clone() for k, v in backbone.state_dict().items()}
        old_head = model.head.conv.weight.detach().clone()
        opt = torch.optim.AdamW(model.head.parameters(), lr=.001)
        pred = model(torch.randn(2, 3, 64, 64, device="cuda"))
        nyuv2.silog_loss(pred, torch.ones_like(pred)*2, torch.ones_like(pred)).backward()
        opt.step()
        self.assertFalse(torch.equal(old_head, model.head.conv.weight))
        for k, v in backbone.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@unittest.skipUnless(HAVE, "Depth/video integration needs downstream dependencies")
class TestTemporalComponents(unittest.TestCase):
    def test_clip_geometry_is_shared_and_eval_is_center_crop(self):
        from PIL import Image
        from torchvision.transforms import RandomResizedCrop
        from torchvision.transforms import functional as TF
        with tempfile.TemporaryDirectory() as d:
            root = tiny_ssv2(Path(d)/"data")
            pixels = np.random.RandomState(7).randint(0, 256, (48, 80, 3), dtype=np.uint8)
            frame = Image.fromarray(pixels)
            for split in ("train", "validation"):
                ds = ssv2.SSV2Clips(root, split, 3, 32, profile=PROFILE)
                with mock.patch.object(ssv2, "decode_video_frames", return_value=[frame]*3), \
                        mock.patch.object(RandomResizedCrop, "get_params", return_value=(3, 5, 20, 30)) as crop:
                    clip, _ = ds[0]
                self.assertEqual(crop.call_count, 1 if split == "train" else 0)
                expected = (TF.resized_crop(frame, 3, 5, 20, 30, [32, 32])
                            if split == "train" else TF.center_crop(TF.resize(frame, 256), [32, 32]))
                expected = (TF.to_tensor(expected)-ssv2.IMAGENET_MEAN)/ssv2.IMAGENET_STD
                for actual in clip:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_segment_centers_short_clips_and_invalid_counts(self):
        self.assertTrue(hasattr(ssv2, "sample_segment_indices"), "segment sampler missing")
        self.assertEqual(ssv2.sample_segment_indices(16, 4, False), [1, 5, 9, 13])
        self.assertEqual(ssv2.sample_segment_indices(3, 5, False), [0, 1, 1, 2, 2])
        for n, k in ((0, 4), (4, 0), (-1, 4), (True, 4), (3.5, 4)):
            with self.subTest(n=n, k=k), self.assertRaises(ValueError):
                ssv2.sample_segment_indices(n, k, False)

    def test_random_sampling_is_seeded_and_within_each_segment(self):
        self.assertTrue(hasattr(ssv2, "sample_segment_indices"), "segment sampler missing")
        torch.manual_seed(44)
        a = [ssv2.sample_segment_indices(16, 4, True) for _ in range(20)]
        torch.manual_seed(44)
        b = [ssv2.sample_segment_indices(16, 4, True) for _ in range(20)]
        self.assertEqual(a, b)
        self.assertGreater(len(set(tuple(x) for x in a)), 1)
        for sample in a:
            for i, x in enumerate(sample):
                self.assertTrue(4*i <= x < 4*(i+1))

    def test_real_video_decode_uses_segment_centers(self):
        from tests.test_downstream_ssv2 import _write_webm
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.webm"
            _write_webm(p, 1, n_frames=16)
            all_frames = ssv2.decode_video_frames(p, 16)
            frames = ssv2.decode_video_frames(p, 4, profile=PROFILE, train=False)
            for actual, i in zip(frames, [1, 5, 9, 13]):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(all_frames[i]))

    def test_video_runner_profile_is_delivered_and_not_canonical(self):
        from downstream import contract
        with tempfile.TemporaryDirectory() as d:
            root = tiny_ssv2(Path(d) / "data")
            cfg = video_config(root)
            cfg["profile"] = PROFILE
            cfg["probe"].update(max_train_samples=0, max_val_samples=0,
                                max_steps_per_epoch=0)
            path, out = Path(d)/"config.json", Path(d)/"out"
            path.write_text(json.dumps(cfg))
            self.assertEqual(ssv2.main(["--config", str(path), "--out", str(out)]), 0)
            self.assertTrue(contract.verify(out, path, 0)[0])
            result = json.loads((out/"results.json").read_text())
            self.assertEqual(result["profile"], PROFILE)
            self.assertFalse(result["record_value"])
            self.assertFalse(result["subset_or_smoke"])
            self.assertFalse(result["canonical_eligible"])


if __name__ == "__main__":
    unittest.main()
