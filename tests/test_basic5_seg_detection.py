"""Behavioral contracts for captured segmentation/detection components."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
    import timm
    import pycocotools
    from PIL import Image
    from torchvision.transforms import functional as TF, RandomCrop
    from downstream import ade20k, coco, contract
    from tests.test_downstream_ade20k import tiny_ade, smoke_config as ade_config
    from tests.test_downstream_coco import tiny_coco, smoke_config as coco_config
    HAVE = True
except ImportError:
    HAVE = False

PROFILE = "capture_basic5_components"


@unittest.skipUnless(HAVE, "Segmentation/detection needs downstream dependencies")
class TestSegmentationGeometry(unittest.TestCase):
    def test_paired_scale_crop_flip_and_ignore_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root = tiny_ade(Path(d), per=1)
            arr = np.random.RandomState(5).randint(0, 256, (37, 59, 3), dtype=np.uint8)
            mask = np.resize(np.array([0, 1, 2, 150, 255], dtype=np.uint8), (37, 59))
            Image.fromarray(arr).save(root / "images/training/img0.jpg")
            Image.fromarray(mask).save(root / "annotations/training/img0.png")
            ds = ade20k.ADE20kSegmentation(root, "training", 16, profile=PROFILE)
            image = Image.open(ds.images[0]).convert("RGB")
            for scale in (0.5, 1.75):
                size = [max(16, round(37 * scale)), max(16, round(59 * scale))]
                with mock.patch.object(torch.Tensor, "uniform_", lambda t, a, b: t.fill_(scale)), \
                     mock.patch.object(RandomCrop, "get_params", return_value=(1, 2, 16, 16)) as crop, \
                     mock.patch.object(torch, "rand", return_value=torch.tensor([0.])):
                    got, target = ds[0]
                crop.assert_called_once()
                expected = TF.hflip(TF.crop(TF.resize(image, size), 1, 2, 16, 16))
                expected = (TF.to_tensor(expected) - ade20k.IMAGENET_MEAN) / ade20k.IMAGENET_STD
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                labels = np.array(TF.hflip(TF.crop(TF.resize(Image.fromarray(mask), size,
                    interpolation=TF.InterpolationMode.NEAREST), 1, 2, 16, 16)), dtype=np.int64)
                labels[labels == 0] = 255
                labels[labels != 255] -= 1
                np.testing.assert_array_equal(target.numpy(), labels)
                self.assertTrue(set(target.unique().tolist()) <= {255, 0, 1, 149})

    def test_eval_is_deterministic_and_legacy_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = tiny_ade(Path(d), per=1)
            captured = ade20k.ADE20kSegmentation(root, "validation", 32, profile=PROFILE)
            legacy = ade20k.ADE20kSegmentation(root, "validation", 32)
            with mock.patch.object(RandomCrop, "get_params", side_effect=AssertionError("eval crop")), \
                 mock.patch.object(torch, "rand", side_effect=AssertionError("eval RNG")):
                for got, expected in zip(captured[0], legacy[0]):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_real_ade_cli_records_incomplete_profile_without_subset(self):
        with tempfile.TemporaryDirectory() as d:
            root = tiny_ade(Path(d)/"data", per=2)
            cfg = ade_config(root)
            cfg["profile"] = PROFILE
            cfg["probe"].update(max_train_samples=0, max_val_samples=0)
            config, out = Path(d)/"cfg.json", Path(d)/"out"
            config.write_text(json.dumps(cfg))
            self.assertEqual(ade20k.main(["--config", str(config), "--out", str(out)]), 0)
            self.assertTrue(contract.verify(out, config, 0)[0])
            result = json.loads((out/"results.json").read_text())
            self.assertEqual(result["profile"], PROFILE)
            self.assertFalse(result["record_value"])
            self.assertFalse(result["canonical_eligible"])
            self.assertFalse(result["subset_or_smoke"])


@unittest.skipUnless(HAVE, "Segmentation/detection needs downstream dependencies")
class TestDetectionComponents(unittest.TestCase):
    def config(self, root=Path("/missing")):
        cfg = coco_config(root)
        cfg["profile"] = PROFILE
        cfg["detector"]["anchor_sizes"] = [8, 16, 32, 64]
        return cfg

    def test_flip_moves_boxes_with_pixels_and_keeps_other_fields(self):
        with tempfile.TemporaryDirectory() as d:
            root = tiny_coco(Path(d), per=1)
            args = (str(root/"images/train2017"), str(root/"annotations/instances_train2017.json"))
            ds = coco.CocoDetectionForFRCNN(*args, train=True, profile=PROFILE)
            plain = coco.CocoDetectionForFRCNN(*args)
            original, original_target = plain[0]
            with mock.patch.object(torch, "rand", return_value=torch.tensor([0.])):
                image, target = ds[0]
            torch.testing.assert_close(image, original.flip(-1), rtol=0, atol=0)
            self.assertEqual(target["boxes"].tolist(), [[32., 8., 56., 32.]])
            for k in ("labels", "area", "image_id", "iscrowd"):
                torch.testing.assert_close(target[k], original_target[k], rtol=0, atol=0)
            with mock.patch.object(torch, "rand", return_value=torch.tensor([1.])):
                for a, b in zip(ds[0][0], original):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            ev = coco.CocoDetectionForFRCNN(*args, train=False, profile=PROFILE)
            with mock.patch.object(torch, "rand", side_effect=AssertionError("eval RNG")):
                torch.testing.assert_close(ev[0][0], original, rtol=0, atol=0)

    def test_empty_boxes_remain_valid_after_flip(self):
        with tempfile.TemporaryDirectory() as d:
            root = tiny_coco(Path(d), per=1)
            p = root/"annotations/instances_train2017.json"
            data = json.loads(p.read_text()); data["annotations"] = []; p.write_text(json.dumps(data))
            ds = coco.CocoDetectionForFRCNN(str(root/"images/train2017"), str(p), train=True, profile=PROFILE)
            with mock.patch.object(torch, "rand", return_value=torch.tensor([0.])):
                _, target = ds[0]
            self.assertEqual(tuple(target["boxes"].shape), (0, 4))
            self.assertEqual(target["boxes"].dtype, torch.float32)

    def test_pyramid_values_gradients_and_order(self):
        self.assertTrue(hasattr(coco, "SimpleFeaturePyramid"), "trainable pyramid missing")
        pyramid = coco.SimpleFeaturePyramid(1, 1)
        for layer in (pyramid.lateral4, pyramid.lateral8, pyramid.lateral16, pyramid.lateral32):
            torch.nn.init.ones_(layer.weight); torch.nn.init.zeros_(layer.bias)
        x = torch.arange(24.).reshape(1, 1, 4, 6).requires_grad_()
        got = pyramid(x)
        self.assertEqual(list(got), ["0", "1", "2", "3"])
        expected = [torch.nn.functional.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
                    for s in (4., 2.)] + [x, torch.nn.functional.max_pool2d(x, 2, 2)]
        for a, b in zip(got.values(), expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(t.sum() for t in got.values()).backward()
        self.assertTrue((x.grad >= 21).all())
        self.assertTrue(all(p.grad is not None for p in pyramid.parameters()))

    def test_detector_registers_frozen_body_and_trainable_pyramid(self):
        cfg = self.config()
        model = coco.build_frozen_detector(cfg["backbone"], cfg["detector"], torch.device("cpu"), profile=PROFILE)
        self.assertTrue(hasattr(model.backbone, "body"))
        self.assertFalse(any(p.requires_grad for p in model.backbone.body.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.fpn.parameters()))
        model.train()
        self.assertFalse(model.backbone.body.training)
        self.assertEqual(model.roi_heads.box_roi_pool.featmap_names, ["0", "1", "2", "3"])
        self.assertEqual(model.rpn.anchor_generator.sizes, ((8,), (16,), (32,), (64,)))
        self.assertEqual(model.transform.size_divisible, 32)
        # The supported timm body retains the existing ImageNet normalization, once.
        image = torch.full((3, 64, 64), .5)
        images, _ = model.transform([image])
        expected = (image - torch.tensor(model.transform.image_mean)[:,None,None]) / torch.tensor(model.transform.image_std)[:,None,None]
        torch.testing.assert_close(images.tensors[0], expected)
        maps = model.backbone(images.tensors)
        self.assertEqual([tuple(m.shape[-2:]) for m in maps.values()], [(16,16),(8,8),(4,4),(2,2)])

    def test_invalid_profile_stride_provider_and_anchor_count_refused(self):
        for mod, cfg in ((ade20k, ade_config(Path("/missing"))), (coco, self.config())):
            cfg["profile"] = "capture_typo"
            with self.assertRaisesRegex(mod.ConfigError, "profile"):
                mod.validate_config(cfg)
        for changes in ({"patch_size": 14}, {"kind": "unknown_provider"}):
            cfg = self.config(); cfg["backbone"].update(changes)
            with self.assertRaisesRegex((ValueError, NotImplementedError), "stride|provider|kind"):
                coco.build_frozen_detector(cfg["backbone"], cfg["detector"], torch.device("cpu"), profile=PROFILE)
        cfg = self.config(); cfg["detector"]["anchor_sizes"] = [8, 16, 32]
        with self.assertRaisesRegex(ValueError, "four|4"):
            coco.build_frozen_detector(cfg["backbone"], cfg["detector"], torch.device("cpu"), profile=PROFILE)

    def test_real_coco_evaluator_six_metrics_and_empty_predictions(self):
        class Predictions(torch.nn.Module):
            def __init__(self, empty=False):
                super().__init__(); self.empty = empty
            def forward(self, images):
                return [{"boxes": torch.tensor([[8.,8.,32.,32.]])[:0 if self.empty else 1],
                         "labels": torch.tensor([1])[:0 if self.empty else 1],
                         "scores": torch.tensor([1.])[:0 if self.empty else 1]} for _ in images]
        with tempfile.TemporaryDirectory() as d:
            root = tiny_coco(Path(d), per=1)
            ds = coco.CocoDetectionForFRCNN(str(root/"images/val2017"), str(root/"annotations/instances_val2017.json"))
            loader = torch.utils.data.DataLoader(ds, collate_fn=coco.collate)
            metrics = coco.evaluate(Predictions(), loader, ds, "cpu", profile=PROFILE)
            for k in ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_small"):
                self.assertAlmostEqual(metrics[k], 1.)
            for k in ("bbox_mAP_medium", "bbox_mAP_large"):
                self.assertEqual(metrics[k], -1.)  # COCOeval's undefined-bin sentinel.
            empty = coco.evaluate(Predictions(True), loader, ds, "cpu", profile=PROFILE)
            self.assertEqual(set(empty), set(metrics))
            self.assertTrue(all(v == 0 for v in empty.values()))

    def test_real_detection_cli_records_incomplete_profile(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config(tiny_coco(Path(d)/"data", per=1))
            cfg["detector"].update(max_train_samples=0, max_val_samples=0)
            path, out = Path(d)/"cfg.json", Path(d)/"out"
            path.write_text(json.dumps(cfg))
            self.assertEqual(coco.main(["--config", str(path), "--out", str(out)]), 0)
            self.assertTrue(contract.verify(out, path, 0)[0])
            result = json.loads((out/"results.json").read_text())
            self.assertEqual(result["profile"], PROFILE)
            self.assertFalse(result["record_value"])
            self.assertFalse(result["canonical_eligible"])
            self.assertFalse(result["subset_or_smoke"])
            self.assertEqual(result["metric_units"], "ratio; -1 means undefined")
            metrics = json.loads((out/"metrics.json").read_text())["metrics"]
            for k in ("coco_map", "coco_map_50", "coco_map_75", "coco_map_small", "coco_map_medium", "coco_map_large"):
                self.assertIn(k, metrics)

    def test_cuda_update_changes_pyramid_not_encoder(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        cfg = self.config()
        model = coco.build_frozen_detector(cfg["backbone"], cfg["detector"], torch.device("cuda"), profile=PROFILE)
        model.train()
        before = {k:v.clone() for k,v in model.backbone.body.state_dict().items()}
        weights = model.backbone.fpn.lateral4.weight.detach().clone()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        losses = model([torch.rand(3,64,64,device="cuda")],
                       [{"boxes":torch.tensor([[8.,8.,32.,32.]],device="cuda"), "labels":torch.tensor([1],device="cuda")}])
        loss = sum(losses.values()); self.assertTrue(torch.isfinite(loss)); loss.backward(); opt.step()
        self.assertFalse(torch.equal(weights, model.backbone.fpn.lateral4.weight))
        for k,v in model.backbone.body.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
