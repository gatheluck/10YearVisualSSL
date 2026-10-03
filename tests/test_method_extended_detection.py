"""Behavioral contracts for Extended boxes, masks and benchmark identity."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AVAILABLE = all(
    importlib.util.find_spec(x) for x in ("torch", "torchvision", "pycocotools")
)


@unittest.skipUnless(AVAILABLE, "detection environment required")
class TestExtendedDetection(unittest.TestCase):
    def setUp(self):
        from downstream import native_detection

        self.api = native_detection

    def fixture(self, root, *, mask=True, federated=False):
        from PIL import Image

        Image.new("RGB", (40, 20), (255, 128, 0)).save(root / "one.png")
        images = [{"id": 1, "file_name": "one.png", "width": 40, "height": 20}]
        if federated:
            images[0].update(neg_category_ids=[9], not_exhaustive_category_ids=[])
        obj = {
            "images": images,
            "categories": [
                {"id": 3, "name": "a", "frequency": "f", "image_count": 1},
                {"id": 9, "name": "b", "frequency": "r", "image_count": 1},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 3,
                    "bbox": [4, 2, 16, 12],
                    "area": 192,
                    "iscrowd": 0,
                    "segmentation": [[4, 2, 20, 2, 20, 14, 4, 14]],
                }
            ],
        }
        path = root / "annotations.json"
        path.write_text(json.dumps(obj))
        return path, obj

    def test_sparse_ids_masks_and_joint_flip(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _obj = self.fixture(root)
            data = self.api.ExtendedDetectionData(root, path, train=True, mask=True)
            with mock.patch("random.random", return_value=0.0):
                image, target = data[0]
            self.assertEqual(data.category_to_label, {3: 1, 9: 2})
            self.assertEqual(data.num_classes, 3)
            torch.testing.assert_close(
                target["boxes"], torch.tensor([[20.0, 2.0, 36.0, 14.0]])
            )
            self.assertEqual(target["labels"].tolist(), [1])
            self.assertEqual(image.shape, (3, 20, 40))
            self.assertEqual(target["masks"].shape, (1, 20, 40))
            self.assertEqual(int(target["masks"].sum()), 192)
            self.assertGreater(int(target["masks"][0, :, 20:].sum()), 0)

    def test_input_resize_and_training_cap_do_not_change_evaluation_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            root = Path(tmp)
            path, obj = self.fixture(root)
            Image.new("RGB", (800, 400)).save(root / "one.png")
            obj["images"][0].update(width=800, height=400)
            obj["annotations"] = [
                dict(obj["annotations"][0], id=i + 1) for i in range(70)
            ]
            path.write_text(json.dumps(obj))
            data = self.api.ExtendedDetectionData(root, path, train=False, mask=True)
            image, target = data[0]
            self.assertEqual(tuple(image.shape), (3, 256, 512))
            self.assertEqual(len(target["boxes"]), 64)
            self.assertEqual(len(data.coco.getAnnIds(imgIds=1)), 70)

    def test_corrupt_masks_paths_duplicate_ids_and_geometry_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, base = self.fixture(root)
            for change in ("duplicate", "escape", "unknown", "geometry", "mask"):
                obj = copy.deepcopy(base)
                if change == "duplicate":
                    obj["images"].append(obj["images"][0])
                if change == "escape":
                    obj["images"][0]["file_name"] = "../one.png"
                if change == "unknown":
                    obj["annotations"][0]["category_id"] = 88
                if change == "geometry":
                    obj["images"][0]["width"] = 39
                if change == "mask":
                    obj["annotations"][0]["segmentation"] = {"bad": True}
                path.write_text(json.dumps(obj))
                with (
                    self.subTest(change=change),
                    self.assertRaises((ValueError, TypeError, KeyError)),
                ):
                    self.api.ExtendedDetectionData(root, path, train=False, mask=True)[
                        0
                    ]

    def prediction(self, label=1):
        import torch

        masks = torch.zeros(1, 1, 20, 40)
        masks[:, :, 2:14, 4:20] = 1
        return {
            "boxes": torch.tensor([[4.0, 2.0, 20.0, 14.0]]),
            "labels": torch.tensor([label]),
            "scores": torch.tensor([0.9]),
            "masks": masks,
        }

    def test_coco_label_inverse_and_livecell_category_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.fixture(root)
            data = self.api.ExtendedDetectionData(root, path, train=False, mask=True)
            for name in ("coco2017", "livecell"):
                score = self.api.evaluate_extended_detection(
                    data, [(1, (20, 40), self.prediction(2))], dataset=name
                )
                self.assertAlmostEqual(
                    score["bbox_ap"], 100.0 if name == "livecell" else 0.0, places=5
                )
                if name == "livecell":
                    self.assertAlmostEqual(score["mask_ap"], 100.0, places=5)
                    self.assertEqual(score["evaluation_max_detections"], 2000)
                self.assertFalse(score["record_value"])
            good = self.api.evaluate_extended_detection(
                data, [(1, (20, 40), self.prediction())], dataset="coco2017"
            )
            self.assertAlmostEqual(good["bbox_ap"], 100.0, places=5)

    def test_empty_predictions_are_zero_but_missing_duplicate_unknown_and_nonfinite_fail(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.fixture(root)
            data = self.api.ExtendedDetectionData(root, path, train=False, mask=False)
            empty = {k: v[:0] for k, v in self.prediction().items()}
            result = self.api.evaluate_extended_detection(
                data, [(1, (20, 40), empty)], dataset="coco2017"
            )
            self.assertEqual(result["bbox_ap"], 0.0)
            bad = self.prediction()
            bad["scores"][0] = float("nan")
            for outputs in (
                [],
                [(1, (20, 40), bad)],
                [(99, (20, 40), empty)],
                [(1, (20, 40), empty)] * 2,
            ):
                with self.assertRaises(ValueError):
                    self.api.evaluate_extended_detection(
                        data, outputs, dataset="coco2017"
                    )
            with self.assertRaises(ValueError):
                self.api.evaluate_extended_detection(
                    data, [(1, (20, 40), empty)], dataset="hico_det"
                )

    @unittest.skipUnless(
        importlib.util.find_spec("lvis"), "official LVIS evaluator required"
    )
    def test_lvis_federated_unknown_categories_and_original_resolution(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, obj = self.fixture(root, federated=True)
            data = self.api.ExtendedDetectionData(root, path, train=False, mask=True)
            pred = self.prediction()
            pred["boxes"] *= 0.5
            pred["masks"] = torch.nn.functional.interpolate(
                pred["masks"], size=(10, 20), mode="nearest"
            )
            result = self.api.evaluate_extended_detection(
                data, [(1, (10, 20), pred)], dataset="lvis_v1"
            )
            self.assertAlmostEqual(result["bbox_ap"], 100.0, places=5)
            self.assertGreater(result["mask_ap"], 90.0)
            self.assertEqual(result["evaluation_max_detections"], 300)
            del obj["images"][0]["neg_category_ids"]
            path.write_text(json.dumps(obj))
            invalid = self.api.ExtendedDetectionData(root, path, train=False, mask=True)
            with self.assertRaises(ValueError):
                self.api.evaluate_extended_detection(
                    invalid, [(1, (20, 40), self.prediction())], dataset="lvis_v1"
                )

    @unittest.skipUnless(
        importlib.util.find_spec("lvis"), "official LVIS evaluator required"
    )
    def test_lvis_federated_negative_and_nonexhaustive_labels_change_false_positive_accounting(
        self,
    ):
        import numpy as np
        import torch

        had_alias = hasattr(np, "float")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, obj = self.fixture(root, federated=True)
            obj["images"][0]["neg_category_ids"] = []
            obj["images"].append(dict(obj["images"][0], id=2))
            obj["annotations"].append(
                dict(obj["annotations"][0], id=2, image_id=2, category_id=9)
            )
            first = self.prediction()
            for key in ("boxes", "labels", "scores", "masks"):
                first[key] = torch.cat([first[key], first[key].clone()])
            first["labels"][1] = 2
            first["scores"][1] = 0.99
            values = []
            for negatives, incomplete in (([], []), ([9], []), ([9], [9])):
                obj["images"][0].update(
                    neg_category_ids=negatives, not_exhaustive_category_ids=incomplete
                )
                path.write_text(json.dumps(obj))
                data = self.api.ExtendedDetectionData(
                    root, path, train=False, mask=True
                )
                score = self.api.evaluate_extended_detection(
                    data,
                    [(1, (20, 40), first), (2, (20, 40), self.prediction(2))],
                    dataset="lvis_v1",
                )
                values.append(score["bbox_ap"])
            self.assertAlmostEqual(values[0], 100.0)
            self.assertLess(values[1], values[0])
            self.assertAlmostEqual(values[2], 100.0)
            self.assertEqual(hasattr(np, "float"), had_alias)
            empty = {key: value[:0] for key, value in first.items()}
            score = self.api.evaluate_extended_detection(
                data, [(1, (20, 40), empty), (2, (20, 40), empty)], dataset="lvis_v1"
            )
            self.assertEqual(score["mask_ap"], 0.0)

    def test_required_ci_runs_detection_and_installs_official_evaluator(self):
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed

        if not HAVE_YAML or not WORKFLOWS.is_dir():
            self.skipTest("checkout and YAML required")
        steps = parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
        command = next(
            s["run"]
            for s in steps
            if s.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        module = "tests.test_method_extended_detection"
        self.assertTrue(_runs_finetune_tests(command, module=module))
        self.assertFalse(_runs_finetune_tests("echo " + module, module=module))
        import shlex

        imported = [
            shlex.split(line)[-1] for line in command.splitlines() if " -c " in line
        ]
        self.assertTrue(
            any("lvis" in line.split("import ")[-1].split(", ") for line in imported)
        )

    def test_prediction_masks_are_probabilities_and_empty_ground_truth_is_not_a_score(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, obj = self.fixture(root)
            data = self.api.ExtendedDetectionData(root, path, train=False, mask=True)
            bad = self.prediction()
            bad["masks"][0, 0, 0, 0] = 2.0
            with self.assertRaises(ValueError):
                self.api.evaluate_extended_detection(
                    data, [(1, (20, 40), bad)], dataset="livecell"
                )
            obj["annotations"] = []
            path.write_text(json.dumps(obj))
            data = self.api.ExtendedDetectionData(root, path, train=False, mask=True)
            with self.assertRaises(ValueError):
                self.api.evaluate_extended_detection(
                    data, [(1, (20, 40), self.prediction())], dataset="livecell"
                )

    def test_livecell_tied_scores_use_category_order_before_detection_cap(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.fixture(root)
            data = self.api.ExtendedDetectionData(root, path, train=False, mask=True)
            pred = self.prediction()
            for key, value in list(pred.items()):
                pred[key] = value.repeat((2001,) + (1,) * (value.ndim - 1))
            pred["labels"][:2000] = 2
            pred["boxes"][:2000] = torch.tensor([30.0, 0.0, 38.0, 8.0])
            pred["masks"][:2000] = 0.0
            result = self.api.evaluate_extended_detection(
                data, [(1, (20, 40), pred)], dataset="livecell"
            )
            self.assertAlmostEqual(result["mask_ap"], 100.0)
            self.assertAlmostEqual(result["bbox_ap"], 100.0)

    def body(self):
        import torch

        class Body(torch.nn.Module):
            out_channels = 8

            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 16, 16)

            def forward_features(self, x):
                return self.conv(x)

        return Body()

    def test_detector_uses_transposed_pyramid_and_explicit_geometry_caps(self):
        from downstream.native_detection import TransposedFeaturePyramid

        for dataset, cap in [("coco2017", 100), ("livecell", 3000), ("lvis_v1", 300)]:
            model = self.api.build_extended_detector(
                self.body(),
                3,
                dataset=dataset,
                reader_profile=None,
                geometry_profile="captured_224_256",
            )
            self.assertIsInstance(model.backbone.fpn, TransposedFeaturePyramid)
            self.assertEqual(model.transform.min_size, (224,))
            self.assertEqual(model.transform.max_size, 256)
            self.assertEqual(model.roi_heads.detections_per_img, cap)
            if dataset == "livecell":
                self.assertEqual(model.rpn._post_nms_top_n["testing"], 3000)
            model.train()
            self.assertFalse(model.backbone.body.training)
        with self.assertRaises(ValueError):
            self.api.build_extended_detector(
                self.body(),
                3,
                dataset="hico_det",
                reader_profile=None,
                geometry_profile="captured_224_256",
            )
        with self.assertRaises(ValueError):
            self.api.build_extended_detector(
                self.body(),
                3,
                dataset="coco2017",
                reader_profile=None,
                geometry_profile="guess",
            )

    def test_box_and_mask_backward_freeze_encoder_but_update_heads_and_readers(self):
        import torch

        torch.set_num_threads(1)
        for name in ("coco2017", "livecell"):
            for profile in (None, "captured_single_block_v1", "captured_cross_self_v1"):
                body = self.body()
                model = self.api.build_extended_detector(
                    body,
                    3,
                    dataset=name,
                    reader_profile=profile,
                    geometry_profile="captured_224_256",
                )
                model.rpn._pre_nms_top_n.update(training=32)
                model.rpn._post_nms_top_n.update(training=16)
                model.train()
                images = [torch.rand(3, 32, 32)]
                target = {
                    "boxes": torch.tensor([[4.0, 4.0, 24.0, 24.0]]),
                    "labels": torch.tensor([1]),
                    "masks": torch.ones(1, 32, 32, dtype=torch.uint8),
                }
                losses = model(images, [target])
                self.assertTrue(all(torch.isfinite(x) for x in losses.values()))
                sum(losses.values()).backward()
                self.assertTrue(
                    all(
                        not p.requires_grad and p.grad is None
                        for p in body.parameters()
                    )
                )
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.roi_heads.parameters()
                    )
                )
                if profile:
                    self.assertTrue(
                        any(
                            p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.backbone.adapter.parameters()
                        )
                    )
                if name == "livecell":
                    self.assertIn("loss_mask", losses)

    def test_cli_artifacts_and_existing_output_protection(self):
        import subprocess
        import sys

        import torch

        from downstream import contract

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation, _ = self.fixture(root)
            predictions = root / "predictions.pt"
            torch.save([(1, (20, 40), self.prediction())], predictions)
            config = root / "config.json"
            example = (
                Path(__file__).resolve().parents[1]
                / "docs/examples/extended_detection_evaluation.json"
            )
            cfg = json.loads(example.read_text())
            cfg.update(
                data_root=str(root),
                annotation=str(annotation),
                predictions=str(predictions),
            )
            config.write_text(json.dumps(cfg))
            out = root / "out"
            args = [
                sys.executable,
                "-m",
                "downstream.extended_detection",
                "--config",
                str(config),
                "--out",
                str(out),
            ]
            result = subprocess.run(
                args, check=False, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(
                result.returncode,
                0,
                (out / "run_manifest.json").read_text()
                if (out / "run_manifest.json").exists()
                else result.stderr,
            )
            self.assertTrue(
                (out / "results.json").exists(),
                "evaluation must deliver results, not merely import",
            )
            self.assertEqual(contract.verify(out, config, 0), (True, []))
            report = json.loads((out / "results.json").read_text())
            self.assertAlmostEqual(report["metrics"]["mask_ap"], 100.0)
            self.assertEqual(len(report["annotation_sha256"]), 64)
            self.assertFalse(report["record_value"])
            before = (out / "results.json").read_bytes()
            self.assertNotEqual(
                subprocess.run(
                    args, check=False, capture_output=True, timeout=60
                ).returncode,
                0,
            )
            self.assertEqual((out / "results.json").read_bytes(), before)

    def test_twenty_actual_provider_loss_routes(self):
        try:
            import importlib

            importlib.import_module("einops")
            importlib.import_module("timm")
            transformers = importlib.import_module("transformers")
            _ = transformers.DINOv3ViTModel, transformers.SiglipVisionModel
        except ImportError:
            self.skipTest("five-family integration dependencies required")
        import torch

        from downstream import spatial_backbones as sb
        from tests.test_method_hf_basic5 import TestVisionFamilies
        from tests.test_method_raev2 import TestK7Backbone
        from tests.test_method_vggt_omega import TestOmega

        torch.set_num_threads(1)
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
        for spec in specs:
            for dataset in ("coco2017", "livecell"):
                for ap in (False, True):
                    with self.subTest(kind=spec["kind"], dataset=dataset, attentive=ap):
                        body = sb.build_frozen_backbone(spec, torch.device("cpu"))
                        profile = (
                            sb._load_provider(
                                sb.discover_providers()[spec["kind"]]
                            ).EXTENDED_DENSE_READER
                            if ap
                            else None
                        )
                        model = self.api.build_extended_detector(
                            body,
                            3,
                            dataset=dataset,
                            reader_profile=profile,
                            geometry_profile="captured_224_256",
                        )
                        model.rpn._pre_nms_top_n["training"] = 32
                        model.rpn._post_nms_top_n["training"] = 16
                        model.train()
                        before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
                        opt = torch.optim.SGD(
                            [p for p in model.parameters() if p.requires_grad], lr=0.01
                        )
                        loss = sum(
                            model(
                                [torch.rand(3, 32, 32)],
                                [
                                    {
                                        "boxes": torch.tensor([[4.0, 4.0, 24.0, 24.0]]),
                                        "labels": torch.tensor([1]),
                                        "masks": torch.ones(
                                            1, 32, 32, dtype=torch.uint8
                                        ),
                                    }
                                ],
                            ).values()
                        )
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        opt.step()
                        self.assertFalse(
                            torch.equal(
                                before, model.roi_heads.box_predictor.cls_score.weight
                            )
                        )
                        self.assertTrue(
                            all(
                                p.grad is None and not p.requires_grad
                                for p in body.parameters()
                            )
                        )


if __name__ == "__main__":
    unittest.main()
