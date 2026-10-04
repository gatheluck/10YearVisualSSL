"""Behavioral contracts for pose, absolute localization and spatial reasoning."""

import importlib.util
import unittest

HAVE = all(importlib.util.find_spec(x) for x in ("torch", "torchvision", "numpy"))


@unittest.skipUnless(HAVE, "downstream dependencies required")
class TestHeads(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.structured_heads"),
            "Structured task components are missing",
        )
        import torch

        from downstream import structured_heads

        torch.set_num_threads(1)
        self.api = structured_heads

    def test_pose_loss_masks_whole_joints_and_preserves_zero_gradient(self):
        import torch

        p = torch.ones(2, 3, 4, 4, requires_grad=True)
        y = torch.zeros_like(p)
        y[:, 1] = float("nan")
        loss = self.api.heatmap_loss(p, y)
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        self.assertEqual(p.grad[:, 1].abs().sum().item(), 0.0)
        self.assertAlmostEqual(p.grad[:, 0].sum().item(), 1.0)
        self.assertEqual(
            self.api.heatmap_loss(p, torch.full_like(p, float("nan"))).item(), 0.0
        )
        for bad in (torch.full_like(p, float("inf")), y.clone()):
            if not bad.isinf().any():
                bad[0, 0, 0, 0] = float("nan")
            with self.assertRaises(ValueError):
                self.api.heatmap_loss(p, bad)

    def test_camera_head_produces_rotations_and_scene_selected_l1(self):
        import torch

        head = self.api.CameraHead(4, 2)
        x = torch.zeros(3, 4)
        p = head(x)
        self.assertEqual(tuple(p.shape), (3, 2, 12))
        torch.testing.assert_close(
            p[..., 3:].reshape(3, 2, 3, 3), torch.eye(3).expand(3, 2, 3, 3)
        )
        with torch.no_grad():
            head.proj.bias[9:12] = 10.0
        p = head(x)
        scenes = torch.tensor([0, 1, 0])
        target = torch.cat(
            [scenes[:, None].float(), p[torch.arange(3), scenes].detach()], 1
        )
        target[:, 1:4] += 2
        loss = self.api.camera_loss(p, target)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        self.assertGreater(head.proj.bias.grad.abs().sum().item(), 0)
        for scene in (-1.0, 2.0, 0.5, float("nan")):
            bad = target.clone()
            bad[0, 0] = scene
            with self.assertRaises(ValueError):
                self.api.camera_loss(p, bad)

    def test_question_order_padding_options_and_visual_gradient(self):
        import torch

        torch.manual_seed(8)
        head = self.api.QuestionHead(4)
        visual = torch.randn(2, 4, requires_grad=True)
        q = torch.tensor(
            [
                [[2, 3, 0], [4, 0, 0], [5, 6, 0], [12, 0, 0]],
                [[7, 8, 9], [10, 0, 0], [11, 0, 0], [0, 0, 0]],
            ]
        )
        logits = head(visual, q)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(logits[1, 2].item(), -1e4)
        padded = torch.nn.functional.pad(q, (0, 9))
        torch.testing.assert_close(head(visual, padded), logits)
        permuted = q[:, [0, 2, 1, 3]]
        torch.testing.assert_close(head(visual, permuted), logits[:, [1, 0, 2]])
        reversed_q = q.clone()
        reversed_q[:, 0] = q[:, 0].flip(1)
        # Interior/leading padding is invalid, not silently consumed as text.
        with self.assertRaises(ValueError):
            head(visual, reversed_q)
        torch.nn.functional.cross_entropy(
            logits, torch.zeros(2, dtype=torch.long)
        ).backward()
        self.assertGreater(visual.grad.abs().sum().item(), 0.0)
        bad = q.clone()
        bad[0, 0] = 0
        with self.assertRaises(ValueError):
            head(visual, bad)

    def test_frozen_global_readout_preserves_normalized_patch_mean_fallback(self):
        import torch

        from downstream import extended_classification as images

        self.assertTrue(callable(getattr(images, "frozen_global_features", None)))

        class PatchOnly:
            def forward_features(self, x):
                return x

        x = torch.tensor([[[[3.0, 3.0]], [[4.0, 4.0]]]])
        torch.testing.assert_close(
            images.frozen_global_features(PatchOnly(), x), torch.tensor([[0.6, 0.8]])
        )

    def test_probe_normalization_freezing_and_readouts(self):
        import torch
        from torch import nn
        from torchvision.transforms.functional import normalize

        from downstream.captured_readers import SINGLE_BLOCK

        class Backbone(nn.Module):
            out_channels = global_channels = 4

            def __init__(self):
                super().__init__()
                self.proj = nn.Conv2d(3, 4, 1)
                self.seen = None

            def forward_features(self, x):
                self.seen = x.detach().clone()
                return self.proj(x)[:, :, ::32, ::32]

            def classification_features(self, x, *, adaptation):
                return self.forward_features(x).mean((2, 3))

        for task, count in [("pose", 14), ("localization", 2), ("reasoning", 1)]:
            for reader in (None, SINGLE_BLOCK):
                torch.manual_seed(5)
                backbone = Backbone()
                model = self.api.Probe(backbone, task, count, reader).train()
                images = torch.rand(2, 3, 64, 64)
                q = torch.tensor([[[1, 2], [3, 4], [5, 6]]] * 2)
                output = model(images, q) if task == "reasoning" else model(images)
                self.assertEqual(
                    tuple(output.shape),
                    {
                        "pose": (2, 14, 56, 56),
                        "localization": (2, 2, 12),
                        "reasoning": (2, 2),
                    }[task],
                )
                torch.testing.assert_close(
                    backbone.seen,
                    normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                )
                self.assertFalse(backbone.training)
                output.square().mean().backward()
                self.assertTrue(
                    all(
                        p.grad is None and not p.requires_grad
                        for p in backbone.parameters()
                    )
                )
                self.assertTrue(
                    any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.head.parameters()
                    )
                )
        with self.assertRaises(ValueError):
            self.api.Probe(Backbone(), "unknown", 1, None)


@unittest.skipUnless(HAVE, "downstream dependencies required")
class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.structured_metrics"),
            "Structured task metrics are missing",
        )
        from downstream import structured_metrics

        self.api = structured_metrics

    def test_decode_quarter_pixel_and_crop_inverse(self):
        import numpy as np

        heat = np.zeros((1, 6, 8))
        heat[0, 3, 4] = 3
        heat[0, 3, 5] = 2
        heat[0, 4, 4] = 1
        xy, peak = self.api.decode_heatmaps(heat, np.array([10, 20, 80, 60]))
        np.testing.assert_allclose(xy, [[52.5, 52.5]])
        np.testing.assert_equal(peak, [3])
        with self.assertRaises(ValueError):
            self.api.decode_heatmaps(heat, [0, 0, -1, 2])

    def test_mpii_one_based_sidecar_and_ignored_joints(self):
        import numpy as np

        pred = np.zeros((1, 16, 2))
        pos = np.ones_like(pred)
        heads = np.array([[[0.0, 0.0], [10.0, 0.0]]])
        missing = np.zeros((1, 16))
        self.assertEqual(self.api.pckh_counts(pred, pos, missing, heads), (14, 14))
        pred[:, 0, 0] = 3.01
        self.assertEqual(self.api.pckh_counts(pred, pos, missing, heads), (13, 14))
        missing[:, 0] = 1
        self.assertEqual(self.api.pckh_counts(pred, pos, missing, heads), (13, 13))
        with self.assertRaises(ValueError):
            self.api.pckh_counts(pred, pos, missing, heads * 0)

    def test_camera_metrics_macro_scene_medians_and_rotation_units(self):
        import numpy as np

        score = self.api.CameraScore("cambridge_landmarks", ["a", "b"])
        for scene, shift in [("a", 1), ("a", 3), ("a", 8), ("b", 10)]:
            pred = np.eye(4)
            pred[0, 3] = shift
            score.update(scene, pred, np.eye(4))
        result = score.result()
        self.assertEqual(result["median_translation_m"], 6.5)
        self.assertEqual(result["median_rotation_deg"], 0.0)
        score = self.api.CameraScore("seven_scenes", ["a"])
        gt = np.eye(4)
        gt[0, 0] += 0.0003
        score.update("a", np.eye(4), gt)
        self.assertAlmostEqual(score.result()["median_rotation_deg"], 0.0)
        with self.assertRaises(ValueError):
            self.api.CameraScore("cambridge_landmarks", ["a"]).update(
                "a", np.eye(4), gt
            )
        with self.assertRaises(ValueError):
            self.api.CameraScore("seven_scenes", ["a", "b"]).result()

    def test_crowdpose_ranking_ignored_people_and_duplicate_images(self):
        import numpy as np

        kp = np.array([[10.0, 10.0, 2.0]] * 14).ravel().tolist()
        gt = {
            "keypoints": kp,
            "bbox": [0, 0, 20, 20],
            "num_keypoints": 14,
            "iscrowd": 0,
        }
        metric = self.api.KeypointAP()
        metric.add_image("a", [gt], [{"keypoints": kp, "score": 0.5}], 0.1)
        self.assertEqual(metric.result()["keypoint_AP"], 1.0)
        with self.assertRaises(ValueError):
            metric.add_image("a", [gt], [])
        miss = np.array(kp).reshape(14, 3)
        miss[:, :2] += 100
        metric = self.api.KeypointAP()
        metric.add_image(
            "a",
            [gt],
            [
                {"keypoints": miss.ravel().tolist(), "score": 0.9},
                {"keypoints": kp, "score": 0.5},
            ],
        )
        self.assertEqual(metric.result()["keypoint_AP"], 0.5)
        capped = self.api.KeypointAP()
        capped.add_image(
            "capped",
            [gt],
            [{"keypoints": miss.ravel().tolist(), "score": 0.1} for _ in range(20)]
            + [{"keypoints": kp, "score": 0.9}],
        )
        self.assertEqual(capped.result()["keypoint_AP"], 1.0)

        self.assertEqual(
            self.api.oks(kp, kp, 1, [0, 0, 20, 20]),
            self.api.oks(kp, kp, 999, [0, 0, 20, 20]),
        )


@unittest.skipUnless(HAVE, "downstream dependencies required")
class TestData(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.structured_data"),
            "Structured sample and split handling is missing",
        )
        import tempfile
        from pathlib import Path

        from downstream import structured_data

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.api = structured_data

    def image(self, name, value=30):
        import numpy as np
        from PIL import Image

        Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8)).save(
            self.root / name
        )
        return {"path": name}

    def manifest(self, dataset):
        import numpy as np

        rows = []
        for i in range(2):
            row = {"id": str(i), "image": self.image(f"{i}.png", 30 + i)}
            if dataset in ("seven_scenes", "cambridge_landmarks"):
                row.update(scene="room", pose=np.eye(4).tolist())
            elif dataset == "three_d_srbench":
                row.update(question="Which café?", options=["left", "right"], target=1)
            else:
                n = 16 if dataset == "mpii_pose" else 14
                person = {
                    "id": f"p{i}",
                    "joints": [[6.0, 6.0]] * n,
                    "labeled": [True] * n,
                    "crop": [0.0, 0.0, 16.0, 16.0],
                }
                if dataset == "mpii_pose":
                    person.update(
                        pos_gt_src=[[7.0, 7.0]] * 16,
                        jnt_missing=[0] * 16,
                        headboxes_src=[[0.0, 0.0], [10.0, 0.0]],
                    )
                else:
                    person.update(
                        keypoints=[[6.0, 6.0, 2.0]] * 14,
                        bbox=[0.0, 0.0, 16.0, 16.0],
                        num_keypoints=14,
                        iscrowd=0,
                    )
                    row["crowd_index"] = 0.1
                row["people"] = [person]
            rows.append(row)
        out = {
            "schema_version": 1,
            "dataset": dataset,
            "split_evidence": "explicit fixture split",
            "train": [rows[0]],
            "validation": [rows[1]],
        }
        if dataset in ("seven_scenes", "cambridge_landmarks"):
            out["scenes"] = ["room"]
        return out

    def load(self, obj):
        import json

        p = self.root / "samples.json"
        p.write_text(json.dumps(obj))
        return self.api.load_data(p, self.root, obj["dataset"])

    def test_five_datasets_and_exact_heatmap_geometry(self):
        import numpy as np
        import torch

        for name in (
            "mpii_pose",
            "crowdpose",
            "seven_scenes",
            "cambridge_landmarks",
            "three_d_srbench",
        ):
            tr, va, m = self.load(self.manifest(name))
            self.assertEqual((len(tr), len(va)), (1, 1))
            self.assertEqual(tuple(tr[0][0].shape), (3, 224, 224))
            self.assertFalse(m["official_membership_verified"])
            self.assertEqual(va[0][-1], 0)
            if name in ("mpii_pose", "crowdpose"):
                target = tr[0][1]
                self.assertEqual(target[0, 21, 21].item(), 1.0)
                self.assertAlmostEqual(target[0, 21, 23].item(), np.exp(-0.5), places=6)
            elif name in ("seven_scenes", "cambridge_landmarks"):
                torch.testing.assert_close(tr[0][1][4:], torch.eye(3).flatten())
            else:
                self.assertEqual(
                    tr[0][1][0].tolist(), [x + 1 for x in "Which café?".encode()]
                )

    def test_leakage_ids_paths_and_data_change_identity(self):
        import copy

        obj = self.manifest("three_d_srbench")
        _, _, before = self.load(obj)
        self.image("1.png", 90)
        _, _, after = self.load(obj)
        self.assertNotEqual(before["assets_sha256"], after["assets_sha256"])
        for mutate in ("overlap", "id", "escape", "answer", "empty"):
            bad = copy.deepcopy(obj)
            if mutate == "overlap":
                bad["validation"][0]["image"] = bad["train"][0]["image"]
            if mutate == "id":
                bad["validation"][0]["id"] = bad["train"][0]["id"]
            if mutate == "escape":
                bad["train"][0]["image"] = {"path": "../outside.png"}
            if mutate == "answer":
                bad["train"][0]["target"] = 2
            if mutate == "empty":
                bad["train"][0]["question"] = " "
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.load(bad)

    def test_pose_label_guard_and_missing_mpii_sidecar(self):
        import copy

        for name in ("mpii_pose", "crowdpose"):
            obj = self.manifest(name)
            bad = copy.deepcopy(obj)
            bad["train"][0]["people"][0]["labeled"][0] = "yes"
            with self.assertRaises(ValueError):
                self.load(bad)
            if name == "mpii_pose":
                del obj["validation"][0]["people"][0]["pos_gt_src"]
                with self.assertRaises(ValueError):
                    self.load(obj)

    def test_manifest_schema_scenes_and_archive_guards(self):
        import copy
        import zipfile

        obj = self.manifest("seven_scenes")
        for change in (
            {"schema_version": True},
            {"dataset": "unknown"},
            {"split_evidence": " "},
            {"train": []},
            {"scenes": ["other"]},
            {"scenes": ["room", "missing"]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(dict(obj, **change))
        with zipfile.ZipFile(self.root / "images.zip", "w") as z:
            z.writestr("0.png", (self.root / "0.png").read_bytes())
        good = copy.deepcopy(obj)
        good["train"][0]["image"] = {"path": "images.zip", "member": "0.png"}
        self.load(good)
        for asset in (
            {"path": "images.zip", "member": "missing.png"},
            {"path": "images.zip", "member": "../0.png"},
            {"path": "0.png", "extra": True},
            {"path": str(self.root / "0.png")},
        ):
            bad = copy.deepcopy(obj)
            bad["train"][0]["image"] = asset
            with self.assertRaises(ValueError):
                self.load(bad)
        broken = self.root / "broken.png"
        broken.write_bytes(b"not an image")
        bad = copy.deepcopy(obj)
        bad["train"][0]["image"] = {"path": "broken.png"}
        tr, _, _ = self.load(bad)
        with self.assertRaises(OSError):
            tr[0]

    def test_qa_collation_preserves_all_bytes_and_variable_choices(self):
        import torch

        obj = self.manifest("three_d_srbench")
        obj["train"][0]["question"] = "x" * 600
        tr, va, _ = self.load(obj)
        a = tr[0]
        b = (
            a[0],
            [self.api.byte_tokens("q")]
            + [self.api.byte_tokens(x) for x in ["a", "b", "c"]],
            2,
        )
        _x, q, y = self.api.collate([a, b])
        self.assertEqual(tuple(q.shape), (2, 4, 600))
        self.assertEqual(q[0, 0].count_nonzero().item(), 600)
        self.assertTrue(torch.equal(y, torch.tensor([1, 2])))
        self.assertEqual(len(va[0][1]), 3)


@unittest.skipUnless(
    HAVE and importlib.util.find_spec("transformers"),
    "downstream encoder dependencies required",
)
class TestExecution(unittest.TestCase):
    image = TestData.image
    manifest = TestData.manifest
    load = TestData.load

    def setUp(self):
        TestData.setUp(self)
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.extended_structured"),
            "Structured task execution is missing",
        )
        from downstream import extended_structured

        self.run_api = extended_structured

    def config(self, dataset, adaptation="frozen"):
        return {
            "task": "extended_structured",
            "profile": "capture_extended_components",
            "dataset": dataset,
            "seed": 0,
            "device": "cpu",
            "data_root": str(self.root),
            "samples": str(self.root / "samples.json"),
            "transform_profile": "captured_structured_v1",
            "adaptation": adaptation,
            "reader_profile": "captured_single_block_v1"
            if adaptation == "attentive"
            else None,
            "backbone": {
                "kind": "siglip2_g",
                "arch": "fixture",
                "encoder": str(self.root / "encoder"),
                "img_size": 32,
                "patch_size": 16,
            },
            "probe": {"epochs": 1, "batch_size": 1, "num_workers": 0},
        }

    def encoder(self):
        import torch
        from transformers import SiglipVisionConfig, SiglipVisionModel

        torch.manual_seed(99)
        torch.set_num_threads(1)
        SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=32,
                patch_size=16,
            )
        ).save_pretrained(self.root / "encoder")

    def test_five_dataset_lp_ap_execution_metrics_and_resume(self):
        import json

        import torch

        from tests.test_method_extended_detection_training import TestDetectionTraining

        self.encoder()
        for dataset in (
            "mpii_pose",
            "crowdpose",
            "seven_scenes",
            "cambridge_landmarks",
            "three_d_srbench",
        ):
            obj = self.manifest(dataset)
            self.load(obj)
            for adaptation in ("frozen", "attentive"):
                cfg = self.config(dataset, adaptation)
                first = self.root / (dataset + adaptation + "1")
                first.mkdir()
                self.run_api.run(cfg, first)
                metrics = json.loads((first / "metrics.json").read_text())
                self.assertTrue(metrics)
                report = json.loads((first / "results.json").read_text())
                self.assertEqual(report["execution"]["updates"], 1)
                self.assertFalse(report["canonical_eligible"])
                self.assertEqual(report["final"]["evaluated_samples"], 1)
                self.assertFalse(
                    any(
                        k.startswith("backbone.")
                        for k in torch.load(first / "probe.pt", weights_only=True)
                    )
                )
                cfg["probe"]["epochs"] = 2
                full = self.root / (dataset + adaptation + "full")
                full.mkdir()
                self.run_api.run(cfg, full)
                cfg["resume"] = str(first / "resume.pt")
                resumed = self.root / (dataset + adaptation + "resumed")
                resumed.mkdir()
                self.run_api.run(cfg, resumed)
                TestDetectionTraining().equal_tree(
                    torch.load(full / "resume.pt", weights_only=True),
                    torch.load(resumed / "resume.pt", weights_only=True),
                )

    def test_cli_contract_no_overwrite_and_invalid_config(self):
        import json
        import subprocess
        import sys

        from downstream import contract

        self.encoder()
        self.load(self.manifest("three_d_srbench"))
        cfg = self.config("three_d_srbench")
        path = self.root / "config.json"
        path.write_text(json.dumps(cfg))
        out = self.root / "out"
        cmd = [
            sys.executable,
            "-m",
            "downstream.extended_structured",
            "--config",
            str(path),
            "--out",
            str(out),
        ]
        p = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=90)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(contract.verify(out, path, 0), (True, []))
        before = (out / "results.json").read_bytes()
        self.assertNotEqual(
            subprocess.run(
                cmd, check=False, capture_output=True, timeout=90
            ).returncode,
            0,
        )
        self.assertEqual((out / "results.json").read_bytes(), before)
        for change in (
            {"dataset": "sun_rgbd"},
            {"adaptation": "finetune"},
            {"seed": 1},
            {"transform_profile": "guess"},
        ):
            with self.assertRaises(ValueError):
                self.run_api.validate_config(dict(cfg, **change))

    def test_recipe_expands_observed_schedule_and_preserves_weight_decay(self):
        for dataset in ("mpii_pose", "seven_scenes", "three_d_srbench"):
            for adaptation, wd in [("frozen", 0.0001), ("attentive", 0.05)]:
                spec = self.run_api.recipe(dataset, adaptation)
                self.assertEqual(
                    spec["warmup"],
                    "5 epochs from 1e-6"
                    if dataset == "three_d_srbench"
                    else "1 epoch from 1e-6",
                )
                self.assertEqual(spec["schedule"], "cosine to 0")
                self.assertEqual(spec["weight_decay"], wd)

    def test_evaluation_requires_exact_population(self):
        import torch

        self.encoder()
        _, val, _ = self.load(self.manifest("three_d_srbench"))
        rows = [self.api.collate([val[0]]), self.api.collate([val[0]])]
        with self.assertRaises(ValueError):
            self.run_api.evaluate(val, rows, lambda x, q: torch.zeros(len(x), 2))
        with self.assertRaises(ValueError):
            self.run_api.evaluate(val, [], lambda x, q: torch.zeros(len(x), 2))

    def test_localization_evaluation_preserves_source_float32_targets(self):
        import numpy as np
        import torch

        obj = self.manifest("cambridge_landmarks")
        obj["validation"][0]["pose"][0][3] = 1.00000006
        _, val, _ = self.load(obj)
        prediction = torch.cat([torch.zeros(3), torch.eye(3).flatten()]).reshape(
            1, 1, 12
        )
        result = self.run_api.evaluate(
            val, [self.api.collate([val[0]])], lambda x, q: prediction
        )
        self.assertEqual(result["median_translation_m"], float(np.float32(1.00000006)))

    def test_nonempty_incomplete_evaluation_is_rejected(self):
        import copy

        import torch

        obj = self.manifest("three_d_srbench")
        row = copy.deepcopy(obj["validation"][0])
        row["id"] = "extra"
        row["image"] = self.image("extra.png", 80)
        obj["validation"].append(row)
        _, val, _ = self.load(obj)
        with self.assertRaises(ValueError):
            self.run_api.evaluate(
                val, [self.api.collate([val[0]])], lambda x, q: torch.zeros(len(x), 2)
            )

    def test_map_pool_magnitude_is_preserved_for_structured_lp(self):
        import torch
        from torchvision.transforms.functional import normalize

        from downstream.spatial_backbones import build_frozen_backbone
        from downstream.structured_heads import Probe

        self.encoder()
        cfg = self.config("seven_scenes")
        backbone = build_frozen_backbone(cfg["backbone"], torch.device("cpu"))
        model = Probe(backbone, "localization", 1, None)
        images = torch.rand(1, 3, 224, 224)
        normalized = normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        expected = backbone.classification_features(normalized, adaptation="finetune")
        observed = []
        handle = model.head.register_forward_pre_hook(
            lambda _m, args: observed.append(args[0].detach())
        )
        try:
            model(images)
        finally:
            handle.remove()
        self.assertGreater(abs(expected.norm().item() - 1), 0.1)
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)

    def test_omega_pool_magnitude_is_preserved_for_structured_lp(self):
        import torch
        from torchvision.transforms.functional import normalize

        from downstream.structured_heads import Probe
        from tests.test_method_vggt_omega import TestOmega

        owner = TestOmega()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        spec, _ = owner.fixture()
        backbone = owner.provider.build(spec)
        model = Probe(backbone, "localization", 1, None)
        images = torch.rand(1, 3, 224, 224)
        normalized = normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        expected = backbone.classification_features(normalized, adaptation="finetune")
        observed = []
        handle = model.head.register_forward_pre_hook(
            lambda _m, args: observed.append(args[0].detach())
        )
        try:
            model(images)
        finally:
            handle.remove()
        self.assertGreater(abs(expected.norm().item() - 1), 0.1)
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)

    def test_all_five_provider_three_family_lp_ap_routes(self):
        import json

        try:
            import einops
            import timm
            from transformers import DINOv3ViTModel, SiglipVisionModel
        except ImportError:
            self.skipTest("five-provider fixtures require full downstream dependencies")
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
        for dataset in ("mpii_pose", "seven_scenes", "three_d_srbench"):
            self.load(self.manifest(dataset))
            for spec in specs:
                for adaptation in ("frozen", "attentive"):
                    with self.subTest(
                        dataset=dataset, kind=spec["kind"], adaptation=adaptation
                    ):
                        cfg = self.config(dataset, adaptation)
                        cfg["backbone"] = spec
                        provider = sb._load_provider(
                            sb.discover_providers()[spec["kind"]]
                        )
                        attr = (
                            "EXTENDED_IMAGE_READER"
                            if dataset == "three_d_srbench"
                            else "EXTENDED_DENSE_READER"
                        )
                        cfg["reader_profile"] = (
                            getattr(provider, attr)
                            if adaptation == "attentive"
                            else None
                        )
                        out = self.root / (dataset + spec["kind"] + adaptation)
                        out.mkdir()
                        self.run_api.run(cfg, out)
                        report = json.loads((out / "results.json").read_text())
                        self.assertEqual(report["execution"]["updates"], 1)
                        self.assertEqual(report["final"]["evaluated_samples"], 1)

    def test_two_rank_question_worker_resume(self):
        import copy
        import json
        import os
        import socket
        import subprocess
        import sys

        import torch

        from tests.test_method_extended_detection_training import TestDetectionTraining

        self.encoder()
        obj = self.manifest("three_d_srbench")
        for split in ("train", "validation"):
            row = copy.deepcopy(obj[split][0])
            row["id"] += "extra"
            row["image"] = self.image(
                row["id"] + ".png", 75 if split == "train" else 85
            )
            row["options"] += ["above"]
            obj[split].append(row)
        self.load(obj)
        cfg = self.config("three_d_srbench", "attentive")
        cfg["probe"]["num_workers"] = 1
        for name, epochs, resume in [
            ("one", 1, None),
            ("full", 2, None),
            ("resumed", 2, str(self.root / "one/resume.pt")),
        ]:
            cfg["probe"]["epochs"] = epochs
            if resume:
                cfg["resume"] = resume
            path = self.root / "config.json"
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
                    "downstream.extended_structured",
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
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        a = torch.load(self.root / "full/resume.pt", weights_only=True)
        b = torch.load(self.root / "resumed/resume.pt", weights_only=True)
        TestDetectionTraining().equal_tree(a, b)
        self.assertEqual(len(b["rng"]), 2)
        report = json.loads((self.root / "resumed/results.json").read_text())
        self.assertEqual(report["final"]["evaluated_samples"], 2)


class TestDelivery(unittest.TestCase):
    def test_guide_examples_and_required_ci_cover_both_suites(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        self.assertTrue(
            (root / "docs/EXTENDED_STRUCTURED.md").is_file(), "structured guide missing"
        )
        for name, dataset in [
            ("pose", "mpii_pose"),
            ("localization", "seven_scenes"),
            ("reasoning", "three_d_srbench"),
        ]:
            cfg = json.loads((root / f"docs/examples/extended_{name}.json").read_text())
            self.assertEqual(cfg["dataset"], dataset)
            if HAVE:
                from downstream.extended_structured import validate_config

                validate_config(cfg)
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if HAVE_YAML:
            for module in (
                "tests.test_method_extended_structured",
                "tests.test_structured_membership",
            ):
                self.assertTrue(
                    any(
                        _runs_finetune_tests(step.get("run", ""), module=module)
                        for workflow in parsed().values()
                        for step in workflow.get("jobs", {})
                        .get("downstream", {})
                        .get("steps", [])
                    ),
                    "structured suite missing from required downstream CI",
                )
