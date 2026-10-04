"""Extended detection training, schedule, membership and continuation contracts."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

AVAILABLE = all(
    importlib.util.find_spec(x) for x in ("torch", "torchvision", "pycocotools", "lvis")
)


@unittest.skipUnless(AVAILABLE, "downstream detection environment required")
class TestDetectionTraining(unittest.TestCase):
    def setUp(self):
        from downstream import extended_detection as api
        from downstream import (
            native_detection,  # Initialize reexports before fixture patches.
        )

        self.assertIs(
            native_detection.build_extended_detector, api.build_extended_detector
        )

        self.api = api
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def equal_tree(self, left, right):
        import torch

        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.equal_tree(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_tree(a, b)
        else:
            self.assertEqual(left, right)

    def config(self, dataset="livecell", adaptation="frozen"):
        return {
            "task": "extended_detection_training",
            "profile": "capture_extended_components",
            "dataset": dataset,
            "seed": 0,
            "device": "cpu",
            "data_root": str(self.root),
            "train_annotations": str(self.root / "train.json"),
            "validation_annotations": str(self.root / "validation.json"),
            "adaptation": adaptation,
            "reader_profile": "captured_single_block_v1"
            if adaptation == "attentive"
            else None,
            "geometry_profile": "captured_224_256",
            "backbone": {"kind": "siglip2_g"},
            "probe": {"epochs": 1, "batch_size": 1, "num_workers": 0},
        }

    def data(self, dataset="livecell", count=3):
        from PIL import Image

        classes = {"coco2017": 80, "livecell": 8, "lvis_v1": 1203}[dataset]
        cats = [
            {"id": i * 2 + 1, "name": f"class{i}", "frequency": "f", "image_count": 1}
            for i in range(classes)
        ]
        for split, n in (("train", count), ("validation", 1)):
            images = []
            annotations = []
            for i in range(n):
                iid = i + 1 + (100 if split == "validation" else 0)
                name = f"{split}-{i}.png"
                Image.new("RGB", (32 + i, 24), (80, i * 40, 100)).save(self.root / name)
                images.append(
                    {
                        "id": iid,
                        "file_name": name,
                        "width": 32 + i,
                        "height": 24,
                        "neg_category_ids": [],
                        "not_exhaustive_category_ids": [],
                    }
                )
                annotations.append(
                    {
                        "id": iid,
                        "image_id": iid,
                        "category_id": 1,
                        "bbox": [2, 2, 12, 12],
                        "area": 144,
                        "iscrowd": 0,
                        "segmentation": [[2, 2, 14, 2, 14, 14, 2, 14]],
                    }
                )
            (self.root / f"{split}.json").write_text(
                json.dumps(
                    {"images": images, "annotations": annotations, "categories": cats}
                )
            )

    def body(self):
        import torch

        class Body(torch.nn.Module):
            out_channels = 8

            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward_features(self, x):
                return torch.nn.functional.adaptive_avg_pool2d(self.conv(x), (4, 4))

        return Body()

    def test_recipe_schedule_warmup_precedes_epoch_drops_and_scales_batch(self):
        import torch

        for dataset in ("coco2017", "livecell", "lvis_v1"):
            for adaptation in ("frozen", "attentive"):
                spec = self.api.training_recipe(dataset, adaptation)
                self.assertEqual(
                    (spec["epochs"], spec["optimizer"], spec["base_lr"]),
                    (12, "SGD", 0.02),
                )
                model = torch.nn.Linear(2, 2)
                opt, schedule = self.api.training_optimizer(
                    model, spec, effective_batch=8, updates_per_epoch=100
                )
                self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01 * 0.001)
                self.assertEqual(opt.param_groups[0]["momentum"], 0.9)
                for step, factor in (
                    (499, 0.001 + 0.999 * 499 / 500),
                    (500, 1.0),
                    (799, 1.0),
                    (800, 0.1),
                    (1100, 0.01),
                ):
                    self.assertAlmostEqual(schedule.lr_lambdas[0](step), factor)
                self.assertEqual(
                    sorted(g["weight_decay"] for g in opt.param_groups), [0.0, 0.0001]
                )
                _, slow = self.api.training_optimizer(
                    model, spec, effective_batch=16, updates_per_epoch=1
                )
                self.assertAlmostEqual(slow.lr_lambdas[0](11), 0.001 + 0.999 * 11 / 500)
        with self.assertRaises(ValueError):
            self.api.training_recipe("ava", "frozen")
        with self.assertRaises(ValueError):
            self.api.training_recipe("coco2017", "finetune")

    def test_split_ontology_and_path_overlap_are_rejected(self):
        self.data()
        cfg = self.config()
        train, val, meta = self.api.load_training_data(cfg)
        self.assertEqual((len(train), len(val), train.num_classes), (3, 1, 9))
        self.assertFalse(meta["official_membership_verified"])
        original = (self.root / "validation.json").read_text()
        for case in ("category", "path", "id"):
            obj = json.loads(original)
            if case == "category":
                obj["categories"][0]["name"] = "changed"
            if case == "path":
                obj["images"][0]["file_name"] = "train-0.png"
            if case == "id":
                obj["images"][0]["id"] = 1
                obj["annotations"][0]["image_id"] = 1
            (self.root / "validation.json").write_text(json.dumps(obj))
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.api.load_training_data(cfg)
        (self.root / "validation.json").write_text(original)
        obj = json.loads((self.root / "train.json").read_text())
        obj["categories"].pop()
        (self.root / "train.json").write_text(json.dumps(obj))
        with self.assertRaises(ValueError):
            self.api.load_training_data(cfg)

    def test_ragged_loader_and_outside_boxes_preserve_correspondence(self):
        import torch

        from downstream.extended_distributed import Session

        self.data()
        train, _, _ = self.api.load_training_data(self.config())
        loader = Session(torch.device("cpu")).loader(
            train,
            {"batch_size": 2, "num_workers": 0},
            0,
            collate_fn=self.api.detection_collate,
        )
        images, _targets = next(iter(loader))
        self.assertEqual(len(images), 2)
        self.assertNotEqual(images[0].shape, images[1].shape)
        target = {
            "boxes": torch.tensor([[-20.0, -20.0, -2.0, -2.0], [1.0, 1.0, 10.0, 10.0]]),
            "labels": torch.tensor([2, 1]),
            "masks": torch.ones(2, 24, 32, dtype=torch.uint8),
            "area": torch.tensor([324.0, 81.0]),
        }
        _, clean = self.api.training_batch(
            ([torch.ones(3, 24, 32)], [target]), torch.device("cpu")
        )
        self.assertEqual(clean[0]["labels"].tolist(), [1])
        self.assertEqual(clean[0]["masks"].shape[0], 1)
        self.assertEqual(len(target["boxes"]), 2)

    def test_configuration_validates_provider_profile_precision_and_tail(self):
        cfg = self.config(adaptation="attentive")
        plan = self.api.validate_training_config(cfg)
        self.assertEqual(plan["tail_policy"], "discard")
        for key, value in (
            ("geometry_profile", "guess"),
            ("reader_profile", "captured_cross_self_v1"),
            ("seed", 1),
            ("adaptation", "finetune"),
        ):
            bad = copy.deepcopy(cfg)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_training_config(bad)
        bad = copy.deepcopy(cfg)
        bad["execution"] = {
            "accumulation_steps": 2,
            "precision": "bf16",
            "tail_policy": "discard",
        }
        with self.assertRaises(ValueError):
            self.api.validate_training_config(bad)

    def run_fixture(self, cfg, out):
        import torch

        out.mkdir(exist_ok=True)
        original = self.api.build_extended_detector

        def small(*args, **kwargs):
            detector = original(*args, **kwargs)
            detector.transform.min_size = (32,)
            detector.transform.max_size = 40
            detector.rpn._pre_nms_top_n = {"training": 16, "testing": 16}
            detector.rpn._post_nms_top_n = {"training": 8, "testing": 8}
            detector.roi_heads.detections_per_img = 4
            return detector

        torch.manual_seed(7)
        body = self.body()
        with (
            patch.object(self.api, "build_frozen_backbone", return_value=body),
            patch.object(self.api, "build_extended_detector", side_effect=small),
        ):
            self.api.train_run(cfg, out)
        self.assertTrue(
            all(p.grad is None and not p.requires_grad for p in body.parameters())
        )
        return torch.load(out / "resume.pt", weights_only=True)

    def test_box_mask_lp_ap_resume_exact_and_checkpoint_excludes_encoder(self):
        import torch

        for dataset in ("coco2017", "livecell"):
            self.data(dataset)
            for adaptation in ("frozen", "attentive"):
                cfg = self.config(dataset, adaptation)
                cfg["probe"]["epochs"] = 2
                cfg["execution"] = {
                    "accumulation_steps": 2,
                    "precision": "fp32",
                    "tail_policy": "discard",
                }
                base = self.root / f"{dataset}-{adaptation}"
                base.mkdir()
                full = self.run_fixture(cfg, base / "full")
                cfg["probe"]["epochs"] = 1
                first = self.run_fixture(cfg, base / "first")
                cfg["probe"]["epochs"] = 2
                cfg["resume"] = str(base / "first/resume.pt")
                resumed = self.run_fixture(cfg, base / "resumed")
                self.equal_tree(full, resumed)
                self.assertEqual(full["runtime"]["updates"], 2)
                self.assertEqual(full["runtime"]["discarded_microbatches"], 2)
                self.assertFalse(
                    any(k.startswith("detector.backbone.body.") for k in full["model"])
                )
                self.assertTrue(
                    any(k.startswith("detector.backbone.fpn.") for k in full["model"])
                )
                for key, value in full["model"].items():
                    torch.testing.assert_close(
                        value, resumed["model"][key], rtol=0, atol=0
                    )
                self.assertTrue(
                    any(
                        not torch.equal(first["model"][k], v)
                        for k, v in full["model"].items()
                    )
                )
                report = json.loads((base / "resumed/results.json").read_text())
                self.assertFalse(report["canonical_eligible"])
                self.assertEqual(report["final"]["images"], 1)

    def test_all_five_provider_tail_policies_are_enforced(self):
        from downstream import spatial_backbones as sb

        seen = []
        for kind, path in sb.discover_providers().items():
            provider = sb._load_provider(path)
            if not hasattr(provider, "EXTENDED_DENSE_READER"):
                continue
            for adaptation in ("frozen", "attentive"):
                cfg = self.config(adaptation=adaptation)
                cfg["backbone"]["kind"] = kind
                cfg["reader_profile"] = (
                    provider.EXTENDED_DENSE_READER
                    if adaptation == "attentive"
                    else None
                )
                expected = provider.EXTENDED_ACCUMULATION_TAILS[adaptation]
                self.assertEqual(
                    self.api.validate_training_config(cfg)["tail_policy"], expected
                )
                cfg["execution"] = {
                    "accumulation_steps": 2,
                    "precision": "fp32",
                    "tail_policy": "flush_scaled"
                    if expected == "discard"
                    else "discard",
                }
                with self.assertRaises(ValueError):
                    self.api.validate_training_config(cfg)
            seen.append(kind)
        self.assertEqual(len(seen), 5)

    def test_ap_short_tail_flushes_and_resume_preserves_state(self):
        import torch

        self.data()
        cfg = self.config(adaptation="attentive")
        cfg["backbone"]["kind"] = "dinov3_hf"
        cfg["execution"] = {
            "accumulation_steps": 2,
            "precision": "fp32",
            "tail_policy": "flush_scaled",
        }
        cfg["probe"]["epochs"] = 2
        full = self.run_fixture(cfg, self.root / "full")
        cfg["probe"]["epochs"] = 1
        self.run_fixture(cfg, self.root / "first")
        cfg["resume"] = str(self.root / "first/resume.pt")
        cfg["probe"]["epochs"] = 2
        resumed = self.run_fixture(cfg, self.root / "resumed")
        self.assertEqual(full["runtime"]["tail_updates"], 2)
        self.assertEqual(full["runtime"]["updates"], 4)
        for k, v in full["model"].items():
            torch.testing.assert_close(v, resumed["model"][k], rtol=0, atol=0)

    def test_cli_trains_evaluates_delivers_manifest_and_refuses_overwrite(self):
        import subprocess
        import sys

        from transformers import SiglipVisionConfig, SiglipVisionModel

        from downstream import contract

        self.data("coco2017", count=2)
        encoder = self.root / "encoder"
        SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                image_size=32,
                patch_size=16,
            )
        ).save_pretrained(encoder)
        example = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/extended_detection_training.json"
        )
        cfg = json.loads(example.read_text())
        self.assertEqual(cfg["backbone"]["img_size"], 384)
        self.assertEqual(cfg["backbone"]["patch_size"], 16)
        cfg.update(
            device="cpu",
            data_root=str(self.root),
            train_annotations=str(self.root / "train.json"),
            validation_annotations=str(self.root / "validation.json"),
            geometry_profile="captured_224_256",
            probe={"epochs": 1, "batch_size": 1, "num_workers": 0},
            execution={
                "accumulation_steps": 1,
                "precision": "fp32",
                "tail_policy": "discard",
            },
        )
        cfg["backbone"].update(
            arch="fixture", encoder=str(encoder), img_size=32, patch_size=16
        )
        path = self.root / "config.json"
        path.write_text(json.dumps(cfg))
        out = self.root / "cli"
        args = [
            sys.executable,
            "-m",
            "downstream.extended_detection",
            "--config",
            str(path),
            "--out",
            str(out),
        ]
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=120, check=False
        )
        self.assertEqual(
            proc.returncode,
            0,
            proc.stderr + str(list(out.iterdir())) if out.exists() else proc.stderr,
        )
        self.assertEqual(contract.verify(out, path, 0), (True, []))
        report = json.loads((out / "results.json").read_text())
        self.assertEqual(report["task"], "extended_detection_training")
        self.assertEqual(report["execution"]["updates"], 2)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        repeated = subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        )
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_required_ci_runs_training_suite(self):
        import shlex

        import yaml

        workflow = yaml.safe_load(
            (
                Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml"
            ).read_text()
        )
        commands = [
            step.get("run", "")
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
        ]
        name = "tests.test_method_extended_detection_training"

        def runs_suite(command):
            for line in command.replace("\\\n", " ").splitlines():
                row = shlex.split(line, comments=True)
                while row and "=" in row[0]:
                    row.pop(0)
                if (
                    len(row) > 3
                    and Path(row[0]).name in ("python", "python3")
                    and row[1:3] == ["-m", "unittest"]
                    and name in row[3:]
                ):
                    return True
            return False

        self.assertTrue(
            runs_suite("PYTHONPATH=. .venv/bin/python -m unittest -v " + name)
        )
        self.assertFalse(runs_suite("echo .venv/bin/python -m unittest " + name))
        self.assertFalse(
            runs_suite(".venv/bin/python -m unittest tests.other # " + name)
        )
        self.assertFalse(runs_suite(".venv/bin/python -m unittest " + name + "_decoy"))
        self.assertTrue(any(runs_suite(command) for command in commands))

    def test_two_rank_gloo_training_and_continuation_match(self):
        import socket
        import subprocess
        import sys

        import torch

        self.data("coco2017", count=4)
        cfg = self.config("coco2017")
        cfg["execution"] = {
            "accumulation_steps": 2,
            "precision": "fp32",
            "tail_policy": "discard",
        }

        def launch(config, out):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            path = out.with_suffix(".json")
            path.write_text(json.dumps(config))
            proc = subprocess.run(
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
                    str(path),
                    str(out),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return torch.load(out / "resume.pt", weights_only=True)

        cfg["probe"]["epochs"] = 2
        full = launch(cfg, self.root / "full")
        cfg["probe"]["epochs"] = 1
        launch(cfg, self.root / "first")
        cfg["probe"]["epochs"] = 2
        cfg["resume"] = str(self.root / "first/resume.pt")
        resumed = launch(cfg, self.root / "resumed")
        self.equal_tree(full, resumed)
        self.assertEqual(full["runtime"]["world_size"], 2)
        self.assertEqual(len(full["rng"]), 2)
        self.assertEqual(full["runtime"]["updates"], 2)
        for key, value in full["model"].items():
            torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
        self.assertEqual(
            json.loads((self.root / "resumed/results.json").read_text())["final"][
                "images"
            ],
            1,
        )

    def test_lvis_training_delivers_federated_box_and_mask_scores(self):
        self.data("lvis_v1", count=1)
        for adaptation in ("frozen", "attentive"):
            cfg = self.config("lvis_v1", adaptation)
            self.run_fixture(cfg, self.root / adaptation)
            report = json.loads((self.root / adaptation / "results.json").read_text())
            self.assertEqual(report["final"]["evaluation_max_detections"], 300)
            self.assertIn("bbox_ap", report["final"])
            self.assertIn("mask_ap", report["final"])
            self.assertFalse(report["record_value"])

    def test_missing_lvis_federated_metadata_fails_before_model_construction(self):
        self.data("lvis_v1", count=1)
        cfg = self.config("lvis_v1")
        p = self.root / "validation.json"
        obj = json.loads(p.read_text())
        obj["images"][0].pop("neg_category_ids")
        p.write_text(json.dumps(obj))
        out = self.root / "invalid"
        out.mkdir()
        with (
            patch.object(self.api, "build_frozen_backbone", return_value=self.body()),
            patch.object(
                self.api,
                "DetectionProbe",
                side_effect=AssertionError(
                    "model constructed before membership validation"
                ),
            ) as build,
        ):
            with self.assertRaises(ValueError):
                self.api.train_run(cfg, out)
            build.assert_not_called()

    def test_real_five_provider_training_routes_deliver_outputs(self):
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
        original = self.api.build_extended_detector

        def small(*args, **kwargs):
            model = original(*args, **kwargs)
            model.rpn._pre_nms_top_n = {"training": 16, "testing": 16}
            model.rpn._post_nms_top_n = {"training": 8, "testing": 8}
            model.roi_heads.detections_per_img = 4
            return model

        for dataset in ("coco2017", "livecell"):
            self.data(dataset, count=1)
            for spec in specs:
                provider = sb._load_provider(sb.discover_providers()[spec["kind"]])
                for adaptation in ("frozen", "attentive"):
                    cfg = self.config(dataset, adaptation)
                    cfg["backbone"] = spec
                    cfg["reader_profile"] = (
                        provider.EXTENDED_DENSE_READER
                        if adaptation == "attentive"
                        else None
                    )
                    out = self.root / f"{dataset}-{spec['kind']}-{adaptation}"
                    out.mkdir()
                    with (
                        self.subTest(
                            dataset=dataset, kind=spec["kind"], adaptation=adaptation
                        ),
                        patch.object(
                            self.api, "build_extended_detector", side_effect=small
                        ),
                    ):
                        self.api.train_run(cfg, out)
                    checkpoint = torch.load(out / "resume.pt", weights_only=True)
                    self.assertEqual(checkpoint["runtime"]["updates"], 1)
                    self.assertTrue(checkpoint["optimizer"]["state"])
                    self.assertFalse(
                        any(
                            k.startswith("detector.backbone.body.")
                            for k in checkpoint["model"]
                        )
                    )
                    self.assertEqual(
                        json.loads((out / "results.json").read_text())["final"][
                            "images"
                        ],
                        1,
                    )

    def test_invalid_schedule_and_probe_settings_fail_closed(self):
        import torch

        spec = self.api.training_recipe("coco2017", "frozen")
        for batch, steps in ((0, 1), (1, 0), (True, 1), (1, True)):
            with self.assertRaises(ValueError):
                self.api.training_optimizer(
                    torch.nn.Linear(2, 2),
                    spec,
                    effective_batch=batch,
                    updates_per_epoch=steps,
                )
        for key, value in (
            ("optimizer", "Adam"),
            ("epochs", 13),
            ("schedule", "cosine"),
            ("warmup", "unknown"),
        ):
            bad = dict(spec)
            bad[key] = value
            with self.assertRaises(ValueError):
                self.api.training_optimizer(
                    torch.nn.Linear(2, 2), bad, effective_batch=2, updates_per_epoch=1
                )
        for key, value in (
            ("epochs", 13),
            ("batch_size", 0),
            ("num_workers", -1),
            ("epochs", True),
        ):
            cfg = self.config()
            cfg["probe"][key] = value
            with self.assertRaises(ValueError):
                self.api.validate_training_config(cfg)
        cfg = self.config()
        cfg["unexpected"] = 1
        with self.assertRaises(ValueError):
            self.api.validate_training_config(cfg)

    def test_captured_warmup_floating_point_order_is_exact(self):
        import torch

        _, scheduler = self.api.training_optimizer(
            torch.nn.Linear(2, 2),
            self.api.training_recipe("coco2017", "frozen"),
            effective_batch=16,
            updates_per_epoch=100,
        )
        for step in range(500):
            self.assertEqual(
                scheduler.lr_lambdas[0](step), 0.001 + (1.0 - 0.001) * (step / 500)
            )

    def test_fixture_patch_does_not_leak_into_public_factory(self):
        import subprocess
        import sys

        script = """
import sys
from tests.test_method_extended_detection_training import TestDetectionTraining
assert 'downstream.native_detection' not in sys.modules
helper=TestDetectionTraining();helper.setUp()
try:
    helper.data('coco2017',count=1)
    helper.run_fixture(helper.config('coco2017'),helper.root/'out')
    from downstream import native_detection
    model=native_detection.build_extended_detector(helper.body(),3,dataset='coco2017',reader_profile=None,geometry_profile='captured_224_256')
    assert model.transform.min_size==(224,),model.transform.min_size
    assert model.transform.max_size==256,model.transform.max_size
finally:helper.doCleanups()
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


def _distributed_worker(rank, cfg, out, port):
    import os

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    helper = TestDetectionTraining()
    helper.setUp()
    try:
        helper.run_fixture(cfg, Path(out))
    finally:
        helper.doCleanups()


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _distributed_worker(
            int(os.environ["RANK"]),
            json.loads(Path(sys.argv[2]).read_text()),
            sys.argv[3],
            int(os.environ["MASTER_PORT"]),
        )
    else:
        unittest.main()
