"""Behavioral coverage for expanded, explicitly selected training recipes."""
import copy
import unittest
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from unittest import mock

try:
    import torch
    from downstream import optimization as opt
    from downstream import spatial_backbones as sb
    import einops
    import timm
    HAVE = True
except ImportError:
    HAVE = False


class TestCoverageDelivery(unittest.TestCase):
    def test_ci_executes_both_modules(self):
        from tests.test_ci import HAVE_YAML, WORKFLOWS, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML or not WORKFLOWS.is_dir():
            self.skipTest("workflow source and YAML required")
        commands = [s["run"] for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
                    if s.get("name") == "Run Basic5 component contracts with downstream dependencies"]
        self.assertEqual(len(commands), 1)
        for module in ("tests.test_method_vjepa2_1_training", "tests.test_basic5_imagenet"):
            self.assertTrue(_runs_finetune_tests(commands[0], module=module))
            self.assertFalse(_runs_finetune_tests("echo " + module, module=module))

    def test_documented_full_imagenet_example_parses_and_validates(self):
        import re
        document = (Path(__file__).resolve().parents[1]/"docs/DOWNSTREAM.md").read_text()
        match = re.search(r"<!-- online-imagenet-example -->\s*```json\s*(.*?)```", document, re.S)
        self.assertIsNotNone(match, "online ImageNet configuration example missing")
        cfg = json.loads(match[1])
        self.assertEqual(cfg["task"], "imagenet_classification")
        if HAVE:
            from downstream.imagenet import validate_config
            validate_config(cfg)


@unittest.skipUnless(HAVE, "torch and author encoder dependencies required")
class TestFinetuneOptimization(unittest.TestCase):
    def config(self, task="ade20k_segmentation"):
        return dict(task=task, profile="capture_basic5_components", adaptation="finetune",
                    optimizer_profile="basic5_finetune_v1",
                    backbone=dict(kind="vjepa2_1", arch="vit_large", encoder="",
                                  img_size=384, patch_size=16, embed_dim=48, depth=12, num_heads=4),
                    **{("detector" if task == "coco_detection" else "probe"):
                       dict(lr="protocol", batch_size=8, epochs=1, max_steps_per_epoch=0)})

    def test_four_tasks_accept_verified_ft_and_scale_their_rates(self):
        for task, algorithm, lr, decay in (
                ("ade20k_segmentation", "AdamW", .0001, .8),
                ("nyuv2_depth", "AdamW", .0001, .8),
                ("coco_detection", "SGD", .01, 1.),
                ("ssv2_video", "AdamW", .0005 * 8 / 256, .75)):
            with self.subTest(task=task):
                report = opt.resolve_optimization(self.config(task), task)
                self.assertEqual(report["optimizer"], algorithm)
                self.assertEqual(report["lr"], lr)
                self.assertEqual(report["layer_decay"], decay)

    def test_groups_cover_encoder_and_head_and_produce_expected_updates(self):
        cfg = self.config()
        model = torch.nn.Module()
        model.backbone = sb.build_trainable_backbone(cfg["backbone"], torch.device("cpu"))
        model.classifier = torch.nn.Linear(48, 3)
        report = opt.resolve_optimization(cfg, cfg["task"])
        optimizer = opt.build_optimizer(model, report)
        groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
        self.assertEqual(sum(len(g["params"]) for g in optimizer.param_groups),
                         len(list(model.parameters())))
        self.assertEqual(set(groups), {id(p) for p in model.parameters()})
        expected = {
            "backbone.encoder.patch_embed_img.proj.weight": (.0001 * .8 ** 13, .05),
            "backbone.encoder.blocks.0.attn.qkv.weight": (.0001 * .8 ** 12, .05),
            "backbone.encoder.blocks.11.attn.qkv.weight": (.0001 * .8, .05),
            "backbone.encoder.blocks.0.norm1.weight": (.0001 * .8 ** 12, 0.),
            "classifier.weight": (.0001, .05), "classifier.bias": (.0001, 0.),
        }
        parameters = dict(model.named_parameters())
        for name, (lr, wd) in expected.items():
            p = parameters[name]
            g = groups[id(p)]
            self.assertAlmostEqual(g["lr"], lr)
            self.assertEqual(g["weight_decay"], wd)
            reference = torch.nn.Parameter(p.detach().clone())
            independent = torch.optim.AdamW([reference], lr=lr, weight_decay=wd)
            for _ in range(2):
                p.grad = torch.ones_like(p)
                reference.grad = torch.ones_like(reference)
                independent.step()
                # Isolate this parameter to compare real optimizer state updates.
                for other in model.parameters():
                    if other is not p:
                        other.grad = None
                optimizer.step()
                torch.testing.assert_close(p, reference, rtol=0, atol=0)

    def test_unverified_provider_and_wrong_track_are_refused(self):
        for change in ({"adaptation": "frozen"}, {"backbone": {"kind": "vit"}},
                       {"profile": "legacy"}):
            cfg = self.config()
            cfg.update(change)
            with self.assertRaises(ValueError):
                opt.resolve_optimization(cfg, cfg["task"])

    def test_reference_schedules_cover_all_four_tasks_and_tracks(self):
        for task, epochs, batch in (("ade20k_segmentation", 20, 8),
                                   ("nyuv2_depth", 30, 8),
                                   ("coco_detection", 12, 16),
                                   ("ssv2_video", 50, 256)):
            for adaptation in ("frozen", "attentive", "finetune"):
                cfg = self.config(task)
                cfg.update(adaptation=adaptation, scheduler_profile="basic5_reference_schedule_v1",
                           optimizer_profile="basic5_finetune_v1" if adaptation == "finetune" else "basic5_frozen_v1")
                settings = cfg["detector" if task == "coco_detection" else "probe"]
                settings.update(epochs=epochs, batch_size=batch)
                with self.subTest(task=task, adaptation=adaptation):
                    report = opt.resolve_optimization(cfg, task)
                    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=report["lr"])
                    scheduler = opt.build_task_scheduler(optimizer, report, 100)
                    self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                                           report["lr"] * .001 if task == "coco_detection" else 1e-6)
                    for update in range(epochs * 100):
                        if task == "coco_detection":
                            factor = (.001 + .999*update/500 if update < 500 else
                                      .01 if update >= 1100 else .1 if update >= 800 else 1.)
                            rate = report["lr"] * factor
                        else:
                            warmup = 500 if task == "ssv2_video" else 100
                            floor = 0. if task == "ssv2_video" and adaptation == "frozen" else 1e-6
                            rate = (1e-6 + (report["lr"]-1e-6)*update/warmup if update < warmup else
                                    floor + (report["lr"]-floor)*(1+math.cos(math.pi*(update-warmup)/(epochs*100-warmup)))/2)
                        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], rate, places=14)
                        optimizer.step()
                        scheduler.step()
                    expected = (report["lr"] * .01 if task == "coco_detection" else
                                (0. if task == "ssv2_video" and adaptation == "frozen" else 1e-6))
                    self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)
                    bad = copy.deepcopy(cfg)
                    bad["detector" if task == "coco_detection" else "probe"]["batch_size"] = batch // 2
                    with self.assertRaises(ValueError):
                        opt.resolve_optimization(bad, task)
                    bad = copy.deepcopy(cfg)
                    bad["detector" if task == "coco_detection" else "probe"]["epochs"] = epochs-1
                    with self.assertRaises(ValueError):
                        opt.resolve_optimization(bad, task)

    def test_invalid_parameter_policies_and_schedule_clocks_are_rejected(self):
        cfg = self.config()
        report = opt.resolve_optimization(cfg, cfg["task"])
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            opt.build_optimizer(model, report)
        model.finetune_group_policy = lambda: (2, {})
        with self.assertRaisesRegex(ValueError, "cover every"):
            opt.build_optimizer(model, report)
        for layer in (-1, 3, True, 1.5):
            model.finetune_group_policy = lambda: (2, {n: (layer, False) for n, p in model.named_parameters()})
            with self.assertRaisesRegex(ValueError, "outside"):
                opt.build_optimizer(model, report)
        for steps in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive"):
                opt.build_task_scheduler(None, {"scheduler_profile": opt.REFERENCE_SCHEDULE}, steps)

    @unittest.skipUnless(all(importlib.util.find_spec(n) for n in
                            ("scipy", "h5py", "av", "pycocotools")),
                         "Full downstream dependencies required")
    def test_real_clis_execute_reference_schedules_and_ft_groups(self):
        from tests.test_basic5_optimization import TestOptimization, nyuv2, coco, ssv2
        from scipy.io import savemat
        from downstream import contract
        for module, fixture, factory in TestOptimization().cases():
            for scheduled in (False, True):
                with self.subTest(task=module.TASK, scheduled=scheduled), tempfile.TemporaryDirectory() as d:
                    root = Path(d) / "data"
                    epochs, batch = {"ade20k_segmentation": (20, 8), "nyuv2_depth": (30, 8),
                                     "coco_detection": (12, 16), "ssv2_video": (50, 256)}[module.TASK]
                    batch = batch if scheduled else 2
                    if module is nyuv2:
                        fixture(root, n=batch+2)
                        savemat(root / "labeled/splits.mat", {"trainNdxs": [[i] for i in range(1, batch+1)],
                                                             "testNdxs": [[batch+1], [batch+2]]})
                    elif module is ssv2:
                        fixture(root)
                        labels = root / "labels/train.json"
                        rows = json.loads(labels.read_text())
                        for i in range(2, batch):
                            (root / f"videos/extra{i}.webm").write_bytes((root / "videos/vid0.webm").read_bytes())
                            rows.append({"id": f"extra{i}", "template": "a"})
                        labels.write_text(json.dumps(rows))
                    else:
                        fixture(root, per=batch)
                    cfg = TestOptimization().config(module, factory, root, "frozen" if scheduled else "finetune")
                    settings = cfg["detector" if module is coco else "probe"]
                    settings.update(batch_size=batch, epochs=epochs if scheduled else 1, max_val_samples=2)
                    if scheduled:
                        cfg["scheduler_profile"] = opt.REFERENCE_SCHEDULE
                    else:
                        cfg["optimizer_profile"] = opt.FT_PROFILE
                        cfg["backbone"] = self.config()["backbone"]
                    path, out = Path(d) / "config.json", Path(d) / "out"
                    path.write_text(json.dumps(cfg))
                    rates = []
                    algorithm = torch.optim.SGD if module is coco or (scheduled and module is ssv2) else torch.optim.AdamW
                    original_step = algorithm.step
                    def step(optimizer, *args, **kwargs):
                        rates.append([g["lr"] for g in optimizer.param_groups])
                        return original_step(optimizer, *args, **kwargs)
                    with mock.patch.object(algorithm, "step", step):
                        self.assertEqual(module.main(["--config", str(path), "--out", str(out)]), 0)
                    result = json.loads((out / "results.json").read_text())
                    report = result["optimization"]
                    if scheduled:
                        self.assertEqual(len(rates), epochs)
                        expected = .02 * .001 if module is coco else 1e-6
                        self.assertAlmostEqual(rates[0][0], expected)
                        self.assertNotEqual(rates[0], rates[1])
                        self.assertEqual(report["schedule"]["profile"], opt.REFERENCE_SCHEDULE)
                        self.assertEqual(report["schedule"]["updates_completed"], epochs)
                        self.assertEqual(report["schedule"]["steps_per_epoch"], 1)
                    else:
                        self.assertTrue(report["parameter_groups"])
                        self.assertTrue(any(g["weight_decay"] == 0 for g in report["parameter_groups"].values()))
                    self.assertFalse(result["canonical_eligible"])
                    self.assertFalse(result["record_value"])
                    self.assertTrue(contract.verify(out, path, 0)[0])
