"""Explicit accumulated BasicFive dense-task execution contracts."""

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

HAVE = all(
    importlib.util.find_spec(n)
    for n in ("torch", "torchvision", "numpy", "PIL", "h5py", "scipy", "pycocotools")
)
if HAVE:
    import torch

    from downstream import ade20k, coco, nyuv2, optimization
    from tests.test_downstream_ade20k import smoke_config as ade_config
    from tests.test_downstream_ade20k import tiny_ade
    from tests.test_downstream_coco import smoke_config as coco_config
    from tests.test_downstream_coco import tiny_coco
    from tests.test_downstream_nyuv2 import smoke_config as nyu_config
    from tests.test_downstream_nyuv2 import tiny_nyuv2

KINDS = (
    "clip_hf",
    "siglip2_g",
    "cradiov4_h",
    "cosmos3_super_vm",
    "dinov3_hf",
    "raev2_k7",
    "vjepa2_1",
    "vggt_omega",
)


class Delivery(unittest.TestCase):
    def test_downstream_ci_executes_dense_contracts(self):
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
            _runs_finetune_tests(
                commands[0], module="tests.test_method_dense_execution"
            )
        )


@unittest.skipUnless(HAVE, "dense-task dependencies required")
class Dense(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def cases(self):
        return (
            (ade20k, ade_config, tiny_ade),
            (nyuv2, nyu_config, tiny_nyuv2),
            (coco, coco_config, tiny_coco),
        )

    def test_documented_examples_are_runnable_configs(self):
        root = Path(__file__).resolve().parents[1]
        for api, _, _ in self.cases():
            cfg = json.loads(
                (root / "docs/examples" / (api.TASK + "_execution.json")).read_text()
            )
            api.validate_config(cfg)
            report = optimization.resolve_optimization(cfg, api.TASK)
            assert isinstance(report, dict)
            self.assertEqual(report["effective_batch"], report["reference_batch"])
            self.assertEqual(
                report["scheduler_profile"], optimization.REFERENCE_SCHEDULE
            )

    def config(
        self, api, factory, root=Path("/unused"), kind="vjepa2_1", adaptation="frozen"
    ):
        cfg = factory(root)
        cfg.update(
            profile="capture_basic5_components",
            adaptation=adaptation,
            optimizer_profile="basic5_finetune_v1"
            if adaptation == "finetune"
            else "basic5_frozen_v1",
            execution={
                "profile": "captured_dense_v1",
                "accumulation_steps": 2,
                "precision": "fp32",
                "tail_policy": "discard",
            },
        )
        cfg["backbone"]["kind"] = kind
        if adaptation == "attentive":
            from downstream.spatial_backbones import attentive_profile

            selected = attentive_profile(kind)
            if selected is not None:
                cfg["reader_profile"] = selected
                if api is coco:
                    cfg["detector_profile"] = "captured_native_detection_v1"
        settings = cfg["detector" if api is coco else "probe"]
        settings.update(
            lr="protocol",
            batch_size=1,
            max_steps_per_epoch=0,
            max_train_samples=0,
            max_val_samples=0,
        )
        if api is coco:
            settings["anchor_sizes"] = [8, 16, 32, 64]
        return cfg

    def test_all_seventy_two_selections_and_effective_learning_rates(self):
        for api, factory, _ in self.cases():
            for kind in KINDS:
                for adaptation in ("frozen", "attentive", "finetune"):
                    with self.subTest(task=api.TASK, kind=kind, adaptation=adaptation):
                        cfg = self.config(
                            api, factory, kind=kind, adaptation=adaptation
                        )
                        api.validate_config(cfg)
                        report = optimization.resolve_optimization(cfg, api.TASK)
                        assert isinstance(report, dict)
                        self.assertEqual(report["effective_batch"], 2)
                        self.assertEqual(report["accumulation_steps"], 2)
                        self.assertEqual(
                            report["lr"],
                            report["base_lr"] * 2 / report["reference_batch"],
                        )
                        from downstream import dense_execution

                        plan = dense_execution.resolve(cfg)
                        self.assertEqual(
                            plan["clip_norm"],
                            1.0
                            if adaptation != "frozen"
                            or (api is coco and kind == "siglip2_g")
                            else None,
                        )

    def test_invalid_and_unsupported_settings_refused_before_data(self):
        for api, factory, _ in self.cases():
            original = self.config(api, factory)
            settings = "detector" if api is coco else "probe"
            changes: list[tuple[str, Any]] = [
                ("execution", dict(original["execution"], **{key: value}))
                for key, values in (
                    ("profile", ["bad"]),
                    ("accumulation_steps", [0, -1, True, 1.5]),
                    ("precision", ["fp16"]),
                    ("tail_policy", ["flush_scaled"]),
                    ("extra", [True]),
                )
                for value in values
            ]
            changes += [
                ("execution", None),
                ("profile", "legacy"),
                ("optimizer_profile", None),
                ("task", "other"),
                ("backbone", dict(original["backbone"], kind="vit")),
                (settings, dict(original[settings], max_steps_per_epoch=1)),
                (settings, dict(original[settings], epochs=101)),
            ]
            for field in ("epochs", "batch_size", "max_steps_per_epoch"):
                for value in (None, True, 1.5, "2", -1):
                    changes.append(
                        (settings, dict(original[settings], **{field: value}))
                    )
            changes.append((settings, dict(original[settings], batch_size=0)))
            no_optimizer = copy.deepcopy(original)
            del no_optimizer["optimizer_profile"]
            for bad in [no_optimizer] + [dict(original, **{k: v}) for k, v in changes]:
                with self.subTest(task=api.TASK, bad=bad):
                    with self.assertRaises(api.ConfigError):
                        api.validate_config(bad)
                    from downstream import dense_execution

                    with self.assertRaises(ValueError):
                        dense_execution.resolve(bad)
            with (
                mock.patch.dict(os.environ, WORLD_SIZE="2", RANK="0", LOCAL_RANK="0"),
                self.assertRaises(api.ConfigError),
            ):
                api.validate_config(self.config(api, factory, kind="raev2_k7"))
            with (
                mock.patch.object(
                    torch.distributed, "is_initialized", return_value=True
                ),
                mock.patch.object(torch.distributed, "get_world_size", return_value=2),
                self.assertRaises(api.ConfigError),
            ):
                api.validate_config(original)

    def test_precision_and_short_epochs_fail_before_any_update(self):
        from downstream import dense_execution

        for api, factory, _ in self.cases():
            cfg = self.config(api, factory)
            cfg["execution"]["precision"] = "bf16"
            with self.assertRaisesRegex(ValueError, "CUDA"):
                api.run(cfg, Path("/unused"))
            cfg["execution"]["precision"] = "fp32"
            model = torch.nn.Linear(2, 2)
            opt = torch.optim.SGD(model.parameters(), lr=0.2)
            before = copy.deepcopy(model.state_dict())
            with self.assertRaisesRegex(ValueError, "optimizer update"):
                dense_execution.prepare(cfg, torch.device("cpu"), opt, None, 1, 0.2)
            for k, v in before.items():
                torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)

    def test_batch_losses_keep_dense_reductions_in_float32(self):
        from downstream import dense_execution

        device = torch.device("cpu")
        prediction = torch.tensor(
            [[[[2.3, 0.2]], [[1.1, 3.7]]]], dtype=torch.bfloat16, requires_grad=True
        )
        image = torch.zeros(1, 3, 1, 2)
        labels = torch.tensor([[[0, 255]]])
        loss = dense_execution.batch_loss(
            lambda x: prediction, (image, labels), device, ade20k.TASK, "fp32"
        )
        expected = torch.nn.functional.cross_entropy(
            prediction.float(), labels, ignore_index=255
        )
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        assert prediction.grad is not None
        self.assertEqual(torch.count_nonzero(prediction.grad[..., 1]), 0)
        depth = torch.tensor([[[2.1, 1.3]]], dtype=torch.bfloat16, requires_grad=True)
        target = torch.tensor([[[3.0, 1.0]]])
        valid = torch.tensor([[[True, False]]])
        loss = dense_execution.batch_loss(
            lambda x: depth, (image, target, valid), device, nyuv2.TASK, "fp32"
        )
        expected = nyuv2.silog_loss(depth.float(), target, valid)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        assert depth.grad is not None
        self.assertEqual(depth.grad[0, 0, 1], 0)
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        targets = [{"labels": torch.tensor([1]), "boxes": torch.ones(1, 4)}]

        def detector(images, received):
            torch.testing.assert_close(images[0], image[0])
            self.assertEqual(received[0]["labels"].item(), 1)
            return {"first": parameter**2, "second": 3 * parameter}

        loss = dense_execution.batch_loss(
            detector, ([image[0]], targets), device, coco.TASK, "fp32"
        )
        self.assertEqual(loss.item(), 10.0)
        loss.backward()
        assert parameter.grad is not None
        self.assertEqual(parameter.grad.item(), 7.0)

    def test_reference_schedule_uses_effective_batch_and_microbatch_clock(self):
        from downstream import dense_execution

        for api, factory, _ in self.cases():
            for adaptation in ("frozen", "attentive", "finetune"):
                cfg = self.config(api, factory, adaptation=adaptation)
                cfg["scheduler_profile"] = optimization.REFERENCE_SCHEDULE
                cfg["execution"]["accumulation_steps"] = 16 if api is coco else 8
                api.validate_config(cfg)
                report = optimization.resolve_optimization(cfg, api.TASK)
                assert isinstance(report, dict)
                self.assertEqual(report["lr"], report["base_lr"])
                model = torch.nn.Linear(3, 2)
                opt = torch.optim.SGD(model.parameters(), lr=report["lr"])
                scheduler = optimization.build_task_scheduler(opt, report, 19)
                if scheduler is None:
                    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
                _, _, runtime = dense_execution.prepare(
                    cfg, torch.device("cpu"), opt, scheduler, 19, report["lr"]
                )
                self.assertEqual(runtime["schedule_horizon"], 19 * report["epochs"])
                self.assertEqual(runtime["schedule_steps_per_epoch"], 19)
                dense_execution.train_epoch(
                    model,
                    [torch.ones(2, 3)] * 19,
                    lambda x, model=model: model(x).square().mean(),
                    opt,
                    scheduler,
                    cfg,
                    dense_execution.resolve(cfg),
                )
                self.assertEqual(
                    scheduler.last_epoch, 19 // cfg["execution"]["accumulation_steps"]
                )

    def test_detection_body_guard_does_not_freeze_trainable_pyramid(self):
        from downstream import dense_execution

        cfg = self.config(coco, coco_config)
        plan = dense_execution.resolve(cfg)
        model = torch.nn.Module()
        model.backbone = cast(Any, torch.nn.Module())
        model.backbone.body = torch.nn.Linear(3, 3).requires_grad_(False)
        model.backbone.fpn = torch.nn.Linear(3, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        batch = torch.ones(2, 3)

        def loss(x):
            return (
                cast(Any, model.backbone)
                .fpn(cast(Any, model.backbone).body(x))
                .square()
                .mean()
            )

        dense_execution.train_epoch(model, [batch, batch], loss, opt, sched, cfg, plan)
        model.backbone.body.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            dense_execution.train_epoch(
                model, [batch, batch], loss, opt, sched, cfg, plan
            )

    def test_accumulated_updates_tail_and_clipping_match_independent_loop(self):
        from downstream import dense_execution

        for api, factory, _ in self.cases():
            for kind in KINDS:
                for adaptation in ("frozen", "attentive", "finetune"):
                    cfg = self.config(api, factory, kind=kind, adaptation=adaptation)
                    plan = dense_execution.resolve(cfg)
                    torch.manual_seed(71)
                    model = torch.nn.Linear(3, 2)
                    reference = copy.deepcopy(model)
                    opt = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9)
                    expected = torch.optim.SGD(
                        reference.parameters(), lr=0.3, momentum=0.9
                    )
                    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.9**i)
                    expected_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        expected, lambda i: 0.9**i
                    )
                    batches = [torch.randn(2, 3) * 20 for _ in range(5)]

                    def loss(net, batch):
                        return net(batch).square().mean()

                    stats = dense_execution.train_epoch(
                        model,
                        batches,
                        lambda b, model=model: loss(model, b),
                        opt,
                        scheduler,
                        cfg,
                        plan,
                    )
                    expected.zero_grad(set_to_none=True)
                    for index, batch in enumerate(batches):
                        (loss(reference, batch) / 2).backward()
                        if (index + 1) % 2 == 0:
                            if adaptation != "frozen" or (
                                api is coco and kind == "siglip2_g"
                            ):
                                torch.nn.utils.clip_grad_norm_(
                                    list(reference.parameters()), 1.0
                                )
                            expected.step()
                            expected_scheduler.step()
                            expected.zero_grad(set_to_none=True)
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        self.assertIsNone(a.grad)
                    self.assertEqual(
                        stats,
                        {
                            "microbatches": 5,
                            "updates": 2,
                            "tail_updates": 0,
                            "discarded_microbatches": 1,
                        },
                    )
                    self.assertEqual(scheduler.last_epoch, 2)

    @unittest.skipUnless(
        HAVE
        and importlib.util.find_spec("timm")
        and importlib.util.find_spec("einops"),
        "tiny encoder dependencies required",
    )
    def test_real_runners_accumulate_and_report_without_promoting_scores(self):
        from scipy.io import savemat

        from downstream import spatial_backbones

        for api, factory, fixture in self.cases():
            for adaptation in ("frozen", "attentive", "finetune"):
                with (
                    self.subTest(task=api.TASK, adaptation=adaptation),
                    tempfile.TemporaryDirectory() as d,
                ):
                    root, out = Path(d) / "data", Path(d) / "out"
                    fixture(root)
                    out.mkdir()
                    if api is nyuv2:
                        savemat(
                            root / "labeled/splits.mat",
                            {"trainNdxs": [[1], [2], [3]], "testNdxs": [[4], [5], [6]]},
                        )
                    cfg = self.config(api, factory, root, adaptation=adaptation)
                    # A real tiny encoder isolates runner behavior from released downloads.
                    spec = dict(cfg["backbone"], kind="vit")
                    real_builder = (
                        spatial_backbones.build_trainable_backbone
                        if adaptation == "finetune"
                        else spatial_backbones.build_frozen_backbone
                    )
                    builder_name = (
                        "build_trainable_backbone"
                        if adaptation == "finetune"
                        else "build_frozen_backbone"
                    )
                    if adaptation == "finetune":
                        # Tiny fixture policy assigns one layer to each encoder parameter.
                        def build(_spec, device, real_builder=real_builder, spec=spec):
                            value = cast(Any, real_builder(spec, device))
                            value.finetune_group_policy = lambda: (
                                1,
                                {
                                    n: (0, False)
                                    for n, p in value.named_parameters()
                                    if p.requires_grad
                                },
                            )
                            return value
                    else:

                        def build(_spec, device, real_builder=real_builder, spec=spec):
                            return real_builder(spec, device)

                    with mock.patch.object(api, builder_name, build):
                        api.run(cfg, out)
                    report = json.loads((out / "results.json").read_text())
                    self.assertFalse(report["record_value"])
                    self.assertFalse(report["canonical_eligible"])
                    runtime = report["execution"]
                    self.assertGreater(runtime["updates"], 0)
                    self.assertEqual(runtime["updates"], runtime["microbatches"] // 2)
                    self.assertEqual(runtime["effective_batch"], 2)
                    self.assertEqual(runtime["scaled_lr"], report["optimization"]["lr"])
