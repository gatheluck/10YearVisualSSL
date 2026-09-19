"""COCO schedule boundaries, actual updates and opt-in CLI provenance."""
import copy
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    from downstream import coco, contract
    from tests.test_downstream_coco import tiny_coco, smoke_config
    HAVE = True
except ImportError:
    HAVE = False

PROFILE = "coco_frozen_1x_v1"


class TestScheduleCI(unittest.TestCase):
    def test_ci_executes_schedule_tests(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        module = "tests.test_coco_schedule"
        command = ".venv/bin/python -m unittest " + module
        self.assertTrue(_runs_finetune_tests(command, module=module))
        for decoy in ("# " + command, "echo " + command, command + "_disabled",
                      ".venv/bin/python -m unittest tests.test_ci # " + module):
            self.assertFalse(_runs_finetune_tests(decoy, module=module))
        commands = [s["run"] for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
                    if s.get("name") == "Run Basic5 component contracts with downstream dependencies"]
        self.assertEqual(len(commands), 1)
        self.assertTrue(_runs_finetune_tests(commands[0], module=module))


@unittest.skipUnless(HAVE, "COCO dependencies required")
class TestSchedule(unittest.TestCase):
    def config(self, root, adaptation="frozen"):
        cfg = smoke_config(root)
        cfg.update(profile="capture_basic5_components", adaptation=adaptation,
                   optimizer_profile="basic5_frozen_v1", scheduler_profile=PROFILE)
        cfg["detector"].update(lr="protocol", batch_size=2, epochs=2,
                               anchor_sizes=[8, 16, 32, 64], max_train_samples=0,
                               max_val_samples=0, max_steps_per_epoch=1)
        return cfg

    def builder(self):
        from downstream import optimization
        self.assertTrue(hasattr(optimization, "build_coco_scheduler"),
                        "COCO schedule behavior has not been implemented")
        return optimization.build_coco_scheduler

    def test_selection_and_invalid_profiles_fail_before_data_access(self):
        cfg = self.config(Path("/missing"))
        for adaptation in ("frozen", "attentive"):
            coco.validate_config({**cfg, "adaptation": adaptation})
        legacy = smoke_config(Path("/missing"))
        coco.validate_config(legacy)
        for changes in ({"scheduler_profile": None}, {"scheduler_profile": "typo"},
                        {"optimizer_profile": None}, {"profile": "legacy"},
                        {"adaptation": "finetune"}):
            with self.subTest(changes=changes), self.assertRaises(coco.ConfigError):
                coco.validate_config({**cfg, **changes})
        bad = copy.deepcopy(cfg)
        del bad["optimizer_profile"]
        with self.assertRaises(coco.ConfigError):
            coco.validate_config(bad)
        for epochs in (0, 13, True, 1.5):
            bad = copy.deepcopy(cfg)
            bad["detector"]["epochs"] = epochs
            with self.subTest(epochs=epochs), self.assertRaises(coco.ConfigError):
                coco.validate_config(bad)
        from downstream.optimization import resolve_optimization
        other = copy.deepcopy(cfg)
        other["task"] = "ade20k_segmentation"
        other["probe"] = copy.deepcopy(cfg["detector"])
        with self.assertRaises(ValueError):
            resolve_optimization(other, "ade20k_segmentation")

    def test_schedule_horizon_and_step_cap_reporting(self):
        for epochs, cap in ((1, 0), (12, 0), (12, 1)):
            with self.subTest(epochs=epochs, cap=cap), tempfile.TemporaryDirectory() as d:
                root, out = Path(d)/"data", Path(d)/"out"
                tiny_coco(root, per=5)
                cfg = self.config(root)
                cfg["detector"].update(epochs=epochs, max_steps_per_epoch=cap)
                config = Path(d)/"config.json"
                config.write_text(json.dumps(cfg))
                metrics = {"bbox_mAP": 0., "bbox_mAP_50": 0., "detections": 0}
                with mock.patch.object(coco, "evaluate", return_value=metrics):
                    self.assertEqual(coco.main(["--config",str(config),"--out",str(out)]), 0)
                result = json.loads((out/"results.json").read_text())
                report = result["optimization"]["schedule"]
                self.assertEqual(report["updates_completed"], epochs*(cap or 2))
                self.assertEqual(report["truncated"], epochs < 12 or cap == 1)
                self.assertEqual(result["subset_or_smoke"], epochs < 12 or cap == 1)
                self.assertEqual(report["warmup_start_factor"], .001)
                self.assertEqual(report["decay_factor"], .1)
                self.assertEqual(report["reference_epochs"], 12)
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])

    def test_runner_milestones_use_full_loader_despite_step_cap(self):
        class LightweightDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))
                self.backbone = torch.nn.Module()
                self.backbone.body = torch.nn.Linear(1, 1).requires_grad_(False)
            def forward(self, images, targets):
                return {"loss": self.weight.square() * images[0].mean()}
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d)/"data", Path(d)/"out"
            tiny_coco(root, per=202)
            cfg = self.config(root)
            cfg["detector"].update(epochs=12, max_steps_per_epoch=100)
            config = Path(d)/"config.json"
            config.write_text(json.dumps(cfg))
            rates = []
            original = torch.optim.SGD.step
            def step(opt, *args, **kwargs):
                rates.append(opt.param_groups[0]["lr"])
                return original(opt, *args, **kwargs)
            metrics = {"bbox_mAP": 0., "bbox_mAP_50": 0., "detections": 0}
            with mock.patch.object(coco, "build_frozen_detector", return_value=LightweightDetector()), \
                 mock.patch.object(coco, "evaluate", return_value=metrics), \
                 mock.patch.object(torch.optim.SGD, "step", step):
                self.assertEqual(coco.main(["--config",str(config),"--out",str(out)]), 0)
            self.assertEqual(len(rates), 1200)
            base = .02*2/16
            for update in (499, 500, 799, 800, 807, 808, 1099, 1100, 1110, 1111):
                factor = (.001+.999*update/500 if update < 500 else
                          1 if update < 808 else .1 if update < 1111 else .01)
                self.assertAlmostEqual(rates[update], base*factor, places=15)

    def test_all_update_boundaries_and_parameter_updates(self):
        build = self.builder()
        for steps in (1, 100, 501):
            for base in (.00125, .02, .32):
                with self.subTest(steps=steps, base=base):
                    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
                    q = torch.nn.Parameter(p.detach().clone())
                    actual = torch.optim.SGD([p], lr=base, momentum=.9, weight_decay=.0001)
                    expected = torch.optim.SGD([q], lr=base, momentum=.9, weight_decay=.0001)
                    schedule = build(actual, steps)
                    for update in range(max(1201, 12 * steps)):
                        # Independent, piecewise specification indexed by the update
                        # about to execute; warmup takes precedence for tiny fixtures.
                        if update <= 499:
                            rate = base * (.001 + .999 * update / 500)
                        elif update < 8 * steps:
                            rate = base
                        elif update < 11 * steps:
                            rate = base / 10
                        else:
                            rate = base / 100
                        self.assertAlmostEqual(actual.param_groups[0]["lr"], rate, places=15)
                        expected.param_groups[0]["lr"] = rate
                        p.grad = torch.tensor([.25], dtype=torch.float64)
                        q.grad = p.grad.clone()
                        actual.step()
                        expected.step()
                        schedule.step()
                        torch.testing.assert_close(p, q, rtol=1e-13, atol=1e-13)
                    self.assertEqual(schedule.last_epoch, max(1201, 12 * steps))

    def test_loop_advances_only_after_successful_optimizer_updates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, images, targets):
                return {"loss": (self.weight * images[0]).square().sum()}
        build = self.builder()
        model = Model()
        opt = torch.optim.SGD(model.parameters(), lr=.02)
        schedule = build(opt, 4)
        batch = ([torch.ones(1)], [{}])
        observed = []
        original = opt.step
        def step(*args, **kwargs):
            observed.append(schedule.last_epoch)
            return original(*args, **kwargs)
        with mock.patch.object(opt, "step", step):
            for _ in range(2):
                coco._train_one_epoch(model, [batch]*4, opt, torch.device("cpu"), 1,
                                     scheduler=schedule)
        self.assertEqual(observed, [0, 1])
        self.assertEqual(schedule.last_epoch, 2)
        with mock.patch.object(opt, "step", side_effect=RuntimeError("update failed")):
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                coco._train_one_epoch(model, [batch], opt, torch.device("cpu"), None,
                                     scheduler=schedule)
        self.assertEqual(schedule.last_epoch, 2)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            coco._train_one_epoch(model, [([torch.tensor([float("nan")])], [{}])],
                                 opt, torch.device("cpu"), None, scheduler=schedule)
        self.assertEqual(schedule.last_epoch, 2)

    def test_real_cli_schedule_metadata_and_frozen_state(self):
        self.check_cli("cpu")

    @unittest.skipUnless(HAVE and torch.cuda.is_available(), "CUDA required")
    def test_cuda_cli(self):
        self.check_cli("cuda")

    def check_cli(self, device):
        for adaptation in ("frozen", "attentive"):
            with self.subTest(adaptation=adaptation), tempfile.TemporaryDirectory() as d:
                root, out = Path(d)/"data", Path(d)/"out"
                tiny_coco(root, per=5)
                cfg = self.config(root, adaptation)
                guide = (Path(__file__).resolve().parents[1]/"docs/DOWNSTREAM.md").read_text()
                example = re.search(r"<!-- coco-schedule-selection -->\s*```json\s*(.*?)```", guide, re.S)
                self.assertIsNotNone(example, "the executable schedule selection example is missing")
                del cfg["scheduler_profile"]
                cfg.update(json.loads(example.group(1)))
                cfg["device"] = device
                config = Path(d)/"config.json"
                config.write_text(json.dumps(cfg))
                models, snapshots, rates = [], [], []
                factory = coco.build_frozen_detector
                original_step = torch.optim.SGD.step
                def make(*args, **kwargs):
                    model = factory(*args, **kwargs)
                    models.append(model)
                    snapshots.append({n:p.clone() for n,p in model.state_dict().items()})
                    return model
                def step(opt, *args, **kwargs):
                    rates.append(opt.param_groups[0]["lr"])
                    return original_step(opt, *args, **kwargs)
                with mock.patch.object(coco, "build_frozen_detector", make), \
                     mock.patch.object(torch.optim.SGD, "step", step):
                    self.assertEqual(coco.main(["--config",str(config),"--out",str(out)]), 0)
                base = .02 * 2 / 16
                self.assertEqual(len(rates), 2)
                self.assertAlmostEqual(rates[0], base*.001)
                self.assertAlmostEqual(rates[1], base*(.001+.999/500))
                frozen = {n for n,p in models[0].named_parameters() if not p.requires_grad}
                self.assertTrue(frozen)
                for n in frozen | set(dict(models[0].named_buffers())):
                    torch.testing.assert_close(models[0].state_dict()[n], snapshots[0][n], rtol=0, atol=0)
                result = json.loads((out/"results.json").read_text())
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])
                self.assertTrue(result["subset_or_smoke"])
                report = result["optimization"]["schedule"]
                self.assertEqual(report["profile"], PROFILE)
                self.assertEqual(report["steps_per_epoch"], 2)
                self.assertEqual(report["updates_completed"], 2)
                self.assertEqual(report["warmup_updates"], 500)
                self.assertEqual(report["milestone_epochs"], [8, 11])
                self.assertTrue(report["truncated"])
                self.assertAlmostEqual(report["next_update_lr"], base*(.001+.999*2/500))
                self.assertTrue(contract.verify(out, config, 0)[0])


if __name__ == "__main__":
    unittest.main()
