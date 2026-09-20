"""Reference-batch dense AP scheduling through numerical and real CLI contracts."""
import copy
import json
import math
import re
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_downstream_ade20k import needs_deps, needs_timm, tiny_ade, smoke_config as ade_config
from tests.test_downstream_nyuv2 import needs_nyuv2, tiny_nyuv2, smoke_config as nyu_config

try:
    import torch
    from downstream import optimization
except ImportError:
    torch = None

PROFILE = "dense_ap_cosine_v1"


def config(factory, root, epochs):
    cfg = factory(root)
    cfg.update(profile="capture_basic5_components", adaptation="attentive",
               optimizer_profile="basic5_frozen_v1", scheduler_profile=PROFILE)
    cfg["probe"].update(batch_size=8, lr="protocol", epochs=epochs,
                        max_train_samples=0, max_val_samples=0, max_steps_per_epoch=0)
    return cfg


class TestDenseScheduleCI(unittest.TestCase):
    def test_documented_selection_is_executable(self):
        guide = (Path(__file__).resolve().parents[1]/"docs/DOWNSTREAM.md").read_text()
        example = re.search(r"<!-- dense-ap-schedule-selection -->\s*```json\s*(.*?)```", guide, re.S)
        self.assertIsNotNone(example, "dense AP executable selection is missing")
        selection = json.loads(example.group(1))
        self.assertEqual(selection, {"profile": "capture_basic5_components", "adaptation": "attentive",
                                     "optimizer_profile": "basic5_frozen_v1", "scheduler_profile": PROFILE})
        if torch is not None:
            cfg = config(ade_config, Path("/absent"), 20)
            cfg.update(selection)
            optimization.resolve_optimization(cfg, cfg["task"])

    def test_downstream_job_executes_module(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        commands = [s["run"] for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
                    if s.get("name") == "Run Basic5 component contracts with downstream dependencies"]
        self.assertEqual(len(commands), 1)
        self.assertTrue(_runs_finetune_tests(commands[0], module="tests.test_dense_ap_schedule"))


@unittest.skipUnless(torch is not None, "torch required")
class TestDenseSchedule(unittest.TestCase):
    def test_selection_refuses_ambiguous_recipes(self):
        for task, epochs in (("ade20k_segmentation", 20), ("nyuv2_depth", 30)):
            cfg = config(ade_config, Path("/absent"), epochs)
            cfg["task"] = task
            self.assertEqual(optimization.resolve_optimization(cfg, task)["lr"], .001)
            for field, values in (("batch_size", (1, 4, 16, True)),
                                  ("epochs", (1, epochs-1, epochs+1, True, 1.5))):
                for value in values:
                    bad = copy.deepcopy(cfg)
                    bad["probe"][field] = value
                    with self.subTest(task=task, field=field, value=value), self.assertRaises(ValueError):
                        optimization.resolve_optimization(bad, task)
            for changes in ({"adaptation": "frozen"}, {"adaptation": "finetune"},
                            {"scheduler_profile": "typo"}, {"optimizer_profile": None},
                            {"task": "ssv2_video"}, {"profile": "legacy"}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    optimization.resolve_optimization({**cfg, **changes}, changes.get("task", task))
            bad = copy.deepcopy(cfg)
            del bad["optimizer_profile"]
            with self.assertRaises(ValueError):
                optimization.resolve_optimization(bad, task)

    def test_all_rates_outputs_gradients_and_adamw_updates(self):
        self.assertTrue(hasattr(optimization, "build_dense_ap_scheduler"), "dense AP schedule is missing")
        for epochs in (20, 30):
            for steps in (1, 2, 7):
                p = torch.nn.Parameter(torch.tensor([.25], dtype=torch.float64))
                q = torch.nn.Parameter(p.detach().clone())
                actual = torch.optim.AdamW([p], lr=.001, weight_decay=.05)
                expected = torch.optim.AdamW([q], lr=.001, weight_decay=.05)
                schedule = optimization.build_dense_ap_scheduler(actual, steps, epochs)
                for update in range(epochs * steps + 3):
                    rate = (1e-6 + (.001-1e-6)*update/steps if update < steps else
                            1e-6 + (.001-1e-6)*(1 + math.cos(math.pi * min(1., (update-steps)/((epochs-1)*steps))))/2)
                    self.assertAlmostEqual(actual.param_groups[0]["lr"], rate, places=15)
                    expected.param_groups[0]["lr"] = rate
                    actual.zero_grad(); expected.zero_grad()
                    a, b = (p * 2 - 1).square().sum(), (q * 2 - 1).square().sum()
                    torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)
                    a.backward(); b.backward()
                    torch.testing.assert_close(p.grad, q.grad, rtol=1e-13, atol=1e-13)
                    actual.step(); expected.step(); schedule.step()
                    torch.testing.assert_close(p, q, rtol=1e-13, atol=1e-13)
                self.assertEqual(schedule.last_epoch, epochs*steps+3)


class TestDenseScheduleCLI(unittest.TestCase):
    @needs_deps
    def test_failed_update_does_not_advance_schedule(self):
        from downstream import ade20k
        model = torch.nn.Conv2d(1, 2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        schedule = optimization.build_dense_ap_scheduler(opt, 2, 20)
        batch = (torch.ones(1, 1, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long))
        with mock.patch.object(opt, "step", side_effect=RuntimeError("update failed")):
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                ade20k._train_one_epoch(model, [batch], opt, torch.device("cpu"), None,
                                       adaptation="attentive", scheduler=schedule)
        self.assertEqual(schedule.last_epoch, 0)
        invalid = (torch.full((1, 1, 2, 2), float("nan")), batch[1])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            ade20k._train_one_epoch(model, [invalid], opt, torch.device("cpu"), None,
                                   adaptation="attentive", scheduler=schedule)
        self.assertEqual(schedule.last_epoch, 0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    @needs_timm
    @needs_nyuv2
    def test_cuda_clis(self):
        try:
            from scipy.io import savemat
        except ImportError:
            self.skipTest("official split fixture requires scipy")
        from downstream import ade20k, nyuv2
        self.check_cli(ade20k, ade_config, 20, "cuda")
        self.check_cli(nyuv2, nyu_config, 30, "cuda")

    @needs_timm
    def test_ade_real_cli(self):
        from downstream import ade20k
        self.check_cli(ade20k, ade_config, 20)

    @needs_nyuv2
    def test_nyu_real_cli(self):
        try:
            from scipy.io import savemat
        except ImportError:
            self.skipTest("official split fixture requires scipy")
        from downstream import nyuv2
        self.check_cli(nyuv2, nyu_config, 30)

    def check_cli(self, module, factory, epochs, device="cpu"):
        from downstream import contract
        for cap in (0, 1):
            with self.subTest(task=module.TASK, cap=cap), tempfile.TemporaryDirectory() as d:
                root, out = Path(d)/"data", Path(d)/"out"
                if epochs == 20:
                    tiny_ade(root, per=17)
                    name = "FrozenSegModel"
                else:
                    from scipy.io import savemat
                    tiny_nyuv2(root, n=19)
                    savemat(root/"labeled/splits.mat", {"trainNdxs": [[i] for i in range(1,18)],
                                                       "testNdxs": [[18], [19]]})
                    name = "FrozenDepthModel"
                cfg = config(factory, root, epochs)
                cfg["device"] = device
                cfg["probe"]["max_steps_per_epoch"] = cap
                path = Path(d)/"config.json"
                path.write_text(json.dumps(cfg))
                rates, models, snapshots = [], [], []
                original, step = getattr(module, name), torch.optim.AdamW.step
                def make(*args, **kwargs):
                    model = original(*args, **kwargs)
                    models.append(model)
                    snapshots.append({n:p.clone() for n,p in model.state_dict().items()})
                    return model
                def update(opt, *args, **kwargs):
                    rates.append(opt.param_groups[0]["lr"])
                    return step(opt, *args, **kwargs)
                with mock.patch.object(module, name, make), mock.patch.object(torch.optim.AdamW, "step", update):
                    self.assertEqual(module.main(["--config",str(path),"--out",str(out)]), 0)
                self.assertEqual(len(rates), epochs*(cap or 2))
                for i, rate in enumerate(rates):
                    expected = (1e-6+(.001-1e-6)*i/2 if i < 2 else
                                1e-6+(.001-1e-6)*(1+math.cos(math.pi*(i-2)/(2*(epochs-1))))/2)
                    self.assertAlmostEqual(rate, expected, places=15)
                state = models[0].state_dict()
                frozen = {n for n,p in models[0].named_parameters() if not p.requires_grad}
                self.assertTrue(frozen)
                for n in frozen | set(dict(models[0].named_buffers())):
                    torch.testing.assert_close(state[n], snapshots[0][n], rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(state[n], snapshots[0][n])
                                    for n,p in models[0].named_parameters() if p.requires_grad))
                result = json.loads((out/"results.json").read_text())
                report = result["optimization"]["schedule"]
                self.assertEqual(report["profile"], PROFILE)
                self.assertEqual(report["steps_per_epoch"], 2)
                self.assertEqual(report["warmup_updates"], 2)
                self.assertEqual(report["reference_epochs"], epochs)
                self.assertEqual(report["updates_completed"], len(rates))
                self.assertEqual(report["truncated"], cap == 1)
                self.assertEqual(result["subset_or_smoke"], cap == 1)
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])
                if not cap:
                    self.assertAlmostEqual(report["next_update_lr"], 1e-6, places=15)
                self.assertTrue(contract.verify(out, path, 0)[0])
