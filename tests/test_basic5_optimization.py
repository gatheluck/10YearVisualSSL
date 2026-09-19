"""Opt-in frozen/AP optimizer behavior through the four real task CLIs."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    from scipy.io import savemat
    from downstream import ade20k, coco, nyuv2, ssv2, contract
    from tests.test_downstream_ade20k import tiny_ade, smoke_config as ade_config
    from tests.test_downstream_coco import tiny_coco, smoke_config as coco_config
    from tests.test_downstream_nyuv2 import tiny_nyuv2, smoke_config as nyu_config
    from tests.test_downstream_ssv2 import tiny_ssv2, smoke_config as video_config
    HAVE = True
except ImportError:
    HAVE = False

PROFILE = "basic5_frozen_v1"


class TestOptimizationCI(unittest.TestCase):
    def test_downstream_job_runs_optimization_with_real_dependencies(self):
        from tests.test_ci import HAVE_YAML, parsed
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        module = "tests.test_basic5_optimization"
        def runs(command):
            return _runs_finetune_tests(command, module=module)
        invocation = "PYTHONPATH=. .venv/bin/python -m unittest " + module
        self.assertTrue(runs(invocation))
        for decoy in ("# " + invocation, "echo " + invocation,
                      invocation + "_disabled", ".venv/bin/python -m unittest tests.test_ci # " + module):
            self.assertFalse(runs(decoy))
        commands = [s["run"] for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
                    if s.get("name") == "Run Basic5 component contracts with downstream dependencies"]
        self.assertEqual(len(commands), 1)
        self.assertTrue(runs(commands[0]))


@unittest.skipUnless(HAVE, "Full downstream dependencies required")
class TestOptimization(unittest.TestCase):
    def cases(self):
        return ((ade20k, tiny_ade, ade_config), (nyuv2, tiny_nyuv2, nyu_config),
                (coco, tiny_coco, coco_config), (ssv2, tiny_ssv2, video_config))

    def config(self, module, factory, root, adaptation="frozen"):
        cfg = factory(root)
        cfg.update(profile="capture_basic5_components", adaptation=adaptation,
                   optimizer_profile=PROFILE)
        settings = cfg["detector" if module is coco else "probe"]
        settings.update(lr="protocol", batch_size=2, max_train_samples=0,
                        max_val_samples=0, max_steps_per_epoch=0)
        if module is coco:
            settings["anchor_sizes"] = [8, 16, 32, 64]
        return cfg

    def test_explicit_selection_and_legacy_compatibility(self):
        for module, _, factory in self.cases():
            for adaptation in ("frozen", "attentive"):
                with self.subTest(task=module.TASK, adaptation=adaptation):
                    module.validate_config(self.config(module, factory, Path("/unused"), adaptation))
            module.validate_config(factory(Path("/unused")))

    def test_refuses_unsupported_or_ignored_settings_before_data_access(self):
        for module, _, factory in self.cases():
            cfg = self.config(module, factory, Path("/missing-dataset"))
            key = "detector" if module is coco else "probe"
            invalid = [{**cfg, "profile": "legacy"}, {**cfg, "adaptation": "finetune"},
                       {**cfg, "task": "unrelated_task"},
                       {**cfg, "optimizer_profile": "typo"},
                       {**cfg, "optimizer_profile": None}]
            for value in (0, -1, True, 2.5, "2"):
                bad = copy.deepcopy(cfg)
                bad[key]["batch_size"] = value
                invalid.append(bad)
            for value in (.001, float("nan"), None):
                bad = copy.deepcopy(cfg)
                bad[key]["lr"] = value
                invalid.append(bad)
            for field, values in (("epochs", (0, -1, True, 1.5)),
                                  ("max_steps_per_epoch", (-1, True, 1.5))):
                for value in values:
                    bad = copy.deepcopy(cfg)
                    bad[key][field] = value
                    invalid.append(bad)
            for bad in invalid:
                with self.subTest(task=module.TASK, cfg=bad):
                    with self.assertRaises(module.ConfigError):
                        module.validate_config(bad)
            for world in ("2", "0", "bad"):
                with mock.patch.dict(os.environ, {"WORLD_SIZE": world}):
                    with self.assertRaises(module.ConfigError):
                        module.validate_config(cfg)
            with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
                 mock.patch.object(torch.distributed, "get_world_size", return_value=2):
                with self.assertRaises(module.ConfigError):
                    module.validate_config(cfg)

    def test_optimizer_two_updates_match_independent_torch_construction(self):
        from downstream.optimization import build_optimizer, resolve_optimization
        for module, _, factory in self.cases():
            for adaptation in ("frozen", "attentive"):
                for batch in (1, 8, 256):
                    with self.subTest(task=module.TASK, adaptation=adaptation, batch=batch):
                        cfg = self.config(module, factory, Path("/unused"), adaptation)
                        cfg["detector" if module is coco else "probe"]["batch_size"] = batch
                        torch.manual_seed(18)
                        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)).double()
                        model[0].requires_grad_(False)
                        ref = copy.deepcopy(model)
                        actual = build_optimizer(model, resolve_optimization(cfg, module.TASK))
                        sgd = module is coco or (module is ssv2 and adaptation == "frozen")
                        base = .02 if module is coco else (.1 if sgd else .001)
                        ref_batch = 16 if module is coco else (256 if module is ssv2 else 8)
                        wd = .0001 if sgd or adaptation == "frozen" else .05
                        params = [p for p in ref.parameters() if p.requires_grad]
                        expected = (torch.optim.SGD(params, lr=base*batch/ref_batch, weight_decay=wd, momentum=.9)
                                    if sgd else torch.optim.AdamW(params, lr=base*batch/ref_batch,
                                                                 weight_decay=wd, betas=(.9, .999)))
                        for step in range(2):
                            x = torch.tensor([[1., -2., 3.], [2., 1., -1.]], dtype=torch.float64)
                            for net, opt in ((model, actual), (ref, expected)):
                                opt.zero_grad(set_to_none=True)
                                (net(x) - step - 1).square().mean().backward()
                                opt.step()
                            for p, q in zip(model.parameters(), ref.parameters()):
                                torch.testing.assert_close(p, q, rtol=0, atol=0)
        model.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "no trainable parameters"):
            build_optimizer(model, resolve_optimization(cfg, module.TASK))

    def test_real_clis_use_scaled_optimizer_and_preserve_frozen_state(self):
        self.check_clis("cpu")

    @unittest.skipUnless(HAVE and torch.cuda.is_available(), "CUDA required")
    def test_cuda_real_clis(self):
        self.check_clis("cuda")

    def check_clis(self, device):
        for module, fixture, factory in self.cases():
            for adaptation in ("frozen", "attentive"):
                with self.subTest(task=module.TASK, adaptation=adaptation), tempfile.TemporaryDirectory() as d:
                    root = Path(d) / "data"
                    if module in (ade20k, coco):
                        fixture(root, per=3)
                    else:
                        fixture(root)
                    if module is ssv2:
                        labels = root / "labels/train.json"
                        rows = json.loads(labels.read_text())
                        self.assertEqual(len(rows), 2)
                        (root / "videos/extra.webm").write_bytes((root / "videos/vid0.webm").read_bytes())
                        labels.write_text(json.dumps(rows + [{"id": "extra", "template": "a"}]))
                    if module is nyuv2:
                        savemat(root / "labeled/splits.mat", {"trainNdxs": [[1], [2], [3]],
                                                           "testNdxs": [[4], [5], [6]]})
                    cfg = self.config(module, factory, root, adaptation)
                    cfg["device"] = device
                    config, out = Path(d) / "cfg.json", Path(d) / "out"
                    config.write_text(json.dumps(cfg))
                    is_sgd = module is coco or (module is ssv2 and adaptation == "frozen")
                    opt_cls = torch.optim.SGD if is_sgd else torch.optim.AdamW
                    base = .02 if module is coco else (.1 if is_sgd else .001)
                    reference_batch = 16 if module is coco else (256 if module is ssv2 else 8)
                    expected_lr = base * 2 / reference_batch
                    expected_wd = .0001 if is_sgd or adaptation == "frozen" else .05
                    models, snapshots, updates, batch_sizes = [], [], [], []
                    factory_name = {ade20k: "FrozenSegModel", nyuv2: "FrozenDepthModel",
                                    coco: "build_frozen_detector",
                                    ssv2: "FrozenFrameAverageClassifier"}[module]
                    original_factory = getattr(module, factory_name)
                    original_step = opt_cls.step

                    def make_model(*args, **kwargs):
                        model = original_factory(*args, **kwargs)
                        models.append(model)
                        snapshots.append({n: p.detach().clone() for n, p in model.state_dict().items()})
                        model.register_forward_pre_hook(lambda m, args: batch_sizes.append(len(args[0]))
                                                        if m.training else None)
                        return model

                    def step(opt, *args, **kwargs):
                        params = [p for g in opt.param_groups for p in g["params"]]
                        self.assertEqual(len(params), len({id(p) for p in params}))
                        self.assertEqual({id(p) for p in params},
                                         {id(p) for p in models[0].parameters() if p.requires_grad})
                        self.assertTrue(all(p.requires_grad for p in params))
                        for group in opt.param_groups:
                            self.assertAlmostEqual(group["lr"], expected_lr)
                            self.assertAlmostEqual(group["weight_decay"], expected_wd)
                            if is_sgd:
                                self.assertEqual(group["momentum"], .9)
                            else:
                                self.assertEqual(group["betas"], (.9, .999))
                        before = [p.detach().clone() for p in params]
                        result = original_step(opt, *args, **kwargs)
                        updates.append(any(not torch.equal(a, p) for a, p in zip(before, params)))
                        return result

                    with mock.patch.object(module, factory_name, make_model), \
                         mock.patch.object(opt_cls, "step", step):
                        self.assertEqual(module.main(["--config", str(config), "--out", str(out)]), 0)
                    self.assertTrue(updates)
                    self.assertTrue(all(updates))
                    self.assertTrue(batch_sizes)
                    self.assertEqual(set(batch_sizes), {2}, "incomplete training batches must be dropped")
                    frozen = {n for n, p in models[0].named_parameters() if not p.requires_grad}
                    self.assertTrue(frozen)
                    buffers = set(dict(models[0].named_buffers()))
                    for name in frozen | buffers:
                        torch.testing.assert_close(models[0].state_dict()[name], snapshots[0][name], rtol=0, atol=0)
                    result = json.loads((out / "results.json").read_text())
                    self.assertFalse(result["canonical_eligible"])
                    self.assertFalse(result["record_value"])
                    report = result["optimization"]
                    self.assertEqual(report["profile"], PROFILE)
                    self.assertEqual(report["optimizer"], opt_cls.__name__)
                    self.assertEqual(report["effective_batch"], 2)
                    self.assertEqual(report["world_size"], 1)
                    self.assertEqual(report["accumulation_steps"], 1)
                    self.assertEqual(report["schedule"], "none")
                    self.assertTrue(report["drop_last"])
                    self.assertAlmostEqual(report["lr"], expected_lr)
                    self.assertAlmostEqual(report["weight_decay"], expected_wd)
                    self.assertTrue(contract.verify(out, config, 0)[0])

    def test_no_full_training_batch_fails_instead_of_reporting_success(self):
        for module, fixture, factory in self.cases():
            with self.subTest(task=module.TASK), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "data"
                fixture(root)
                if module is nyuv2:
                    savemat(root / "labeled/splits.mat", {"trainNdxs": [[1], [2], [3]],
                                                       "testNdxs": [[4], [5], [6]]})
                cfg = self.config(module, factory, root)
                cfg["detector" if module is coco else "probe"]["batch_size"] = 100
                config, out = Path(d) / "cfg.json", Path(d) / "out"
                config.write_text(json.dumps(cfg))
                self.assertEqual(module.main(["--config", str(config), "--out", str(out)]), 1)
                manifest = json.loads((out / contract.MANIFEST).read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertIn("full training batch", manifest["error"])
                self.assertFalse((out / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
