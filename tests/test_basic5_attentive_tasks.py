"""Exercise attentive components through downstream task entry points."""
import json
import shlex
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests._checkout import needs_checkout

try:
    import torch
    from torch import nn
    from scipy.io import savemat
    from downstream import ade20k, nyuv2, coco, ssv2, contract
    from downstream.spatial_backbones import build_frozen_backbone
    from tests.test_basic5_readers import SPEC
    from tests.test_downstream_ade20k import tiny_ade, smoke_config as ade_config
    from tests.test_downstream_nyuv2 import tiny_nyuv2, smoke_config as nyu_config
    from tests.test_downstream_coco import tiny_coco, smoke_config as coco_config
    from tests.test_downstream_ssv2 import tiny_ssv2, smoke_config as video_config
    HAVE = True
except ImportError:
    HAVE = False

PROFILE = "capture_basic5_components"


class TestAttentiveCI(unittest.TestCase):
    @needs_checkout
    def test_downstream_lock_runs_attentive_task_contracts(self):
        from tests.test_ci import HAVE_YAML, parsed
        if not HAVE_YAML:
            self.skipTest("PyYAML is required")
        workflow = parsed()["tests.yml"]
        commands = [shlex.split(step["run"].replace("\\\n", " "))
                    for job in workflow["jobs"].values()
                    for step in job.get("steps", [])
                    if step.get("name") == "Run Basic5 component contracts with downstream dependencies"]
        self.assertEqual(len(commands), 1)
        self.assertIn("tests.test_basic5_attentive_tasks", commands[0])


@unittest.skipUnless(HAVE, "Attentive task tests require downstream dependencies")
class TestAttentiveTasks(unittest.TestCase):
    def test_config_requires_explicit_component_profile_and_known_adaptation(self):
        for module, factory in ((ade20k, ade_config), (nyuv2, nyu_config),
                                (coco, coco_config), (ssv2, video_config)):
            for adaptation in ("frozen", "attentive"):
                cfg = factory(Path("/missing"))
                cfg.update(profile=PROFILE, adaptation=adaptation)
                with self.subTest(task=module.__name__, adaptation=adaptation):
                    module.validate_config(cfg)
                for invalid in ({"profile": "legacy", "adaptation": "attentive"},
                                {"adaptation": "typo"}):
                    with self.assertRaises(module.ConfigError):
                        module.validate_config({**cfg, **invalid})

    def models(self, device):
        for kind in ("seg", "depth", "det", "video"):
            backbone = build_frozen_backbone(SPEC, device)
            if kind == "seg":
                model = ade20k.FrozenSegModel(backbone, 3, adaptation="attentive")
            elif kind == "depth":
                model = nyuv2.FrozenDepthModel(backbone, profile=PROFILE,
                                               adaptation="attentive")
            elif kind == "det":
                model = coco.FrozenPyramidBackbone(backbone, adaptation="attentive")
            else:
                model = ssv2.FrozenFrameAverageClassifier(backbone, 3,
                                                         adaptation="attentive")
            yield kind, backbone, model.to(device)

    def check_updates(self, device):
        for kind, backbone, model in self.models(device):
            with self.subTest(kind=kind, device=str(device)):
                model.train()
                encoder = {k: v.clone() for k, v in backbone.state_dict().items()}
                reader = model.reader if kind == "video" else model.adapter
                before = {k: v.clone() for k, v in reader.state_dict().items()}
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], lr=.001)
                x = torch.randn((2, 3, 3, 32, 32) if kind == "video"
                                else (2, 3, 32, 32), device=device, requires_grad=True)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x)
                    loss = (sum(v.square().mean() for v in output.values())
                            if isinstance(output, dict) else (output - 1).square().mean())
                    loss.backward()
                    self.assertIsNone(x.grad)
                    self.assertTrue(all(p.grad is None for p in backbone.parameters()))
                    optimizer.step()
                self.assertFalse(torch.equal(before["input_projection.weight"],
                                             reader.input_projection.weight))
                for k, value in backbone.state_dict().items():
                    torch.testing.assert_close(value, encoder[k], rtol=0, atol=0)
                with torch.no_grad():
                    model.eval()(x)

    def test_two_steps_update_readers_but_never_encoders(self):
        self.check_updates(torch.device("cpu"))

    @unittest.skipUnless(HAVE and torch.cuda.is_available(), "CUDA is required")
    def test_cuda_two_step_updates(self):
        self.check_updates(torch.device("cuda"))

    def test_video_reader_receives_each_frame_without_normalization_or_temporal_pool(self):
        backbone = build_frozen_backbone(SPEC, torch.device("cpu"))
        model = ssv2.FrozenFrameAverageClassifier(backbone, 3, adaptation="attentive")
        x = torch.randn(2, 4, 3, 32, 32)
        expected = backbone.forward_features(x.reshape(8, 3, 32, 32)).mean((-2, -1))
        seen = []
        handle = model.reader.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
        model(x)
        handle.remove()
        torch.testing.assert_close(seen[0], expected.reshape(2, 4, -1), rtol=0, atol=0)
        self.assertEqual(model.classifier.in_features, 512)
        self.assertEqual(torch.count_nonzero(model.classifier.weight), 0)
        self.assertEqual(torch.count_nonzero(model.classifier.bias), 0)

    def test_legacy_models_preserve_state_and_random_initialization(self):
        for cls in (ade20k.FrozenSegModel, nyuv2.FrozenDepthModel,
                    coco.FrozenPyramidBackbone, ssv2.FrozenFrameAverageClassifier):
            backbone = build_frozen_backbone(SPEC, torch.device("cpu"))
            torch.manual_seed(45)
            old = cls(backbone)
            rng = torch.get_rng_state()
            torch.manual_seed(45)
            explicit = cls(backbone, adaptation="frozen")
            self.assertEqual(set(old.state_dict()), set(explicit.state_dict()))
            for key, val in old.state_dict().items():
                torch.testing.assert_close(val, explicit.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_real_clis_update_attention_clip_gradients_and_refuse_table_recording(self):
        self.check_clis("cpu")

    @unittest.skipUnless(HAVE and torch.cuda.is_available(), "CUDA is required")
    def test_cuda_real_clis(self):
        self.check_clis("cuda")

    def check_clis(self, device):
        cases = ((ade20k, tiny_ade, ade_config), (nyuv2, tiny_nyuv2, nyu_config),
                 (coco, tiny_coco, coco_config), (ssv2, tiny_ssv2, video_config))
        for module, fixture, factory in cases:
            with self.subTest(task=module.__name__), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "data"
                fixture(root)
                if module is nyuv2:
                    savemat(root / "labeled" / "splits.mat", {"trainNdxs": [[1], [2], [3]],
                                                 "testNdxs": [[4], [5], [6]]})
                cfg = factory(root)
                cfg.update(profile=PROFILE, adaptation="attentive", device=device)
                if module is coco:
                    cfg["detector"]["anchor_sizes"] = [8, 16, 32, 64]
                train_cfg = cfg["detector" if module is coco else "probe"]
                train_cfg.update(max_train_samples=0, max_val_samples=0, max_steps_per_epoch=0)
                config, out = Path(d) / "cfg.json", Path(d) / "out"
                config.write_text(json.dumps(cfg))
                updates, clipped = [], []
                models = []
                factory_name = {ade20k: "FrozenSegModel", nyuv2: "FrozenDepthModel",
                                coco: "build_frozen_detector",
                                ssv2: "FrozenFrameAverageClassifier"}[module]
                model_factory = getattr(module, factory_name)
                def make_model(*args, **kwargs):
                    model = model_factory(*args, **kwargs)
                    models.append(model)
                    return model
                # Observe the real optimizer and real gradients, without replacing training.
                optimizer_class = torch.optim.SGD if module is coco else torch.optim.AdamW
                original_step = optimizer_class.step
                original_clip = torch.nn.utils.clip_grad_norm_
                def step(opt, *args, **kwargs):
                    params = [p for g in opt.param_groups for p in g["params"]]
                    # Attention's 256/512 bottleneck projection is absent in the baseline heads.
                    readers = [p for p in params if p.ndim >= 2 and p.shape[0] in (256, 512)
                               and p.shape[1] == cfg["backbone"]["embed_dim"]]
                    self.assertTrue(readers, "optimizer must include reader input projection")
                    self.assertTrue(all(p.requires_grad for p in params))
                    self.assertEqual({id(p) for p in params},
                                     {id(p) for p in models[0].parameters() if p.requires_grad})
                    before = [p.detach().clone() for p in params]
                    result = original_step(opt, *args, **kwargs)
                    updates.append(any(not torch.equal(a, p) for a, p in zip(before, params)))
                    return result
                def clip(params, max_norm, *args, **kwargs):
                    params = list(params)
                    self.assertEqual(max_norm, 1.0)
                    self.assertEqual({id(p) for p in params},
                                     {id(p) for p in models[0].parameters() if p.requires_grad})
                    before = [(p, p.grad.clone()) for p in params if p.grad is not None]
                    result = original_clip(params, max_norm, *args, **kwargs)
                    self.assertTrue(torch.isfinite(result))
                    # Verify every gradient against the actual clipping rule. A
                    # second float32 reduction can differ from the foreach norm.
                    coefficient = (max_norm / (result + 1e-6)).clamp(max=1)
                    for param, grad in before:
                        torch.testing.assert_close(param.grad, grad * coefficient, rtol=0, atol=0)
                    clipped.append(float(result) > max_norm)
                    return result
                with mock.patch.object(optimizer_class, "step", step), \
                     mock.patch.object(module, factory_name, make_model), \
                     mock.patch.object(torch.nn.utils, "clip_grad_norm_", clip):
                    rc = module.main(["--config", str(config), "--out", str(out)])
                    self.assertEqual(rc, 0, (out / contract.MANIFEST).read_text())
                self.assertTrue(updates)
                self.assertTrue(all(updates))
                self.assertEqual(len(clipped), len(updates))
                self.assertTrue(any(clipped), "exercise actual clipping, not only the no-op case")
                self.assertTrue(contract.verify(out, config, 0)[0])
                result = json.loads((out / "results.json").read_text())
                self.assertEqual(result["adaptation"], "attentive")
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])
                self.assertFalse(result["subset_or_smoke"])


if __name__ == "__main__":
    unittest.main()
