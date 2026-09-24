"""Full-gradient component paths, not certification of complete FT recipes."""
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock

from tests._checkout import needs_checkout

try:
    import torch
    from torch import nn
    from scipy.io import savemat
    from downstream import ade20k, nyuv2, coco, ssv2, contract
    from downstream.spatial_backbones import build_trainable_backbone
    from tests.test_basic5_readers import SPEC
    from tests.test_downstream_ade20k import tiny_ade, smoke_config as ade_config
    from tests.test_downstream_nyuv2 import tiny_nyuv2, smoke_config as nyu_config
    from tests.test_downstream_coco import tiny_coco, smoke_config as coco_config
    from tests.test_downstream_ssv2 import tiny_ssv2, smoke_config as video_config
    HAVE = True
except ImportError:
    HAVE = False

PROFILE = "capture_basic5_components"


def _runs_finetune_tests(command, *, module="tests.test_basic5_finetune_tasks"):
    for line in command.replace("\\\n", " ").splitlines():
        words = shlex.split(line, comments=True)
        if words[:1] == ["PYTHONPATH=."]:
            words = words[1:]
        if (words[:3] == [".venv/bin/python", "-m", "unittest"]
                and module in words[3:]):
            return True
    return False


class TestFinetuneCI(unittest.TestCase):
    def test_invocation_detector_rejects_exact_name_in_comments_and_echo(self):
        invocation = "PYTHONPATH=. .venv/bin/python -m unittest -v tests.test_basic5_finetune_tasks"
        self.assertTrue(_runs_finetune_tests(invocation))
        for decoy in ("# " + invocation, "echo " + shlex.quote(invocation), "echo " + invocation,
                      "PYTHONPATH=. .venv/bin/python -m unittest tests.test_ci # tests.test_basic5_finetune_tasks",
                      ".venv/bin/python -m unittest tests.test_basic5_finetune_tasks_disabled",
                      "echo tests.test_basic5_finetune_tasks"):
            with self.subTest(decoy=decoy):
                self.assertFalse(_runs_finetune_tests(decoy))

    @needs_checkout
    def test_downstream_job_executes_finetune_tests(self):
        from tests.test_ci import HAVE_YAML, parsed
        if not HAVE_YAML:
            self.skipTest("PyYAML required")
        commands = [s["run"]
                    for j in parsed()["tests.yml"]["jobs"].values()
                    for s in j.get("steps", [])
                    if s.get("name") == "Run Basic5 component contracts with downstream dependencies"]
        self.assertEqual(len(commands), 1)
        self.assertTrue(_runs_finetune_tests(commands[0]))


@unittest.skipUnless(HAVE, "Full downstream dependencies required")
class TestFinetuneTasks(unittest.TestCase):
    def test_configs_accept_explicit_ft_and_refuse_legacy_and_frozen_only_providers(self):
        for module, factory in ((ade20k, ade_config), (nyuv2, nyu_config),
                                (coco, coco_config), (ssv2, video_config)):
            cfg = factory(Path("/unused"))
            cfg.update(profile=PROFILE, adaptation="finetune")
            with self.subTest(task=module.__name__):
                module.validate_config(cfg)
                providers = sorted(set(module.KINDS) - {"vit"})
                self.assertTrue(providers, "exercise a recognized frozen-only provider")
                frozen_spec = {**cfg["backbone"], "kind": providers[0]}
                module.validate_config({**cfg, "adaptation": "frozen", "backbone": frozen_spec})
                for bad in ({**cfg, "profile": "legacy"},
                            {**cfg, "adaptation": "typo"},
                            {**cfg, "backbone": frozen_spec}):
                    with self.assertRaises(module.ConfigError):
                        module.validate_config(bad)

    def models(self, device):
        for kind in ("seg", "depth", "det", "video"):
            backbone = build_trainable_backbone(SPEC, device)
            cls = {"seg": ade20k.FrozenSegModel, "depth": nyuv2.FrozenDepthModel,
                   "det": coco.FrozenPyramidBackbone,
                   "video": ssv2.FrozenFrameAverageClassifier}[kind]
            kwargs = {"profile": PROFILE} if kind == "depth" else {}
            yield kind, backbone, cls(backbone, adaptation="finetune", **kwargs).to(device)

    def check_updates(self, device):
        torch.set_num_threads(1)
        for kind, backbone, model in self.models(device):
            with self.subTest(kind=kind, device=str(device)):
                model.train()
                self.assertTrue(backbone.training)
                self.assertTrue(all(p.requires_grad for p in backbone.parameters()))
                self.assertIsNone(getattr(model, "adapter", None))
                self.assertIsNone(getattr(model, "reader", None))
                before = {n: p.detach().clone() for n, p in backbone.named_parameters()}
                head = model.classifier if kind == "video" else (model.fpn if kind == "det" else model.head)
                head_before = [p.detach().clone() for p in head.parameters()]
                x = torch.randn((2, 3, 3, 32, 32) if kind == "video" else (2, 3, 32, 32),
                                device=device, requires_grad=True)
                opt = torch.optim.SGD(model.parameters(), lr=.01)
                for _ in range(2):
                    opt.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = (sum(t.square().mean() for t in out.values()) if isinstance(out, dict)
                            else (out - 1).square().mean())
                    loss.backward()
                    opt.step()
                self.assertIsNotNone(x.grad)
                self.assertGreater(float(x.grad.abs().sum()), 0)
                self.assertTrue(any(not torch.equal(before[n], p) for n, p in backbone.named_parameters()))
                self.assertTrue(any(not torch.equal(b, p) for b, p in zip(head_before, head.parameters())))
                model.eval()
                self.assertFalse(backbone.training)
                feature_grad = []
                forward_features = backbone.forward_features
                def observe_features(*args):
                    value = forward_features(*args)
                    feature_grad.append(value.requires_grad)
                    return value
                with torch.no_grad(), mock.patch.object(backbone, "forward_features", observe_features):
                    out = model(x)
                self.assertEqual(feature_grad, [False])
                for value in (out.values() if isinstance(out, dict) else [out]):
                    self.assertFalse(value.requires_grad)
                # Save/reload must include the updated encoder, not only the head.
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / "state.pt"
                    torch.save(model.state_dict(), path)
                    state = torch.load(path, weights_only=True, map_location=device)
                    with torch.no_grad():
                        for p in model.parameters():
                            p.zero_()
                    model.load_state_dict(state, strict=True)
                    for n, p in model.state_dict().items():
                        torch.testing.assert_close(p, state[n], rtol=0, atol=0)
                    with torch.no_grad():
                        restored = model(x)
                    values = (out.values() if isinstance(out, dict) else [out])
                    restored_values = (restored.values() if isinstance(restored, dict) else [restored])
                    for before_value, after_value in zip(values, restored_values):
                        torch.testing.assert_close(before_value, after_value, rtol=0, atol=0)

    def test_cpu_encoder_and_head_update_without_attention(self):
        self.check_updates(torch.device("cpu"))

    @unittest.skipUnless(HAVE and torch.cuda.is_available(), "CUDA required")
    def test_cuda_encoder_and_head_update_without_attention(self):
        self.check_updates(torch.device("cuda"))

    def test_video_uses_per_frame_l2_before_temporal_mean_and_zero_head(self):
        backbone = build_trainable_backbone(SPEC, torch.device("cpu"))
        model = ssv2.FrozenFrameAverageClassifier(backbone, 3, adaptation="finetune")
        x = torch.randn(2, 4, 3, 32, 32)
        expected = torch.nn.functional.normalize(
            backbone.forward_features(x.reshape(8, 3, 32, 32)).mean((-2, -1)).float(), dim=-1
        ).reshape(2, 4, -1).mean(1)
        seen = []
        handle = model.classifier.register_forward_pre_hook(lambda m, a: seen.append(a[0]))
        model(x)
        handle.remove()
        torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
        self.assertTrue(seen[0].requires_grad)
        self.assertEqual(torch.count_nonzero(model.classifier.weight), 0)

    def check_clis(self, device):
        for module, fixture, factory, model_name in (
            (ade20k, tiny_ade, ade_config, "FrozenSegModel"),
            (nyuv2, tiny_nyuv2, nyu_config, "FrozenDepthModel"),
            (coco, tiny_coco, coco_config, "build_frozen_detector"),
            (ssv2, tiny_ssv2, video_config, "FrozenFrameAverageClassifier"),
        ):
            with self.subTest(task=module.__name__), tempfile.TemporaryDirectory() as d:
                root = Path(d) / "data"
                fixture(root)
                if module is nyuv2:
                    savemat(root / "labeled/splits.mat", {"trainNdxs": [[1], [2], [3]],
                                                        "testNdxs": [[4], [5], [6]]})
                cfg = factory(root)
                cfg.update(profile=PROFILE, adaptation="finetune", device=device)
                train = cfg["detector" if module is coco else "probe"]
                train.update(max_train_samples=0, max_val_samples=0, max_steps_per_epoch=0)
                if module is coco:
                    train["anchor_sizes"] = [8, 16, 32, 64]
                config, out = Path(d) / "cfg.json", Path(d) / "out"
                config.write_text(json.dumps(cfg))
                captured, updated, clipped = [], [], []
                original_factory = getattr(module, model_name)
                def make(*args, **kwargs):
                    model = original_factory(*args, **kwargs)
                    encoder = model.backbone.body if module is coco else model.backbone
                    captured.append((model, encoder))
                    return model
                optimizer_cls = torch.optim.SGD if module is coco else torch.optim.AdamW
                original_step, original_clip = optimizer_cls.step, torch.nn.utils.clip_grad_norm_
                def step(opt, *args, **kwargs):
                    model, encoder = captured[0]
                    params = [p for g in opt.param_groups for p in g["params"]]
                    self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
                    self.assertTrue(encoder.training)
                    self.assertTrue(all(p.requires_grad for p in encoder.parameters()))
                    before = [p.detach().clone() for p in encoder.parameters()]
                    result = original_step(opt, *args, **kwargs)
                    updated.append(any(not torch.equal(a, p) for a, p in zip(before, encoder.parameters())))
                    return result
                def clip(params, max_norm, *args, **kwargs):
                    params = list(params)
                    self.assertEqual(max_norm, 1.0)
                    self.assertEqual({id(p) for p in params}, {id(p) for p in captured[0][0].parameters()})
                    value = original_clip(params, max_norm, *args, **kwargs)
                    self.assertTrue(torch.isfinite(value))
                    clipped.append(True)
                    return value
                with mock.patch.object(module, model_name, make), \
                     mock.patch.object(optimizer_cls, "step", step), \
                     mock.patch.object(torch.nn.utils, "clip_grad_norm_", clip):
                    rc = module.main(["--config", str(config), "--out", str(out)])
                    self.assertEqual(rc, 0, (out / contract.MANIFEST).read_text())
                self.assertTrue(any(updated), "actual CLI must update encoder tensors")
                self.assertEqual(len(clipped), len(updated))
                self.assertTrue(contract.verify(out, config, 0)[0])
                result = json.loads((out / "results.json").read_text())
                self.assertEqual(result["adaptation"], "finetune")
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])
                self.assertFalse(result["subset_or_smoke"])

    def test_cpu_actual_clis(self):
        self.check_clis("cpu")

    @unittest.skipUnless(HAVE and torch.cuda.is_available(), "CUDA required")
    def test_cuda_actual_clis(self):
        self.check_clis("cuda")
