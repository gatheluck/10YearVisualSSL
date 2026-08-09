#!/usr/bin/env python3
"""Specification for 19_byol (Grill et al., 2020; arXiv:2006.07733).

BYOL (Bootstrap Your Own Latent), the ResNet-50 path. An **online** network
(ResNet-50 backbone -> projector -> predictor) is trained so its prediction of
one view matches a **target** network's projection of the other view; the target
is an exponential-moving-average (EMA) copy of the online backbone + projector,
with no predictor and no gradient. The loss is a symmetric negative cosine
similarity -- **no negatives, no queue**. The EMA momentum tau follows a cosine
schedule from 0.996 to 1.0.

`encoder.pt` is the online ResNet-50 backbone (`online_encoder.*`); the projector,
predictor and target network are training machinery and are excluded.
`linear_eval` probes the backbone (2048-d). The captured step 2 (ViT, which needs
`timm`) is excluded, as in every port.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _method_import import load_from        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHOD = ROOT / "methods" / "19_byol"
BIN = ROOT / "bin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import numpy                                       # noqa: F401
    import torchvision                                 # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "19_byol needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("byol_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: narrow projector/predictor and a 32px
# input (ResNet-50 downsamples to 1x1, still valid). The paper's 4096/256 MLPs,
# 1000 epochs and LARS lr 0.2 live in the shipped config.
MODEL = {"arch": "resnet50", "encoder_dim": 2048, "proj_hidden_dim": 256,
         "proj_output_dim": 64, "pred_hidden_dim": 256, "pred_output_dim": 64,
         "image_size": 32}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0,
              "learning_rate": 0.2, "lr_scale_by_batch": False,
              "lr_scale_base": 256, "momentum": 0.9, "weight_decay": 1.5e-6,
              "trust_coefficient": 0.001, "warmup_epochs": 0, "min_lr": 0.0,
              "ema_tau_base": 0.996, "ema_tau_final": 1.0}
TRAIN = {**MODEL, **STEP1_ONLY}
EVAL_TRAIN = {"arch": "resnet50", "image_size": 32, "epochs": 2,
              "batch_size": 2, "num_workers": 0, "lr": 0.1, "momentum": 0.9,
              "weight_decay": 0.0}

PROJ_OUT = 64
FEATURE_DIM = 2048


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 256, (48, 48, 3), dtype="uint8")).save(
            cls / f"{i}.png")
    return root


def tiny_split(root: Path, per: int = 3) -> Path:
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        for label, cls in enumerate(("c0", "c1")):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per):
                base = np.full((48, 48, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (48, 48, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="byol-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "step1", "seed": 0, "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg

    def eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "encoder": str(self.tmp / "encoder.pt"),
               "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("byol_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.BYOLResNet50(
            encoder_dim=MODEL["encoder_dim"],
            proj_hidden_dim=MODEL["proj_hidden_dim"],
            proj_output_dim=MODEL["proj_output_dim"],
            pred_hidden_dim=MODEL["pred_hidden_dim"],
            pred_output_dim=MODEL["pred_output_dim"])

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, MODEL["image_size"], MODEL["image_size"])

    @needs_deps
    def test_forward_returns_online_predictions_and_target_projections(self):
        import torch
        model = self._model(self.models())
        model.train()
        p1, p2, z1, z2 = model(self._batch(torch), self._batch(torch))
        for t in (p1, p2, z1, z2):
            self.assertEqual(tuple(t.shape), (2, PROJ_OUT))

    @needs_deps
    def test_the_target_network_is_frozen(self):
        model = self._model(self.models())
        self.assertTrue(all(not p.requires_grad
                            for p in model.target_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad
                            for p in model.target_projector.parameters()))

    @needs_deps
    def test_the_ema_update_moves_the_target_toward_the_online_net(self):
        import torch
        model = self._model(self.models())
        with torch.no_grad():
            for p in model.online_encoder.parameters():
                p.add_(1.0)                      # move the online net away
        online = next(iter(model.online_encoder.parameters())).clone()
        before = next(iter(model.target_encoder.parameters())).clone()
        model.update_target_network(tau=0.9)
        after = next(iter(model.target_encoder.parameters()))
        self.assertFalse(torch.allclose(before, after), "the target did not move")
        # It must move *toward* the online net (EMA), not merely change: a decay
        # that dropped the online term would move it, but away, not closer.
        self.assertLess((after - online).abs().sum().item(),
                        (before - online).abs().sum().item(),
                        "the EMA update did not move the target toward online")

    @needs_deps
    def test_encode_returns_the_backbone_feature(self):
        import torch
        model = self._model(self.models())
        feats = model.encode(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))


class TestTheLoss(unittest.TestCase):
    def models(self):
        return load("byol_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_identical_prediction_and_target_gives_minus_one(self):
        import torch
        crit = self.models().BYOLLoss()
        v = torch.randn(4, PROJ_OUT)
        loss = crit(v, v, v, v)               # p==z on both views
        self.assertAlmostEqual(loss.item(), -1.0, places=5)

    @needs_deps
    def test_the_target_is_detached(self):
        import torch
        crit = self.models().BYOLLoss()
        p1 = torch.randn(4, PROJ_OUT, requires_grad=True)
        p2 = torch.randn(4, PROJ_OUT, requires_grad=True)
        z1 = torch.randn(4, PROJ_OUT, requires_grad=True)
        z2 = torch.randn(4, PROJ_OUT, requires_grad=True)
        crit(p1, p2, z1, z2).backward()
        self.assertIsNone(z1.grad, "gradient flowed into the target projection")
        self.assertIsNone(z2.grad)
        self.assertIsNotNone(p1.grad)


class TestTheEmaSchedule(unittest.TestCase):
    def models(self):
        return load("byol_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_tau_rises_from_base_toward_final(self):
        m = self.models()
        start = m.compute_ema_tau(0, 100, tau_base=0.996, tau_final=1.0)
        end = m.compute_ema_tau(100, 100, tau_base=0.996, tau_final=1.0)
        self.assertAlmostEqual(start, 0.996, places=5)
        self.assertAlmostEqual(end, 1.0, places=5)
        self.assertLess(start, end)


class TestTheOptimizer(unittest.TestCase):
    def models(self):
        return load("byol_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_a_step_updates_the_parameters(self):
        import torch
        p = torch.nn.Parameter(torch.ones(4, 4))
        opt = self.models().LARS([p], lr=0.1, momentum=0.9, weight_decay=1.5e-6,
                                 trust_coefficient=0.001)
        (p.sum()).backward()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.allclose(before, p.detach()),
                         "LARS.step did not move the parameter")


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("byol_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_two_views_are_produced_and_differ(self):
        import torch
        from PIL import Image
        import numpy as np
        dm = self.dataset_mod()
        tf = dm.BYOLTwoViewTransform(img_size=MODEL["image_size"],
                                     augmentation="byol")
        img = Image.fromarray(
            np.random.RandomState(0).randint(0, 256, (48, 48, 3), dtype="uint8"))
        v1, v2 = tf(img)
        self.assertEqual(tuple(v1.shape),
                         (3, MODEL["image_size"], MODEL["image_size"]))
        self.assertEqual(tuple(v2.shape),
                         (3, MODEL["image_size"], MODEL["image_size"]))
        self.assertFalse(torch.equal(v1, v2),
                         "the two views are identical, not independently augmented")


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_online_backbone_comes_out(self):
        got = adapter.extract_encoder({
            "online_encoder.0.weight": 1, "online_encoder.4.0.conv1.weight": 2,
            "online_projector.layer1.0.weight": 3, "predictor.layer2.weight": 4,
            "target_encoder.0.weight": 5})
        self.assertEqual(set(got),
                         {"online_encoder.0.weight",
                          "online_encoder.4.0.conv1.weight"})

    def test_the_projector_predictor_and_target_are_left_out(self):
        got = adapter.extract_encoder({"online_encoder.1.weight": 1,
                                       "online_projector.layer2.bias": 2,
                                       "predictor.layer2.bias": 3,
                                       "target_projector.layer2.bias": 4})
        self.assertEqual(set(got), {"online_encoder.1.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"predictor.layer2.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["proj_output_dim"], 64)
        self.assertEqual(built["training"]["epochs"], 1)
        self.assertEqual(built["training"]["ema_tau_base"], 0.996)

    def test_a_missing_step1_setting_is_refused_by_name(self):
        for key in TRAIN:
            with self.subTest(key=key):
                cfg = self.config()
                cfg["train"] = {k: v for k, v in TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_step1_setting_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"nonsense": 1}),
                                  out=self.out)
        self.assertIn("nonsense", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(stage="step2"), out=self.out)

    def test_output_is_refused(self):
        cfg = self.config()
        cfg["output"] = {"save_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))


class TestLinearEvalConfig(Base):
    def test_linear_eval_is_accepted(self):
        adapter.to_run_config(self.eval_config(), out=self.out)

    def test_the_encoder_must_be_named(self):
        cfg = self.eval_config()
        del cfg["encoder"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("encoder", str(e.exception))

    def test_step1_only_settings_are_not_part_of_the_probe(self):
        cfg = self.eval_config(train={"ema_tau_base": 0.996})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("ema_tau_base", str(e.exception))


class TestTheEvalProducesNoEncoder(Base):
    def _reason(self, cfg):
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return adapter._absent_reason(p)

    def test_linear_eval_declares_no_encoder(self):
        self.assertTrue(self._reason(self.eval_config()))

    def test_step1_gives_no_reason(self):
        self.assertIsNone(self._reason(self.config()))


class TestTheMetricsAreInTheVocabulary(unittest.TestCase):
    def test_step1_maps_a_pretext_loss(self):
        self.assertEqual(adapter.STEP1_METRIC_NAMES["final_loss"],
                         "final_pretext_loss")
        for target in adapter.STEP1_METRIC_NAMES.values():
            if target is not None:
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_eval_maps_the_comparable_probe_numbers(self):
        mapped = set(adapter.LINEAR_EVAL_METRIC_NAMES.values())
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, mapped)
            self.assertEqual(adapterlib.METRIC_VOCABULARY[name],
                             adapterlib.COMPARABLE)


class TestTheDeviceIsResolved(Base):
    def trainer(self):
        return load("byol_trainer", METHOD / "train_step1_byol.py")

    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        t = self.trainer()
        with mock.patch.object(t.torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                t.resolve_device("cuda", 0)
            self.assertEqual(t.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(t.resolve_device("auto", 0).type, "cpu")

    def test_run_resolves_the_device(self):
        import ast
        src = (METHOD / "train_step1_byol.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


class TestAStep1Smoke(Base):
    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data")
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.config(**over)), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return cfg, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_writes_an_encoder_and_a_pretext_loss(self):
        self.run_adapter()
        self.assertTrue((self.out / "encoder.pt").is_file())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m)

    @needs_deps
    def test_the_encoder_pt_it_wrote_loads_back(self):
        self.run_adapter()
        import torch
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved)
        load("this_methods_models", METHOD / "models" / "__init__.py")
        model = adapter.load_encoder(saved, self.config())
        loaded = model.state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")

    @needs_deps
    def test_the_same_config_twice_gives_the_same_encoder(self):
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


class TestALinearEvalSmoke(Base):
    def _step1(self):
        tiny_split(self.tmp / "data")
        s1data = self.tmp / "s1data"
        tiny_imagefolder(s1data)
        s1cfg = {"stage": "step1", "seed": 0, "data_root": str(s1data),
                 "device": "cpu", "train": dict(TRAIN)}
        p = self.tmp / "s1.json"
        p.write_text(json.dumps(s1cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        s1out = self.tmp / "s1out"
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(s1out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return s1out

    def run_eval(self, **over):
        s1out = self._step1()
        cfg = self.eval_config(encoder=str(s1out / "encoder.pt"), **over)
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        return p, r

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_eval()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_reports_the_comparable_probe_numbers(self):
        self.run_eval()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy"):
            self.assertIn(name, m)

    @needs_deps
    def test_it_produces_no_encoder_and_says_so(self):
        self.run_eval()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)


class TestTheOriginalIsReferencedNotCopied(unittest.TestCase):
    def test_no_distributed_or_tensorboard_machinery_is_used(self):
        import ast
        tree = ast.parse((METHOD / "train_step1_byol.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)
        self.assertNotIn("autocast", used)


if __name__ == "__main__":
    unittest.main()
