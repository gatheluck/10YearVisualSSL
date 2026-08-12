#!/usr/bin/env python3
"""Specification for 22_mocov3 (Chen et al., 2021; arXiv:2104.02057).

MoCo v3, the **ViT** path. Two augmented views feed a **base** encoder (ViT +
3-layer MLP projector + a 2-layer predictor) and a **momentum** encoder (an EMA
copy of the base, no gradient); a symmetric **InfoNCE** loss contrasts the
predicted query against the momentum key. Unlike the ResNet ports, MoCo v3's
step 1 is genuinely ViT-based -- `timm` provides the VisionTransformer -- but the
ViT is built from scratch (no pretrained download), so the run stays hermetic.

`encoder.pt` is the base ViT **trunk** (`base_encoder.*` minus the projector
`base_encoder.head.*`); the projector, predictor and momentum encoder are
training machinery and are excluded. `linear_eval` probes the ViT's CLS feature
(embed_dim). The captured step 2 (also ViT) is excluded, as in every port.
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
METHOD = ROOT / "methods" / "22_mocov3"
BIN = ROOT / "bin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import numpy                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import timm                                        # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "22_mocov3 needs torch, numpy, torchvision, timm")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("mocov3_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: vit_small at a 32px input (patch16 -> a
# 2x2 token grid), narrow projector/predictor. The paper's vit_base / 224px /
# proj 256 / mlp 4096 / 300 epochs live in the shipped config.
MODEL = {"arch": "vit_small", "proj_dim": 32, "mlp_dim": 64,
         "stop_grad_conv1": True, "img_size": 32}
MOCOV3 = {"temperature": 0.2, "momentum": 0.99, "momentum_cosine": True}
DATA = {"crop_min": 0.08}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0,
              "learning_rate": 1.0e-3, "min_lr": 0.0, "weight_decay": 0.1,
              "warmup_epochs": 0, "betas": [0.9, 0.95]}
TRAIN = {**MODEL, **MOCOV3, **DATA, **STEP1_ONLY}
EVAL_TRAIN = {"arch": "vit_small", "img_size": 32, "epochs": 2, "batch_size": 2,
              "num_workers": 0, "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

EMBED_DIM = 384  # vit_small CLS feature


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "train" / "class0"
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
        self.tmp = Path(tempfile.mkdtemp(prefix="mocov3-"))
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
        return load("mocov3_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.build_mocov3_vit(
            arch=MODEL["arch"], proj_dim=MODEL["proj_dim"],
            mlp_dim=MODEL["mlp_dim"], temperature=MOCOV3["temperature"],
            momentum=MOCOV3["momentum"],
            stop_grad_conv1=MODEL["stop_grad_conv1"], img_size=MODEL["img_size"])

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, MODEL["img_size"], MODEL["img_size"])

    @needs_deps
    def test_forward_returns_a_finite_scalar_loss(self):
        import torch
        model = self._model(self.models())
        model.train()
        loss = model(self._batch(torch), self._batch(torch))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @needs_deps
    def test_the_momentum_encoder_is_frozen(self):
        model = self._model(self.models())
        self.assertTrue(all(not p.requires_grad
                            for p in model.momentum_encoder.parameters()))

    @needs_deps
    def test_the_momentum_update_moves_the_key_toward_the_query(self):
        import torch
        model = self._model(self.models())
        with torch.no_grad():
            for p in model.base_encoder.parameters():
                p.add_(1.0)
        q = next(iter(model.base_encoder.parameters())).clone()
        before = next(iter(model.momentum_encoder.parameters())).clone()
        model._momentum_update(0.9)
        after = next(iter(model.momentum_encoder.parameters()))
        self.assertFalse(torch.allclose(before, after), "the key did not move")
        self.assertLess((after - q).abs().sum().item(),
                        (before - q).abs().sum().item(),
                        "the momentum update did not move the key toward query")

    @needs_deps
    def test_get_backbone_returns_the_cls_feature(self):
        import torch
        model = self._model(self.models())
        feats = model.get_backbone()(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))


class TestTheEmaSchedule(unittest.TestCase):
    def trainer(self):
        return load("mocov3_trainer", METHOD / "train_step1_mocov3.py")

    @needs_deps
    def test_momentum_rises_toward_one(self):
        t = self.trainer()
        start = t.adjust_moco_momentum(0.0, 100, base_momentum=0.99)
        end = t.adjust_moco_momentum(100.0, 100, base_momentum=0.99)
        self.assertAlmostEqual(start, 0.99, places=5)
        self.assertAlmostEqual(end, 1.0, places=5)
        self.assertLess(start, end)


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("mocov3_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_two_views_and_a_label(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().MoCoV3Dataset(
            str(self.tmp / "data"), img_size=MODEL["img_size"],
            crop_min=DATA["crop_min"])
        item = ds[0]
        self.assertEqual(len(item), 3)
        v1, v2, label = item
        self.assertEqual(tuple(v1.shape),
                         (3, MODEL["img_size"], MODEL["img_size"]))
        self.assertEqual(tuple(v2.shape),
                         (3, MODEL["img_size"], MODEL["img_size"]))
        self.assertFalse(torch.equal(v1, v2),
                         "the two views are identical, not independently augmented")


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_base_trunk_comes_out(self):
        got = adapter.extract_encoder({
            "base_encoder.cls_token": 1, "base_encoder.blocks.0.norm1.weight": 2,
            "base_encoder.head.0.weight": 3, "momentum_encoder.cls_token": 4,
            "predictor.0.weight": 5})
        self.assertEqual(set(got),
                         {"base_encoder.cls_token",
                          "base_encoder.blocks.0.norm1.weight"})

    def test_the_projector_predictor_and_momentum_are_left_out(self):
        got = adapter.extract_encoder({"base_encoder.pos_embed": 1,
                                       "base_encoder.head.2.weight": 2,
                                       "predictor.0.weight": 3,
                                       "momentum_encoder.head.0.weight": 4})
        self.assertEqual(set(got), {"base_encoder.pos_embed"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"predictor.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["arch"], "vit_small")
        self.assertEqual(built["mocov3"]["temperature"], 0.2)
        self.assertEqual(built["training"]["epochs"], 1)

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
        cfg = self.eval_config(train={"temperature": 0.2})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("temperature", str(e.exception))


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
        return load("mocov3_trainer", METHOD / "train_step1_mocov3.py")

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
        src = (METHOD / "train_step1_mocov3.py").read_text()
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
        tree = ast.parse((METHOD / "train_step1_mocov3.py").read_text())
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
