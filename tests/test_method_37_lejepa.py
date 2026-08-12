#!/usr/bin/env python3
"""Specification for 37_lejepa (Balestriero & LeCun, 2025; arXiv:2511.08544).

LeJEPA: each image is seen as several augmented views; a timm ViT backbone + a
projection MLP maps each view to a projected feature, trained by SIGReg (an
Epps-Pulley Gaussian regularizer over random 1-D slices of the batch) plus a
cross-view invariance loss, combined as SIGReg * lambda + invariance * (1 - lambda).
The ViT is timm's (built from scratch, hermetic) and SIGReg is reimplemented
locally, so this ports self-contained -- no submodule, no downloaded weights.

`encoder.pt` is the bare backbone (`backbone.*`, the prefix stripped so it loads
into a plain timm model); the projection MLP is training machinery and is excluded.
`linear_eval` probes the backbone's num_features feature. The captured step 2 (ViT
fine-tune) is excluded, as in every port.
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
METHOD = ROOT / "methods" / "37_lejepa"
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
    HAVE_DEPS, "37_lejepa needs torch, numpy, torchvision, timm")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("lejepa_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny timm ViT at a 32px input (patch16 ->
# a 2x2 = 4 patch grid, feature_dim 192), 2 views, a tiny projector and a small
# SIGReg (few knots/slices). The paper's ViT-B/16 / 224px / 4 views / 100 epochs
# live in the shipped config.
NAME = "vit_tiny_patch16_224"
IMG = 32
FEATURE_DIM = 192   # vit_tiny num_features
VIEWS = 2
PROJ_DIM = 32

MODEL = {"name": NAME, "img_size": IMG, "drop_path_rate": 0.0,
         "proj_hidden_dim": 64, "proj_dim": PROJ_DIM, "proj_layers": 2,
         "final_bn": True}
AUG = {"views": VIEWS, "crop_scale": [0.5, 1.0],
       "color_jitter": [0.4, 0.4, 0.4, 0.1], "color_jitter_p": 0.5,
       "grayscale_p": 0.2, "blur_p": 0.0, "blur_kernel": 3, "solarize_p": 0.0,
       "hflip_p": 0.5}
LEJEPA = {"lambda": 0.02, "sigreg_t_max": 3.0, "sigreg_knots": 5,
          "sigreg_num_slices": 16, "sigreg_seed": 123}
STEP1_ONLY = {"num_workers": 0, "epochs": 1, "batch_size": 4, "lr": 1.0e-3,
              "min_lr": 1.0e-6, "beta1": 0.9, "beta2": 0.95, "eps": 1.0e-8,
              "weight_decay": 0.05, "warmup_epochs": 0, "clip_grad": 0.0}
TRAIN = {**MODEL, **AUG, **LEJEPA, **STEP1_ONLY}
EVAL_TRAIN = {"name": NAME, "img_size": IMG, "drop_path_rate": 0.0,
              "epochs": 2, "batch_size": 4, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}


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
        self.tmp = Path(tempfile.mkdtemp(prefix="lejepa-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0, "data_root": str(self.tmp / "data"),
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
        return load("lejepa_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.build_lejepa(model_name=NAME, img_size=IMG, drop_path_rate=0.0,
                              proj_hidden_dim=64, proj_dim=PROJ_DIM,
                              proj_layers=2, final_bn=True)

    def _views(self, torch, n=4, v=VIEWS):
        return torch.randn(n, v, 3, IMG, IMG)

    @needs_deps
    def test_forward_returns_features_and_projections(self):
        import torch
        model = self._model(self.models())
        features, proj = model(self._views(torch))
        self.assertEqual(tuple(features.shape), (4, VIEWS, FEATURE_DIM))
        self.assertEqual(tuple(proj.shape), (VIEWS, 4, PROJ_DIM))
        self.assertTrue(torch.isfinite(proj).all())

    @needs_deps
    def test_get_encoder_gives_one_feature_per_image(self):
        import torch
        model = self._model(self.models())
        enc = model.get_encoder()
        feats = enc(torch.randn(3, 3, IMG, IMG))
        self.assertEqual(tuple(feats.shape), (3, FEATURE_DIM))

    @needs_deps
    def test_identical_views_have_zero_invariance(self):
        # The invariance loss is (proj.mean(0) - proj)^2; identical views must
        # give identical projections across the view axis, so a mean of zero.
        import torch
        model = self._model(self.models()).eval()
        one = torch.randn(4, 1, 3, IMG, IMG)
        views = one.expand(4, VIEWS, 3, IMG, IMG).contiguous()
        with torch.no_grad():
            _f, proj = model(views)
        inv = (proj.mean(dim=0, keepdim=True) - proj).abs().max().item()
        self.assertLess(inv, 1e-4, "identical views did not give equal projections")

    @needs_deps
    def test_the_encoder_carries_no_projector(self):
        model = self._model(self.models())
        enc = model.get_encoder()
        self.assertFalse(any(k.startswith("projector.") for k in enc.state_dict()),
                         "the encoder must not carry the projection MLP")


class TestTheSIGReg(unittest.TestCase):
    def models(self):
        return load("lejepa_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_returns_a_finite_nonnegative_scalar(self):
        import torch
        torch.manual_seed(0)
        sig = self.models().SIGReg(t_max=3.0, knots=5, num_slices=16, seed=123)
        proj = torch.randn(VIEWS, 8, PROJ_DIM)
        val = sig(proj)
        self.assertEqual(val.dim(), 0)
        self.assertTrue(torch.isfinite(val))
        # A finite batch never matches the Gaussian target exactly, so the
        # statistic is strictly positive (a zeroed statistic would be a dead
        # regularizer).
        self.assertGreater(val.item(), 0.0)

    @needs_deps
    def test_too_few_knots_is_refused(self):
        with self.assertRaises(ValueError):
            self.models().SIGReg(t_max=3.0, knots=2, num_slices=16, seed=123)

    @needs_deps
    def test_the_slice_seed_advances_each_call(self):
        import torch
        m = self.models()
        proj = torch.randn(VIEWS, 8, PROJ_DIM)
        a = m.SIGReg(t_max=3.0, knots=5, num_slices=16, seed=123)
        first = a(proj).item()
        self.assertEqual(int(a.step.item()), 1, "the slice-seed step did not advance")
        b = m.SIGReg(t_max=3.0, knots=5, num_slices=16, seed=123)
        self.assertAlmostEqual(b(proj).item(), first, places=5,
                               msg="a fresh SIGReg is not reproducible on the same input")


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("lejepa_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_stacked_views_and_a_label(self):
        tiny_imagefolder(self.tmp / "data" / "train")
        d = self.dataset_mod()
        ds = d.MultiViewImageFolder(
            str(self.tmp / "data"),
            d.build_train_transform(img_size=IMG, blur_p=0.0, solarize_p=0.0),
            views=VIEWS)
        views, label = ds[0]
        self.assertEqual(tuple(views.shape), (VIEWS, 3, IMG, IMG))

    @needs_deps
    def test_the_loader_batches_views(self):
        tiny_imagefolder(self.tmp / "data" / "train")
        loader, _ = self.dataset_mod().get_lejepa_dataloader(
            str(self.tmp / "data"), batch_size=4, views=VIEWS,
            num_workers=0, img_size=IMG, seed=0, blur_p=0.0, solarize_p=0.0)
        batch_views, labels = next(iter(loader))
        self.assertEqual(tuple(batch_views.shape), (4, VIEWS, 3, IMG, IMG))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_backbone_comes_out_prefix_stripped(self):
        got = adapter.extract_encoder({
            "backbone.blocks.0.norm1.weight": 1,
            "backbone.cls_token": 2,
            "projector.net.0.weight": 3, "projector.net.1.weight": 4})
        self.assertEqual(set(got), {"blocks.0.norm1.weight", "cls_token"})

    def test_the_projector_is_left_out(self):
        got = adapter.extract_encoder({"backbone.norm.weight": 1,
                                       "projector.net.0.bias": 2})
        self.assertEqual(set(got), {"norm.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"projector.net.0.weight": 1})
        self.assertIn("backbone", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["name"], NAME)
        self.assertEqual(built["augmentation"]["views"], VIEWS)
        self.assertEqual(built["lejepa"]["sigreg"]["knots"], 5)
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
        cfg = self.eval_config(train={"views": VIEWS})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("views", str(e.exception))


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
        return load("lejepa_trainer", METHOD / "train_step1_lejepa.py")

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
        src = (METHOD / "train_step1_lejepa.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


class TestAStep1Smoke(Base):
    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data" / "train")
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
        backbone = adapter.load_encoder(saved, self.eval_config())
        loaded = backbone.backbone.state_dict()
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
        tiny_imagefolder(s1data / "train")
        s1cfg = {"stage": "pretrain", "seed": 0, "data_root": str(s1data),
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
        tree = ast.parse((METHOD / "train_step1_lejepa.py").read_text())
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
