#!/usr/bin/env python3
"""Specification for 03_colorization (Zhang, Isola & Efros, ECCV 2016).

Colorization as a self-supervised pretext: an image is converted to CIE Lab, the
**L** (lightness) channel is the input, and a VGG-style CNN predicts the **ab**
colour channels, quantised into 313 in-gamut bins, as a per-pixel classification
(rebalanced cross-entropy). `encoder.pt` is the CNN encoder trunk; `linear_eval`
probes its global-average-pooled feature.

Despite the capture's requirements.txt naming opencv / scikit-image / scikit-learn,
the lab's own code uses none of them: the RGB->Lab conversion and the ab
quantisation are pure numpy. So this ports self-contained, torch-only. The 313
ab-bin centres (`pts_in_hull.npy`, Zhang et al.) are vendored as a constant. The
captured ViT step 2 is excluded, as in every port.
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
METHOD = ROOT / "methods" / "03_colorization"
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
    HAVE_DEPS, "03_colorization needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("colorization_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to train a step on a CPU: a 32px crop (the encoder downsamples by
# 8, so conv7 is 4x4), one epoch, class rebalancing off. The paper's 224px crop /
# 313 bins / 300 epochs / rebalancing live in the shipped config.
MODEL = {"num_bins": 313, "img_size": 36, "crop_size": 32}
PRETRAIN_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1.0e-5,
              "beta1": 0.9, "beta2": 0.999, "weight_decay": 1.0e-4,
              "lr_decay_epochs": [], "lr_decay_rate": 0.5,
              "use_class_rebalancing": False, "rebalance_lambda": 0.5,
              "rebalance_sample_size": 4}
TRAIN = {**MODEL, **PRETRAIN_ONLY}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

ENCODER_DIM = 512  # global-average-pooled conv7 feature


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
        self.tmp = Path(tempfile.mkdtemp(prefix="color-"))
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


class TestRGB2Lab(unittest.TestCase):
    """The numpy Lab conversion must match published CIE Lab (D65, sRGB)."""

    ORACLES = {
        (255, 255, 255): (100.00, 0.00, 0.00),
        (0, 0, 0): (0.00, 0.00, 0.00),
        (255, 0, 0): (53.24, 80.09, 67.20),
        (0, 255, 0): (87.74, -86.18, 83.18),
        (0, 0, 255): (32.30, 79.19, -107.86),
    }

    @needs_deps
    def test_solid_colors_match_cie_reference(self):
        import numpy as np
        data = load("colorization_data", METHOD / "data" / "__init__.py")
        for (r, g, b), (eL, ea, eb) in self.ORACLES.items():
            img = np.full((4, 4, 3), (r, g, b), dtype="uint8")
            l_ch, ab = data.rgb_to_lab(img)
            got = (float(l_ch.mean()), float(ab[..., 0].mean()),
                   float(ab[..., 1].mean()))
            for got_c, exp_c, name in zip(got, (eL, ea, eb), "Lab"):
                self.assertAlmostEqual(got_c, exp_c, delta=0.05,
                                       msg=f"{name} of rgb({r},{g},{b})")


class TestAbQuantization(unittest.TestCase):
    def data_mod(self):
        return load("colorization_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_the_vendored_hull_is_313_by_2(self):
        import numpy as np
        pts = self.data_mod().get_ab_points()
        self.assertEqual(tuple(np.asarray(pts).shape), (313, 2))

    @needs_deps
    def test_quantize_gives_valid_bin_indices(self):
        import numpy as np
        d = self.data_mod()
        ab = np.array([[[50.0, -20.0], [-10.0, 30.0]], [[0.0, 0.0], [80.0, 60.0]]],
                      dtype="float32")  # [2, 2, 2]
        bins = d.quantize_ab_fast(ab)
        self.assertEqual(tuple(bins.shape), (2, 2))
        self.assertTrue(int(bins.min()) >= 0 and int(bins.max()) < 313)

    @needs_deps
    def test_class_weights_cover_all_bins_and_are_positive(self):
        import torch
        tmp = Path(tempfile.mkdtemp(prefix="cw-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        tiny_imagefolder(tmp / "data")
        w = self.data_mod().get_class_weights(str(tmp / "data"), num_bins=313,
                                              sample_size=4, lambda_smooth=0.5)
        w = torch.as_tensor(w)
        self.assertEqual(tuple(w.shape), (313,))
        self.assertTrue(bool((w > 0).all()))


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("colorization_models", METHOD / "models" / "__init__.py")

    def _l(self, torch, b=2):
        return torch.randn(b, 1, MODEL["crop_size"], MODEL["crop_size"])

    @needs_deps
    def test_forward_predicts_an_ab_distribution(self):
        import torch
        m = self.models()
        model = m.build_colorization_cnn({"model": {"num_bins": 313}})
        model.eval()
        out = model(self._l(torch))
        self.assertEqual(tuple(out.shape),
                         (2, 313, MODEL["crop_size"], MODEL["crop_size"]))

    @needs_deps
    def test_the_encoder_returns_one_feature_per_image(self):
        import torch
        m = self.models()
        enc = m.build_colorization_cnn({"model": {"num_bins": 313}}).get_encoder()
        enc.eval()
        feats = enc(self._l(torch))
        self.assertEqual(tuple(feats.shape), (2, ENCODER_DIM))


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("colorization_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_an_l_channel_and_ab_bins(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().ColorizationDataset(
            str(self.tmp / "data"), mode="train",
            image_size=MODEL["img_size"], crop_size=MODEL["crop_size"])
        l_ch, bins = ds[0]
        self.assertEqual(tuple(l_ch.shape),
                         (1, MODEL["crop_size"], MODEL["crop_size"]))
        self.assertEqual(tuple(bins.shape),
                         (MODEL["crop_size"], MODEL["crop_size"]))
        self.assertEqual(bins.dtype, torch.int64)
        self.assertTrue(int(bins.min()) >= 0 and int(bins.max()) < 313)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_encoder_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.0.weight": 1, "encoder.3.weight": 2,
            "decoder.0.weight": 3, "head.weight": 4, "head.bias": 5})
        self.assertEqual(set(got), {"encoder.0.weight", "encoder.3.weight"})

    def test_the_decoder_and_head_are_left_out(self):
        got = adapter.extract_encoder({"encoder.1.weight": 1,
                                       "decoder.2.weight": 2, "head.weight": 3})
        self.assertEqual(set(got), {"encoder.1.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"head.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["num_bins"], 313)
        self.assertEqual(built["data"]["crop_size"], 32)
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
        # beta1 is a step-1 Adam knob; the probe (SGD) must reject it as unknown.
        cfg = self.eval_config(train={"beta1": 0.9})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("beta1", str(e.exception))


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
        self.assertEqual(adapter.PRETRAIN_METRIC_NAMES["final_loss"],
                         "final_pretext_loss")
        for target in adapter.PRETRAIN_METRIC_NAMES.values():
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
        return load("colorization_trainer",
                    METHOD / "train_pretrain_colorization.py")

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
        src = (METHOD / "train_pretrain_colorization.py").read_text()
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
        tree = ast.parse((METHOD / "train_pretrain_colorization.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native CNN
# path. A from-scratch, self-contained ViT-B/16 (hand-written, no timm) reads the
# L channel; a CNN decoder produces the same 313-bin ab classification. The trunk
# lives under encoder.*, so the same extract/probe convention applies. Tiny dims
# so a CPU smoke is cheap; num_bins stays 313 (the dataset quantizes to 313).
VIT_MODEL_ARGS = {"img_size": 32, "patch_size": 16, "num_bins": 313,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_MODEL_KNOBS = {"img_size": 36, "crop_size": 32, "patch_size": 16,
                   "embed_dim": 16, "depth": 1, "num_heads": 2,
                   "mlp_ratio": 4.0, "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_TRAIN_TINY = {"arch": "vit", **VIT_MODEL_KNOBS, "num_bins": 313,
                  "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 1.5e-4,
                  "weight_decay": 0.05, "warmup_epochs": 0, "min_lr": 0.0,
                  "clip_grad": 1.0, "use_class_rebalancing": False,
                  "rebalance_lambda": 0.5, "rebalance_sample_size": 4,
                  "save_at_epochs": [1, 2]}
VIT_EVAL_TINY = {"arch": "vit", **VIT_MODEL_KNOBS, "epochs": 2, "batch_size": 2,
                 "num_workers": 0, "lr": 0.1, "momentum": 0.9,
                 "weight_decay": 0.0}
FEATURE_DIM_VIT = VIT_MODEL_KNOBS["embed_dim"]


class TestVitConfigTranslation(Base):
    def vit_config(self, train=None, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": dict(train if train is not None else VIT_TRAIN_TINY)}
        for k, v in over.items():
            cfg[k] = v
        return cfg

    def test_the_vit_step2_config_is_accepted(self):
        built = adapter.to_run_config(self.vit_config(), self.out)
        self.assertEqual(built["arch"], "vit")
        self.assertEqual(built["model"]["embed_dim"], 16)
        self.assertEqual(built["model"]["num_bins"], 313)
        self.assertEqual(built["training"]["save_at_epochs"], [1, 2])

    def test_the_native_path_has_no_top_level_arch(self):
        built = adapter.to_run_config(self.config(), self.out)
        self.assertNotIn("arch", built)

    def test_a_bad_arch_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.vit_config(train={**VIT_TRAIN_TINY, "arch": "vitt"}),
                self.out)
        self.assertIn("arch", str(e.exception))

    def test_a_missing_vit_setting_is_refused_by_name(self):
        for key in VIT_TRAIN_TINY:
            if key == "arch":
                continue
            with self.subTest(key=key):
                t = {k: v for k, v in VIT_TRAIN_TINY.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.vit_config(train=t), self.out)
                self.assertIn(key, str(e.exception))

    def test_a_native_knob_does_not_leak_into_the_vit_path(self):
        for key in ("beta1", "beta2", "lr_decay_rate"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(
                        self.vit_config(train={**VIT_TRAIN_TINY, key: 1}),
                        self.out)
                self.assertIn(key, str(e.exception))

    def test_a_vit_knob_does_not_leak_into_the_native_path(self):
        for key in ("embed_dim", "warmup_epochs", "min_lr", "clip_grad",
                    "save_at_epochs"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(train={key: 1}), self.out)
                self.assertIn(key, str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_colorization", METHOD / "models" / "vit_colorization.py")
        return vm.build_vit_colorization(**VIT_MODEL_ARGS)

    @needs_deps
    def test_get_encoder_returns_the_cls_feature(self):
        import torch
        feats = self._model().get_encoder()(torch.randn(2, 1, 32, 32))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM_VIT))

    @needs_deps
    def test_forward_is_a_per_pixel_ab_classification(self):
        import torch
        out = self._model()(torch.randn(2, 1, 32, 32))
        self.assertEqual(tuple(out.shape),
                         (2, VIT_MODEL_ARGS["num_bins"], 32, 32))

    @needs_deps
    def test_encoder_pt_holds_only_the_trunk(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith("encoder.") for k in got))
        self.assertFalse([k for k in got if k.startswith("decoder")])

    @needs_deps
    def test_load_encoder_round_trips_the_trunk(self):
        import torch
        saved = adapter.extract_encoder(self._model().state_dict())
        cfg = {"train": {"arch": "vit", **VIT_MODEL_KNOBS}}
        model = adapter.load_encoder(saved, cfg)
        loaded = model.state_dict()
        pairs = 0
        for k, want in saved.items():
            got = loaded.get(k)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{k} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the trunk")


class TestAVitStep2Smoke(Base):
    def _adapter(self, cfg_dict, out):
        cfg = self.tmp / (out.name + ".json")
        cfg.write_text(json.dumps(cfg_dict), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(out)], cwd=METHOD, env=env,
            capture_output=True, text=True)
        return cfg, r

    @needs_deps
    def test_pretrain_milestones_then_probe_passes_contract(self):
        tiny_imagefolder(self.tmp / "data")
        pre = self.tmp / "pre_out"
        _, r = self._adapter(
            {"stage": "pretrain", "seed": 0,
             "data_root": str(self.tmp / "data"), "device": "cpu",
             "train": dict(VIT_TRAIN_TINY)}, pre)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((pre / "encoder.pt").is_file())
        for n in (1, 2):
            self.assertTrue((pre / f"encoder_epoch{n}.pt").is_file(),
                            f"milestone encoder_epoch{n}.pt not written")

        tiny_split(self.tmp / "eval")
        ev = self.tmp / "eval_out"
        cfg, r = self._adapter(
            {"stage": "linear_eval", "seed": 0,
             "data_root": str(self.tmp / "eval"), "device": "cpu",
             "encoder": str(pre / "encoder_epoch2.pt"),
             "train": dict(VIT_EVAL_TINY)}, ev)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out", str(ev),
             "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        m = json.loads((ev / "metrics.json").read_text())["metrics"]
        self.assertIn("final_linear_probe_top1_accuracy", m)
        self.assertFalse((ev / "encoder.pt").exists())


if __name__ == "__main__":
    unittest.main()
