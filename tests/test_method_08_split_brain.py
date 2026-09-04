#!/usr/bin/env python3
"""Specification for 08_split_brain (Zhang, Isola & Efros, CVPR 2017).

Split-Brain Autoencoders: an image is converted to CIE Lab and split into its L
and ab channels; **two cross-channel** sub-networks predict one from the other --
net1 maps L -> the quantised ab channels (313 bins), net2 maps ab -> the quantised
L channel (50 bins), each as a per-pixel classification. `encoder.pt` is the two
sub-network encoders; `linear_eval` probes their concatenated features (512-d).

The RGB->Lab conversion and the ab/L quantisation are pure numpy (the capture's
own comment says the released ab target IS NumPy argmin), so this ports
self-contained, torch-only -- scipy/scikit-image are not dependencies. The 313
ab-bin constant (`pts_in_hull.npy`, sha256 pinned) is vendored. The captured ViT
step 2 (timm) is excluded, as in every port.
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
METHOD = ROOT / "methods" / "08_split_brain"
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
    HAVE_DEPS, "08_split_brain needs torch, numpy, torchvision")

try:
    import timm                                          # noqa: F401
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("splitbrain_adapter", METHOD / "adapter" / "__init__.py")

AB_CLASSES = 313
L_CLASSES = 50
ENCODER_DIM = 512  # net1 encoder (256) + net2 encoder (256), concatenated

# Small enough to train a step on a CPU: a 32px crop, one epoch.
MODEL = {"crop_size": 32}
PRETRAIN_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1.0e-3,
              "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.0,
              "lr_decay_epochs": [], "lr_decay_rate": 0.1}
TRAIN = {**MODEL, **PRETRAIN_ONLY}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}


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
        self.tmp = Path(tempfile.mkdtemp(prefix="splitbrain-"))
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


class TestColourAndQuantization(unittest.TestCase):
    ORACLES = {
        (255, 255, 255): (100.00, 0.00, 0.00),
        (0, 0, 0): (0.00, 0.00, 0.00),
        (255, 0, 0): (53.24, 80.09, 67.20),
        (0, 0, 255): (32.30, 79.19, -107.86),
    }

    def data_mod(self):
        return load("splitbrain_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_rgb2lab_matches_cie_reference(self):
        import numpy as np
        d = self.data_mod()
        for (r, g, b), (eL, ea, eb) in self.ORACLES.items():
            lab = np.asarray(d.rgb2lab(np.full((4, 4, 3), (r, g, b),
                                               dtype="uint8")))
            got = lab.reshape(-1, 3).mean(axis=0)
            for got_c, exp_c, name in zip(got, (eL, ea, eb), "Lab"):
                self.assertAlmostEqual(float(got_c), exp_c, delta=0.05,
                                       msg=f"{name} of rgb({r},{g},{b})")

    @needs_deps
    def test_the_codebook_is_313_by_2(self):
        import numpy as np
        self.assertEqual(tuple(np.asarray(self.data_mod().load_ab_codebook())
                               .shape), (AB_CLASSES, 2))

    @needs_deps
    def test_quantize_l_gives_50_bins(self):
        import numpy as np
        d = self.data_mod()
        L = np.array([[0.0, 50.0], [99.9, 100.0]], dtype="float32")
        q = d.quantize_l(L)
        self.assertTrue(int(q.min()) >= 0 and int(q.max()) < L_CLASSES)

    @needs_deps
    def test_quantize_ab_gives_valid_bins(self):
        import numpy as np
        d = self.data_mod()
        ab = np.array([[[50.0, -20.0], [0.0, 0.0]]], dtype="float32")
        q = d.quantize_ab(ab)
        self.assertEqual(tuple(q.shape), (1, 2))
        self.assertTrue(int(q.min()) >= 0 and int(q.max()) < AB_CLASSES)


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("splitbrain_models", METHOD / "models" / "__init__.py")

    def _inputs(self, torch, b=2):
        return (torch.randn(b, 1, MODEL["crop_size"], MODEL["crop_size"]),
                torch.randn(b, 2, MODEL["crop_size"], MODEL["crop_size"]))

    @needs_deps
    def test_forward_predicts_both_cross_channels(self):
        import torch
        model = self.models().build_split_brain_from_config({})
        model.eval()
        l, ab = self._inputs(torch)
        ab_pred, l_pred = model(l, ab)
        self.assertEqual(ab_pred.shape[0], 2)
        self.assertEqual(ab_pred.shape[1], AB_CLASSES)   # net1: L -> ab bins
        self.assertEqual(l_pred.shape[1], L_CLASSES)     # net2: ab -> L bins
        self.assertEqual(ab_pred.shape[2:], l_pred.shape[2:])

    @needs_deps
    def test_extract_features_concatenates_both_encoders(self):
        import torch
        model = self.models().build_split_brain_from_config({})
        model.eval()
        l, ab = self._inputs(torch)
        feats = model.extract_features(l, ab)
        self.assertEqual(tuple(feats.shape), (2, ENCODER_DIM))


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("splitbrain_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_inputs_targets_and_a_label(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().SplitBrainDataset(
            str(self.tmp / "data"), crop_size=MODEL["crop_size"], train=True)
        item = ds[0]
        self.assertEqual(len(item), 5)
        l_input, ab_input, l_target, ab_target, label = item
        cs = MODEL["crop_size"]
        self.assertEqual(tuple(l_input.shape), (1, cs, cs))
        self.assertEqual(tuple(ab_input.shape), (2, cs, cs))
        self.assertEqual(tuple(l_target.shape), (cs, cs))
        self.assertEqual(tuple(ab_target.shape), (cs, cs))
        self.assertTrue(0 <= int(l_target.min()) and int(l_target.max()) < L_CLASSES)
        self.assertTrue(0 <= int(ab_target.min()) and int(ab_target.max()) < AB_CLASSES)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_two_encoders_come_out(self):
        got = adapter.extract_encoder({
            "net1.encoder.0.weight": 1, "net2.encoder.0.weight": 2,
            "net1.decoder.0.weight": 3, "net2.decoder.4.weight": 4})
        self.assertEqual(set(got),
                         {"net1.encoder.0.weight", "net2.encoder.0.weight"})

    def test_the_decoders_are_left_out(self):
        got = adapter.extract_encoder({"net1.encoder.3.weight": 1,
                                       "net1.decoder.0.weight": 2,
                                       "net2.decoder.2.weight": 3})
        self.assertEqual(set(got), {"net1.encoder.3.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"net1.decoder.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
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
        return load("splitbrain_trainer",
                    METHOD / "train_pretrain_split_brain.py")

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
        src = (METHOD / "train_pretrain_split_brain.py").read_text()
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
    def test_no_distributed_tensorboard_or_timm_machinery_is_used(self):
        import ast
        tree = ast.parse((METHOD / "train_pretrain_split_brain.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)
        self.assertNotIn("timm", used)


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native
# AlexNet path. The two cross-channel branches keep their roles, but each backbone
# is a half-width ViT-B/16 (embed_dim 384, 6 heads, per-branch in_chans 1/2) + a
# conv decoder. encoder.pt keeps net1.encoder.* / net2.encoder.*; the eval feature
# is the concatenated CLS of both branches. Tiny dims so a CPU smoke is cheap.
VIT_MODEL_ARGS = {"img_size": 32, "patch_size": 16, "embed_dim": 16, "depth": 1,
                  "num_heads": 2, "mlp_ratio": 4.0}
VIT_MODEL_KNOBS = {"crop_size": 32, "patch_size": 16, "embed_dim": 16,
                   "depth": 1, "num_heads": 2, "mlp_ratio": 4.0}
VIT_TRAIN_TINY = {"arch": "vit", **VIT_MODEL_KNOBS, "epochs": 2, "batch_size": 2,
                  "num_workers": 0, "lr": 6.0e-4, "weight_decay": 0.05,
                  "warmup_epochs": 0, "min_lr": 0.0, "save_at_epochs": [1, 2]}
VIT_EVAL_TINY = {"arch": "vit", **VIT_MODEL_KNOBS, "epochs": 2, "batch_size": 2,
                 "num_workers": 0, "lr": 0.1, "momentum": 0.9,
                 "weight_decay": 0.0}
FEATURE_DIM_VIT = 2 * VIT_MODEL_KNOBS["embed_dim"]   # both branches' CLS


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
        self.assertEqual(built["model"]["num_heads"], 2)
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
        for key in ("embed_dim", "warmup_epochs", "min_lr", "save_at_epochs"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(train={key: 1}), self.out)
                self.assertIn(key, str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_split_brain", METHOD / "models" / "vit_split_brain.py")
        return vm.build_split_brain_vit(**VIT_MODEL_ARGS)

    @needs_timm
    def test_extract_features_concatenates_both_branches(self):
        import torch
        feats = self._model().extract_features(torch.randn(2, 1, 32, 32),
                                               torch.randn(2, 2, 32, 32))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM_VIT))

    @needs_timm
    def test_forward_predicts_both_cross_channel_maps(self):
        import torch
        ab_pred, l_pred = self._model()(torch.randn(2, 1, 32, 32),
                                        torch.randn(2, 2, 32, 32))
        # crop 32, patch 16 -> grid 2 -> decoder x4 -> 8x8
        self.assertEqual(tuple(ab_pred.shape), (2, AB_CLASSES, 8, 8))
        self.assertEqual(tuple(l_pred.shape), (2, L_CLASSES, 8, 8))

    @needs_timm
    def test_the_bidirectional_loss_reaches_every_parameter(self):
        # Split-brain's ab-CE + L-CE must touch both branches (the capture's
        # DDP-no-unused-parameters property); a dead branch is a real bug.
        import torch
        import torch.nn.functional as F
        model = self._model()
        ab_pred, l_pred = model(torch.randn(2, 1, 32, 32),
                                torch.randn(2, 2, 32, 32))
        ab_t = torch.randint(0, AB_CLASSES, (2, 8, 8))
        l_t = torch.randint(0, L_CLASSES, (2, 8, 8))
        (F.cross_entropy(ab_pred, ab_t) + F.cross_entropy(l_pred, l_t)).backward()
        dead = [n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None]
        self.assertEqual(dead, [], f"parameters got no gradient: {dead[:5]}")

    @needs_timm
    def test_encoder_pt_holds_only_the_two_trunks(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith(("net1.encoder.", "net2.encoder."))
                            for k in got))
        self.assertFalse([k for k in got if "decoder" in k])

    @needs_timm
    def test_load_encoder_round_trips_the_trunks(self):
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
        self.assertGreater(pairs, 0, "no saved weight reached the trunks")


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

    @needs_timm
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


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. It reuses this method's
    own encoder loader and eval pipeline, so the check is that it returns the
    512-d concatenated branch-encoder feature -- raw, before the probe's
    normalise -- one row per val image, with honest meta.

    Unlike the ResNet backbones, Split-Brain's eval feature is the whole
    two-branch model (its `extract_features(l, ab)`), so the provider passes the
    model returned by `load_encoder` directly to `extract_features`, mirroring
    the eval main. The encoder.pt is built from the *shipped* linear_eval
    config's architecture (the native AlexNet path) via the same
    `extract_encoder` filter the adapter writes with; random weights do not
    affect the shape-and-plumbing this proves. Modules load through `load`
    (`load_from`), which purges any other method's `adapter`/`models` first --
    the whole suite runs many methods in one interpreter.
    """

    def _shipped_config(self) -> dict:
        import yaml
        return yaml.safe_load(
            (METHOD / "configs" / "linear_eval.yaml").read_text())

    def _make_encoder(self, cfg: dict) -> Path:
        import torch
        models = load("splitbrain_models", METHOD / "models" / "__init__.py")
        model = models.build_split_brain_from_config({})
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("splitbrain_feature_provider",
                    METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_512d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("08_split_brain provider not yet present")
        import numpy as np
        data_root = tiny_split(self.tmp / "data")
        cfg = self._shipped_config()
        encoder_pt = self._make_encoder(cfg)

        prov = self._provider()
        feats, labels, meta = prov.extract_val_features(
            encoder_path=str(encoder_pt), data_root=str(data_root),
            split="val", device="cpu", batch_size=2, num_workers=0)

        feats = np.asarray(feats)
        self.assertEqual(feats.ndim, 2)
        self.assertEqual(feats.shape[0], 6, "6 val images expected")
        self.assertEqual(feats.shape[1], ENCODER_DIM,
                         "concatenated branch feature is 512-d")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], ENCODER_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("08_split_brain provider not yet present")
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        data_root = tiny_split(self.tmp / "data")
        encoder_pt = self._make_encoder(self._shipped_config())

        record = {"method": METHOD.name, "status": "ready",
                  "provider": str(prov_path), "encoder": str(encoder_pt)}
        out = self.tmp / "features"
        updated = driver.extract_one(
            record, data_root=str(data_root), split="val", out=out,
            device="cpu", batch_size=2, num_workers=0)

        self.assertEqual(updated["status"], "ok",
                         updated.get("reason", ""))
        method_out = out / METHOD.name
        feats = np.load(method_out / "features.npy")
        labels = np.load(method_out / "labels.npy")
        meta = json.loads((method_out / "meta.json").read_text())
        self.assertEqual(feats.shape, (6, ENCODER_DIM))
        self.assertEqual(labels.shape[0], 6)
        self.assertEqual(meta["encoder_sha256"],
                         hashlib.sha256(encoder_pt.read_bytes()).hexdigest())

    @needs_deps
    def test_the_isolated_driver_run_extracts_this_method_end_to_end(self):
        """The whole driver, real subprocess, real provider. Unlike the
        synthetic-provider driver test in tests/test_extract_features.py, this
        runs a method whose adapter imports the shared `adapterlib` -- so it
        catches the class of regression where the isolated worker cannot see a
        repository-root module the provider needs (the worker puts ROOT on
        sys.path, as bin/launch.py sets PYTHONPATH=ROOT)."""
        if not (METHOD / "feature_provider.py").is_file():
            self.skipTest("08_split_brain provider not yet present")
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        data_root = tiny_split(self.tmp / "data")
        encoder_pt = self._make_encoder(self._shipped_config())
        out = self.tmp / "features"
        manifest = driver.run(
            METHOD.parent, data_root=str(data_root), split="val", out=out,
            encoders={METHOD.name: str(encoder_pt)}, encoders_root=None,
            device="cpu", batch_size=2, num_workers=0,
            venvs_root=ROOT / ".venvs")

        rec = {r["method"]: r for r in manifest["records"]}[METHOD.name]
        self.assertEqual(rec["status"], "ok", rec.get("reason", ""))
        feats = np.load(out / METHOD.name / "features.npy")
        self.assertEqual(feats.shape, (6, ENCODER_DIM))


if __name__ == "__main__":
    unittest.main()
