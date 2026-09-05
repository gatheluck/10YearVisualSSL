#!/usr/bin/env python3
"""Specification for 11_cpc (van den Oord et al., 2018; arXiv:1807.03748).

Contrastive Predictive Coding, the **visual CPC 2018** path. This port covers the
lab's paper-faithful variant (`visual_cpc2018`): each image becomes a grid of
overlapping patches, a ResNet-v2-101-style no-BN encoder maps every patch to a
z-vector, a PixelCNN-style masked-convolution context autoregresses over the grid,
and an InfoNCE loss predicts future rows' z-vectors from the context. `encoder.pt`
is the patch encoder; `linear_eval` probes the grid-averaged z (`avg_z`).

The capture's older local baseline (`cpc_resnet`) is a documented protocol
mismatch (its own CPC_PRETRAIN_PAPER_READY_BLOCK.md marks it "must not be submitted
as a paper-ready Step 1 job"), so it is excluded; only the corrected
`visual_cpc2018` path is the native port. The capture's unified ViT-B/16 Step 2
(`arch: vit`) is ported additively: the ViT's patch tokens as the CPC z-grid, a
column-GRU context, InfoNCE, probed at the ViT CLS token.
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
METHOD = ROOT / "methods" / "11_cpc"
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
    HAVE_DEPS, "11_cpc needs torch, numpy, torchvision")

try:
    import timm                                        # noqa: F401
    HAVE_TIMM = HAVE_DEPS
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("cpc_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to train a step on a CPU: a narrow encoder, a 2x2 patch grid, one
# prediction step. The paper's 7x7 grid / width_mult 1.0 / 5 steps live in the
# shipped config; the port relaxes the grid check so a 2x2 smoke can run.
MODEL = {"z_dim": 256, "c_dim": 256, "pred_steps": 1, "context_layers": 2,
         "encoder_width_mult": 0.25}
DATA = {"source_size": 40, "img_size": 32, "patch_size": 16,
        "patch_crop_size": 14, "stride": 16}
TRAIN = {**MODEL, **DATA, "epochs": 1, "batch_size": 2, "num_workers": 0,
         "lr": 2.0e-4, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.0}
EVAL_TRAIN = {**MODEL, **DATA, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}
GRID = 2
FEATURE_DIM = MODEL["z_dim"]


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "train" / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype="uint8")).save(
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
                base = np.full((64, 64, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (64, 64, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cpc-"))
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
               "encoder": str(self.tmp / "encoder.pt"), "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("cpc_models", METHOD / "models" / "__init__.py")

    def _grid(self, torch, b=2):
        return torch.randn(b, GRID, GRID, 3, DATA["patch_size"],
                           DATA["patch_size"])

    @needs_deps
    def test_forward_returns_z_and_context_grids(self):
        import torch
        m = self.models()
        model = m.build_visual_cpc2018_from_config({"model": MODEL})
        z_grid, c_grid = model(self._grid(torch))
        self.assertEqual(tuple(z_grid.shape), (2, GRID, GRID, MODEL["z_dim"]))
        self.assertEqual(tuple(c_grid.shape), (2, GRID, GRID, MODEL["c_dim"]))

    @needs_deps
    def test_the_encoder_returns_avg_z_per_image(self):
        import torch
        m = self.models()
        enc = m.build_visual_cpc2018_from_config({"model": MODEL}).get_encoder()
        enc.eval()
        feats = enc(self._grid(torch))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))

    @needs_deps
    def test_cpc_loss_is_a_finite_scalar(self):
        import torch
        m = self.models()
        model = m.build_visual_cpc2018_from_config({"model": MODEL})
        z_grid, c_grid = model(self._grid(torch))
        loss = model.cpc_loss(z_grid, c_grid, use_ddp_negatives=False)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("cpc_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_a_patch_grid_and_a_label(self):
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().VisualCPC2018Dataset(
            str(self.tmp / "data"), mode="train", image_size=DATA["img_size"],
            source_size=DATA["source_size"], patch_size=DATA["patch_size"],
            patch_crop_size=DATA["patch_crop_size"], stride=DATA["stride"])
        patches, label = ds[0]
        self.assertEqual(tuple(patches.shape),
                         (GRID, GRID, 3, DATA["patch_size"], DATA["patch_size"]))
        self.assertEqual(ds.grid_size, GRID)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_encoder_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.conv1.weight": 1, "encoder.layer1.0.conv1.weight": 2,
            "context.net.0.weight": 3, "predictors.0.weight": 4})
        self.assertEqual(set(got),
                         {"encoder.conv1.weight", "encoder.layer1.0.conv1.weight"})

    def test_the_context_and_predictors_are_left_out(self):
        got = adapter.extract_encoder({"encoder.conv1.weight": 1,
                                       "context.net.0.weight": 2,
                                       "predictors.0.weight": 3})
        self.assertEqual(set(got), {"encoder.conv1.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"context.net.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["z_dim"], 256)
        self.assertEqual(built["data"]["patch_size"], 16)
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

    def test_adam_settings_are_not_part_of_the_probe(self):
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
        return load("cpc_trainer", METHOD / "train_pretrain_cpc2018.py")

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
        src = (METHOD / "train_pretrain_cpc2018.py").read_text()
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
        tree = ast.parse((METHOD / "train_pretrain_cpc2018.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native
# visual-CPC-2018 patch-encoder path. A timm ViT-B/16's patch tokens become the
# CPC z-grid; a column-wise GRU gives the context; k linear predictors score
# InfoNCE. `encoder.pt` is the ViT (encoder.*), probed at its CLS token. Tiny
# dims for a CPU smoke: a 2x2 patch grid (img 32 / patch 16), embed_dim 16.
VIT_MODEL = {"z_dim": 16, "c_dim": 16, "pred_steps": 1, "img_size": 32,
             "patch_size": 16, "embed_dim": 16, "depth": 1, "num_heads": 2,
             "mlp_ratio": 4.0, "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_TRAIN_TINY = {"arch": "vit", **VIT_MODEL, "temperature": 0.07,
                  "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 6.0e-4,
                  "weight_decay": 0.05, "warmup_epochs": 0, "min_lr": 0.0,
                  "save_at_epochs": [1, 2]}
VIT_EVAL_TINY = {"arch": "vit", **VIT_MODEL, "epochs": 1, "batch_size": 2,
                 "num_workers": 0, "lr": 0.01, "momentum": 0.9,
                 "weight_decay": 0.0}


class TestVitConfigTranslation(Base):
    def vit_config(self, train=None, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": dict(train if train is not None else VIT_TRAIN_TINY)}
        cfg.update(over)
        return cfg

    def test_the_vit_step2_config_is_accepted(self):
        built = adapter.to_run_config(self.vit_config(), out=self.out)
        self.assertEqual(built["arch"], "vit")
        self.assertEqual(built["model"]["embed_dim"], 16)
        self.assertEqual(built["cpc"]["temperature"], 0.07)
        self.assertEqual(built["training"]["save_at_epochs"], [1, 2])

    def test_native_path_unchanged_when_arch_absent(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertNotIn("arch", built)

    def test_a_bad_arch_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.vit_config(train={**VIT_TRAIN_TINY,
                                                         "arch": "resnext"}),
                                  out=self.out)
        self.assertIn("arch", str(e.exception))

    def test_a_missing_vit_setting_is_refused_by_name(self):
        for key in VIT_TRAIN_TINY:
            if key == "arch":
                continue
            with self.subTest(key=key):
                t = {k: v for k, v in VIT_TRAIN_TINY.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.vit_config(train=t), out=self.out)
                self.assertIn(key, str(e.exception))

    def test_native_knob_does_not_leak_into_the_vit_path(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.vit_config(train={**VIT_TRAIN_TINY, "context_layers": 2}),
                out=self.out)
        self.assertIn("context_layers", str(e.exception))

    def test_a_vit_knob_does_not_leak_into_the_native_path(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"save_at_epochs": [1]}),
                                  out=self.out)
        self.assertIn("save_at_epochs", str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_cpc", METHOD / "models" / "vit_cpc.py")
        return vm.build_cpc_vit(**VIT_MODEL)

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, VIT_MODEL["img_size"], VIT_MODEL["img_size"])

    @needs_timm
    def test_the_encoder_returns_the_cls_feature(self):
        import torch
        feats = self._model().get_encoder()(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, VIT_MODEL["embed_dim"]))

    @needs_timm
    def test_forward_returns_z_and_context_grids(self):
        import torch
        z_grid, c_grid = self._model()(self._batch(torch))
        self.assertEqual(z_grid.shape, c_grid.shape)
        self.assertEqual(z_grid.shape[0], 2)
        self.assertEqual(z_grid.shape[-1], VIT_MODEL["z_dim"])

    @needs_timm
    def test_cpc_loss_is_a_finite_scalar(self):
        import torch
        model = self._model()
        z_grid, c_grid = model(self._batch(torch))
        loss = model.cpc_loss_fast(z_grid, c_grid, temperature=0.07)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @needs_timm
    def test_z_dim_must_equal_embed_dim(self):
        # The z-grid IS the ViT's patch tokens, so its channel count is fixed to
        # the ViT hidden dim; a mismatch would silently mis-reshape.
        vm = load("vit_cpc", METHOD / "models" / "vit_cpc.py")
        bad = {**VIT_MODEL, "z_dim": VIT_MODEL["embed_dim"] + 8}
        with self.assertRaises(ValueError) as e:
            vm.build_cpc_vit(**bad)
        self.assertIn("embed_dim", str(e.exception))

    @needs_timm
    def test_encoder_pt_holds_only_the_vit(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith("encoder.") for k in got))
        self.assertFalse([k for k in got if k.startswith("context")])
        self.assertFalse([k for k in got if k.startswith("predictors")])

    @needs_timm
    def test_load_encoder_round_trips_the_vit_weights(self):
        import torch
        saved = adapter.extract_encoder(self._model().state_dict())
        cfg = {"train": {"arch": "vit", **VIT_MODEL}}
        loaded = adapter.load_encoder(saved, cfg).state_dict()
        pairs = 0
        for k, want in saved.items():
            got = loaded.get(k)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{k} came back changed")
        self.assertGreater(pairs, 0)


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

    def _eval_cfg(self, encoder) -> dict:
        return {"stage": "linear_eval", "seed": 0,
                "data_root": str(self.tmp / "eval"), "device": "cpu",
                "encoder": str(encoder), "train": dict(VIT_EVAL_TINY)}

    @needs_timm
    def test_pretrain_milestones_then_probe_passes_contract(self):
        tiny_imagefolder(self.tmp / "data")
        pre = self.tmp / "pre_out"
        _, r = self._adapter(
            {"stage": "pretrain", "seed": 0, "data_root": str(self.tmp / "data"),
             "device": "cpu", "train": dict(VIT_TRAIN_TINY)}, pre)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((pre / "encoder.pt").is_file())
        for n in (1, 2):
            self.assertTrue((pre / f"encoder_epoch{n}.pt").is_file(),
                            f"milestone encoder_epoch{n}.pt not written")

        tiny_split(self.tmp / "eval")
        ev = self.tmp / "eval_out"
        cfg, r = self._adapter(self._eval_cfg(pre / "encoder_epoch2.pt"), ev)
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
    z_dim-d grid-averaged patch feature (`avg_z`) -- raw, before the probe's
    normalise -- one row per val image, with honest meta.

    The encoder.pt is built from the *shipped* linear_eval config's
    architecture (the provider reads that config). The shipped config carries
    no `arch` key, so it is the native `cpc2018` patch-encoder path with
    z_dim=1024; random weights do not affect the shape-and-plumbing this
    proves. Modules load through `load` (`load_from`), which purges any other
    method's `adapter`/`models` first -- the whole suite runs many methods in
    one interpreter.
    """

    # Shipped linear_eval.yaml: no `arch` key (native cpc2018), z_dim=1024.
    SHIPPED_FEATURE_DIM = 1024

    def _shipped_config(self) -> dict:
        import yaml
        return yaml.safe_load(
            (METHOD / "configs" / "linear_eval.yaml").read_text())

    def _make_encoder(self, cfg: dict) -> Path:
        import torch
        models = load("cpc_models", METHOD / "models" / "__init__.py")
        trainer = load("cpc_trainer", METHOD / "train_pretrain_cpc2018.py")
        model = models.build_visual_cpc2018_from_config(
            trainer.model_config(cfg["train"]))
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("cpc_feature_provider", METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_zdim_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("11_cpc provider not yet present")
        import numpy as np
        data_root = tiny_split(self.tmp / "data")
        cfg = self._shipped_config()
        self.assertEqual(cfg["train"].get("arch", "cpc2018"), "cpc2018",
                         "shipped config is the native cpc2018 path")
        encoder_pt = self._make_encoder(cfg)

        prov = self._provider()
        feats, labels, meta = prov.extract_val_features(
            encoder_path=str(encoder_pt), data_root=str(data_root),
            split="val", device="cpu", batch_size=2, num_workers=0)

        feats = np.asarray(feats)
        self.assertEqual(feats.ndim, 2)
        self.assertEqual(feats.shape[0], 6, "6 val images expected")
        self.assertEqual(feats.shape[1], self.SHIPPED_FEATURE_DIM,
                         "avg_z is z_dim-d (1024) on the shipped config")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], self.SHIPPED_FEATURE_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("11_cpc provider not yet present")
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

        self.assertEqual(updated["status"], "ok", updated.get("reason", ""))
        method_out = out / METHOD.name
        feats = np.load(method_out / "features.npy")
        labels = np.load(method_out / "labels.npy")
        meta = json.loads((method_out / "meta.json").read_text())
        self.assertEqual(feats.shape, (6, self.SHIPPED_FEATURE_DIM))
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
            self.skipTest("11_cpc provider not yet present")
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
        self.assertEqual(feats.shape, (6, self.SHIPPED_FEATURE_DIM))


if __name__ == "__main__":
    unittest.main()
