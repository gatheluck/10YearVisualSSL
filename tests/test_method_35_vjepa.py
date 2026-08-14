#!/usr/bin/env python3
"""Specification for 35_vjepa (V-JEPA; Bardes et al., 2024; arXiv:2404.08471).

V-JEPA latent prediction: a context encoder sees the visible tokens, an EMA target
encoder encodes the whole input, and a predictor predicts the target's
representations at masked positions. This port covers the capture's step-2 image
adaptation (num_frames=1) -- a from-scratch comparable row, not the step-1 caveat
probe of the released video model. The ViT + predictor + 3D mask collator are the
official facebookresearch/jepa code, pinned as third_party/jepa and imported (never
copied); the src/app imports are lazy to stay in-process collision-free.

`encoder.pt` is the EMA target encoder (keys backbone.*; no separate head).
`linear_eval` probes its mean-pooled tokens. Licence: CC BY-NC 4.0 (research-use,
documented).
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
from _checkout import needs_checkout         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHOD = ROOT / "methods" / "35_vjepa"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "jepa"
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
    HAVE_DEPS, "35_vjepa needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("vjepa_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny vit_tiny at 64px (num_frames=1,
# patch16 -> a 4x4 = 16 patch grid so the block masks fit), a shallow predictor
# (pred_embed_dim 96, divisible by vit_tiny's 3 heads), 2 mask configs. The paper's
# vit_base / 224px / 300 epochs live in the shipped config.
MODEL = {"model_name": "vit_tiny", "crop_size": 64, "patch_size": 16,
         "num_frames": 1, "tubelet_size": 1, "pred_depth": 1,
         "pred_embed_dim": 96, "uniform_power": False, "use_mask_tokens": True,
         "zero_init_mask_tokens": True, "use_sdpa": True}
EMBED = 192   # vit_tiny embed dim
MASK = [{"aspect_ratio": [0.75, 1.5], "num_blocks": 4,
         "spatial_scale": [0.15, 0.2], "temporal_scale": [1.0, 1.0],
         "max_temporal_keep": 1.0, "max_keep": None},
        {"aspect_ratio": [0.75, 1.5], "num_blocks": 1,
         "spatial_scale": [0.5, 0.7], "temporal_scale": [1.0, 1.0],
         "max_temporal_keep": 1.0, "max_keep": None}]
DATA = {"num_workers": 0, "use_color_jitter": True}
LOSS = {"loss_exp": 1.0, "reg_coeff": 0.0}
TRAINING = {"epochs": 1, "batch_size": 2, "lr": 6.0e-4, "start_lr": 0.0,
            "final_lr": 1.0e-6, "weight_decay": 0.05, "final_weight_decay": 0.05,
            "warmup_epochs": 0, "beta1": 0.9, "beta2": 0.95, "eps": 1.0e-8,
            "clip_grad": 10.0, "ema_start": 0.996, "ema_final": 1.0,
            "save_at_epochs": []}
TRAIN = {**MODEL, **DATA, "mask": MASK, **LOSS, **TRAINING}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}


def _submodule_present() -> bool:
    return (UPSTREAM / "app" / "vjepa" / "utils.py").is_file()


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 256, (80, 80, 3), dtype="uint8")).save(
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
                base = np.full((80, 80, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (80, 80, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vjepa-"))
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


class TestThePinnedUpstream(unittest.TestCase):
    @needs_checkout
    def test_the_adapter_records_the_checked_out_commit(self):
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("the submodule is not checked out here")
        self.assertEqual(r.stdout.strip(), adapter.UPSTREAM["commit"])

    def test_provenance_agrees_and_records_non_commercial_licence(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(prov["upstream"]["commit"], adapter.UPSTREAM["commit"])
        self.assertIn("facebookresearch/jepa", adapter.UPSTREAM["repo"])
        blob = json.dumps(prov).lower()
        self.assertIn("cc by-nc", blob)
        self.assertIn("non-commercial", blob)


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("vjepa_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_encoder_mean_pool_is_one_feature_per_image(self):
        import torch
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        enc = self.models().build_vjepa_encoder(MODEL, torch.device("cpu"))
        feats = enc(torch.randn(2, 3, 64, 64)).mean(dim=1)
        self.assertEqual(tuple(feats.shape), (2, EMBED))

    @needs_deps
    def test_build_vjepa_returns_encoder_and_predictor(self):
        import torch
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        enc, pred = self.models().build_vjepa(MODEL, num_mask_tokens=len(MASK),
                                              device=torch.device("cpu"))
        self.assertIsNotNone(enc)
        self.assertIsNotNone(pred)


class TestTheEMA(unittest.TestCase):
    def trainer(self):
        return load("vjepa_trainer", METHOD / "train_pretrain_vjepa.py")

    @needs_deps
    def test_ema_moves_target_toward_source(self):
        import torch
        import torch.nn as nn
        t = self.trainer()
        target, source = nn.Linear(4, 4), nn.Linear(4, 4)
        for p in target.parameters():
            p.data.zero_()
        for p in source.parameters():
            p.data.fill_(10.0)
        t.update_ema(target, source, momentum=0.9)
        moved = next(target.parameters()).data.mean().item()
        self.assertGreater(moved, 0.5)
        self.assertLess(moved, 5.0)


class TestTheData(Base):
    def data_mod(self):
        return load("vjepa_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_loader_yields_images_and_masks(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        tiny_imagefolder(self.tmp / "data" / "train")
        loader, _ = self.data_mod().get_vjepa_dataloader(
            str(self.tmp / "data"), batch_size=2, cfgs_mask=MASK,
            crop_size=64, num_frames=1, patch_size=16, tubelet_size=1,
            num_workers=0, seed=0)
        images, labels, masks_enc, masks_pred = next(iter(loader))
        self.assertEqual(tuple(images.shape), (2, 3, 64, 64))
        self.assertEqual(len(masks_pred), len(MASK))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_the_whole_target_encoder_is_kept(self):
        got = adapter.extract_encoder({"backbone.pos_embed": 1,
                                       "backbone.blocks.0.norm1.weight": 2})
        self.assertEqual(set(got),
                         {"backbone.pos_embed", "backbone.blocks.0.norm1.weight"})

    def test_an_empty_state_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({})
        self.assertIn("empty", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["model_name"], "vit_tiny")
        self.assertEqual(len(built["mask"]), len(MASK))
        self.assertEqual(built["loss"]["loss_exp"], 1.0)
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
            adapter.to_run_config(self.config(train={"nonsense": 1}), out=self.out)
        self.assertIn("nonsense", str(e.exception))

    def test_an_empty_mask_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"mask": []}), out=self.out)
        self.assertIn("mask", str(e.exception))

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
        cfg = self.eval_config(train={"loss_exp": 1.0})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("loss_exp", str(e.exception))


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
        return load("vjepa_trainer", METHOD / "train_pretrain_vjepa.py")

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
        src = (METHOD / "train_pretrain_vjepa.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


class TestAStep1Smoke(Base):
    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data" / "train")
        c = self.config(**over)
        c["data_root"] = str(self.tmp / "data")
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(c), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return cfg, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_writes_an_encoder_and_a_pretext_loss(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        self.run_adapter()
        self.assertTrue((self.out / "encoder.pt").is_file())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m)

    @needs_deps
    def test_each_milestone_encoder_is_written(self):
        """save_at_epochs writes checkpoint_epoch_{N}.pth per milestone; the
        adapter hands over encoder_epoch{N}.pt (the target encoder) for each so
        the 100/200/300 sweep can probe every frozen milestone. This config is
        already the unified ViT-B/16 Step 2, so no recipe key is needed."""
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        _, r = self.run_adapter(train={"epochs": 2, "save_at_epochs": [1, 2]})
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((self.out / "encoder.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch1.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch2.pt").is_file())

    @needs_deps
    def test_the_encoder_pt_it_wrote_loads_back(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        self.run_adapter()
        import torch
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved)
        # Bind `models` to this method's before load_encoder imports it (the
        # in-process suite shares the `models` package name across methods).
        load("this_methods_models", METHOD / "models" / "__init__.py")
        model = adapter.load_encoder(saved, self.eval_config())
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
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    @needs_deps
    def test_the_manifest_records_the_pinned_upstream(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


class TestALinearEvalSmoke(Base):
    def _step1(self):
        tiny_split(self.tmp / "data")
        s1data = self.tmp / "s1data"
        tiny_imagefolder(s1data / "train")
        s1cfg = {"stage": "pretrain", "seed": 0,
                 "data_root": str(s1data), "device": "cpu",
                 "train": dict(TRAIN)}
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
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        cfg, r = self.run_eval()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_reports_the_comparable_probe_numbers(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        self.run_eval()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy"):
            self.assertIn(name, m)

    @needs_deps
    def test_it_produces_no_encoder_and_says_so(self):
        if not _submodule_present():
            self.skipTest("the jepa submodule is not checked out here")
        self.run_eval()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)


class TestTheOriginalIsReferencedNotCopied(unittest.TestCase):
    def test_no_distributed_or_tensorboard_machinery_is_used(self):
        import ast
        tree = ast.parse((METHOD / "train_pretrain_vjepa.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)


if __name__ == "__main__":
    unittest.main()
