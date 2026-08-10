#!/usr/bin/env python3
"""Specification for 33_pirl (Misra & van der Maaten, CVPR 2020).

PIRL: Pretext-Invariant Representation Learning. A ResNet-50 trunk produces a
representation of an image and of a jigsaw-shuffled view of the same image (nine
patches through the shared trunk, projected and concatenated). Both are contrasted
against a momentum-updated **memory bank** (one row per training image) with an
NCE cross-entropy; the loss is a convex combination of the image-NCE and the
jigsaw-NCE. The memory bank is updated with the image representation each step.

`encoder.pt` is the ResNet-50 trunk (`encoder.*`); the image/jigsaw projection
heads are excluded, and the memory bank lives in the loss module (a buffer), not
the model, so it is never in the model state. `linear_eval` probes the trunk
(2048-d). The captured step 2 (ViT) is excluded, as in every port.
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
METHOD = ROOT / "methods" / "33_pirl"
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
    HAVE_DEPS, "33_pirl needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("pirl_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: ResNet-50 at a 32px image and 16px jigsaw
# patches (3x3 grid = 9 patches), a narrow feature_dim, few negatives. The paper's
# 224px / feature_dim 128 / 32000 negatives / 800 epochs live in the shipped config.
IMG = 32
FEAT = 16
NUM_PATCHES = 9
PATCH = 16
BACKBONE_DIM = 2048  # ResNet-50 trunk feature

MODEL = {"arch": "resnet50", "feature_dim": FEAT, "num_patches": NUM_PATCHES}
DATA = {"image_size": IMG, "jigsaw_resize": 48, "jigsaw_crop_size": 48,
        "jigsaw_grid_size": 3, "jigsaw_patch_size": PATCH, "num_workers": 0}
NCE = {"temperature": 0.07, "nce_momentum": 0.5, "num_negatives": 4}
LOSS = {"jigsaw_weight": 0.5}
MEMORY = {"initialize_from_model": True}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "lr": 0.03, "momentum": 0.9,
              "weight_decay": 1.0e-4, "lr_milestones": [], "lr_gamma": 0.1,
              "warmup_epochs": 0}
TRAIN = {**MODEL, **DATA, **NCE, **LOSS, **MEMORY, **STEP1_ONLY}
EVAL_TRAIN = {"arch": "resnet50", "image_size": IMG, "epochs": 2,
              "batch_size": 2, "num_workers": 0, "lr": 0.1, "momentum": 0.9,
              "weight_decay": 0.0}


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
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pirl-"))
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
        return load("pirl_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.build_resnet_pirl(feature_dim=FEAT, num_patches=NUM_PATCHES)

    @needs_deps
    def test_forward_original_is_a_normalized_feature(self):
        import torch
        model = self._model(self.models())
        feats = model.forward_original(torch.randn(2, 3, IMG, IMG))
        self.assertEqual(tuple(feats.shape), (2, FEAT))
        self.assertTrue(torch.allclose(feats.norm(dim=1),
                                       torch.ones(2), atol=1e-4))

    @needs_deps
    def test_forward_jigsaw_is_a_normalized_feature(self):
        import torch
        model = self._model(self.models())
        patches = torch.randn(2, NUM_PATCHES, 3, PATCH, PATCH)
        feats = model.forward_jigsaw(patches)
        self.assertEqual(tuple(feats.shape), (2, FEAT))
        self.assertTrue(torch.allclose(feats.norm(dim=1),
                                       torch.ones(2), atol=1e-4))

    @needs_deps
    def test_forward_returns_both_views(self):
        import torch
        model = self._model(self.models())
        img_f, jig_f = model(torch.randn(2, 3, IMG, IMG),
                             torch.randn(2, NUM_PATCHES, 3, PATCH, PATCH))
        self.assertEqual(tuple(img_f.shape), (2, FEAT))
        self.assertEqual(tuple(jig_f.shape), (2, FEAT))

    @needs_deps
    def test_get_encoder_returns_the_backbone_feature(self):
        import torch
        model = self._model(self.models())
        feats = model.get_encoder()(torch.randn(2, 3, IMG, IMG))
        self.assertEqual(tuple(feats.shape), (2, BACKBONE_DIM))


class TestTheMemoryNCE(unittest.TestCase):
    def loss_mod(self):
        return load("pirl_loss", METHOD / "loss" / "__init__.py")

    def _bank(self, m, n=6):
        return m.PIRLMemoryBankNCE(num_samples=n, feature_dim=FEAT,
                                   temperature=NCE["temperature"],
                                   momentum=NCE["nce_momentum"],
                                   num_negatives=NCE["num_negatives"])

    @needs_deps
    def test_the_memory_bank_is_normalized(self):
        import torch
        bank = self._bank(self.loss_mod())
        self.assertEqual(tuple(bank.memory.shape), (6, FEAT))
        self.assertTrue(torch.allclose(bank.memory.norm(dim=1),
                                       torch.ones(6), atol=1e-4))

    @needs_deps
    def test_forward_returns_a_finite_scalar_loss(self):
        import torch
        import torch.nn.functional as F
        bank = self._bank(self.loss_mod())
        q = F.normalize(torch.randn(2, FEAT), dim=1)
        loss = bank(q, torch.tensor([0, 1]))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @needs_deps
    def test_update_memory_moves_the_indexed_rows_toward_the_features(self):
        import torch
        import torch.nn.functional as F
        bank = self._bank(self.loss_mod())
        idx = torch.tensor([0, 1])
        feats = F.normalize(torch.randn(2, FEAT), dim=1)
        before = bank.memory[idx].clone()
        bank.update_memory(feats, idx)
        after = bank.memory[idx]
        self.assertFalse(torch.allclose(before, after), "the bank did not move")
        expected = F.normalize(NCE["nce_momentum"] * before
                               + (1.0 - NCE["nce_momentum"]) * feats, dim=1)
        self.assertTrue(torch.allclose(after, expected, atol=1e-5))


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("pirl_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_image_patches_index_and_label(self):
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().ImageNetPIRLDataset(
            root=str(self.tmp / "data"), image_size=IMG, jigsaw_resize=48,
            jigsaw_crop_size=48, jigsaw_grid_size=3, jigsaw_patch_size=PATCH,
            train=True)
        image, patches, index, label = ds[0]
        self.assertEqual(tuple(image.shape), (3, IMG, IMG))
        self.assertEqual(tuple(patches.shape), (NUM_PATCHES, 3, PATCH, PATCH))
        self.assertEqual(index, 0)
        self.assertIsInstance(int(label), int)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_backbone_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.0.weight": 1, "encoder.4.0.conv1.weight": 2,
            "projector.weight": 3, "jigsaw_projector.weight": 4})
        self.assertEqual(set(got),
                         {"encoder.0.weight", "encoder.4.0.conv1.weight"})

    def test_the_projection_heads_are_left_out(self):
        got = adapter.extract_encoder({"encoder.7.2.bn3.weight": 1,
                                       "projector.bias": 2,
                                       "jigsaw_projector.bias": 3})
        self.assertEqual(set(got), {"encoder.7.2.bn3.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"projector.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["arch"], "resnet50")
        self.assertEqual(built["nce"]["temperature"], 0.07)
        self.assertEqual(built["nce"]["momentum"], 0.5)
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
        cfg = self.eval_config(train={"jigsaw_weight": 0.5})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("jigsaw_weight", str(e.exception))


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
        return load("pirl_trainer", METHOD / "train_step1_pirl.py")

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
        src = (METHOD / "train_step1_pirl.py").read_text()
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
        tree = ast.parse((METHOD / "train_step1_pirl.py").read_text())
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
