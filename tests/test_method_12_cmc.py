#!/usr/bin/env python3
"""Specification for 12_cmc (Tian et al., 2019; arXiv:1906.05849).

Contrastive Multiview Coding, the **AlexNet** path. This port covers the lab's
paper-faithful step 1: an RGB image is converted to CIE **Lab** and split into
its L (1-channel) and ab (2-channel) views; a two-branch half-size AlexNet maps
each view to an L2-normalised 128-d embedding; an **NCE** loss over two momentum
**memory banks** (one per view, cross-view scored) pulls the two views of an
image together and apart from K negatives. `encoder.pt` is the two-branch
encoder; `linear_eval` probes the layer-6 features of both branches, concatenated
(the paper's best single-branch layer).

The Lab conversion is reimplemented in numpy (sRGB -> XYZ(D65) -> CIE Lab) so the
port keeps the torch-only closure -- scikit-image is not a dependency. The
captured ViT step 2 and the deprecated ResNet variant are excluded, as in every
port.
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
METHOD = ROOT / "methods" / "12_cmc"
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
    HAVE_DEPS, "12_cmc needs torch, numpy, torchvision")

try:
    import timm                                        # noqa: F401
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("cmc_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to train a step on a CPU: a narrow feat_dim, a 64px input (the
# fixed-6x6 fc6 is relaxed with an adaptive pool so a small smoke can run), a
# handful of negatives. The paper's 224px / feat_dim 128 / K 16384 live in the
# shipped config.
MODEL = {"feat_dim": 32, "img_size": 64}
NCE = {"temperature": 0.07, "nce_momentum": 0.5, "num_negatives": 4}
PRETRAIN_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.03,
              "momentum": 0.9, "weight_decay": 1.0e-4,
              "lr_decay_epochs": [], "lr_decay_rate": 0.1, "crop_low": 0.2}
TRAIN = {**MODEL, **NCE, **PRETRAIN_ONLY}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

# fc6 of each branch is Linear(128*6*6, 2048); the probe concatenates both
# branches at layer 6, so the comparable feature is 2 * 2048.
LAYER6_DIM = 2 * 2048


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "train" / "class0"
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
        self.tmp = Path(tempfile.mkdtemp(prefix="cmc-"))
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
    """The numpy Lab conversion must match published CIE Lab (D65, sRGB), so the
    port is faithful without scikit-image."""

    # skimage/CIE reference Lab for solid sRGB colors (D65, 2-degree observer).
    ORACLES = {
        (255, 255, 255): (100.00, 0.00, 0.00),
        (0, 0, 0): (0.00, 0.00, 0.00),
        (255, 0, 0): (53.24, 80.09, 67.20),
        (0, 255, 0): (87.74, -86.18, 83.18),
        (0, 0, 255): (32.30, 79.19, -107.86),
        (128, 128, 128): (53.59, 0.00, 0.00),
    }

    @needs_deps
    def test_solid_colors_match_cie_reference(self):
        import numpy as np
        from PIL import Image
        data = load("cmc_data", METHOD / "data" / "__init__.py")
        rgb2lab = data.RGB2Lab()
        for (r, g, b), (eL, ea, eb) in self.ORACLES.items():
            img = Image.fromarray(np.full((4, 4, 3), (r, g, b), dtype="uint8"))
            lab = np.asarray(rgb2lab(img))
            self.assertEqual(lab.shape, (4, 4, 3))
            got = lab.reshape(-1, 3).mean(axis=0)
            for got_c, exp_c, name in zip(got, (eL, ea, eb), "Lab"):
                self.assertAlmostEqual(
                    float(got_c), exp_c, delta=0.05,
                    msg=f"{name} of rgb({r},{g},{b}): {got_c} vs {exp_c}")


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("cmc_models", METHOD / "models" / "__init__.py")

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, MODEL["img_size"], MODEL["img_size"])

    @needs_deps
    def test_forward_returns_two_view_features(self):
        import torch
        m = self.models()
        model = m.build_cmc_from_config({"model": MODEL})
        model.train()
        feat_l, feat_ab = model(self._batch(torch), layer=8)
        self.assertEqual(tuple(feat_l.shape), (2, MODEL["feat_dim"]))
        self.assertEqual(tuple(feat_ab.shape), (2, MODEL["feat_dim"]))

    @needs_deps
    def test_layer8_is_l2_normalised(self):
        import torch
        m = self.models()
        model = m.build_cmc_from_config({"model": MODEL})
        model.eval()
        feat_l, feat_ab = model(self._batch(torch), layer=8)
        for feat in (feat_l, feat_ab):
            norms = feat.norm(dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms),
                                           atol=1e-4))

    @needs_deps
    def test_the_encoder_returns_one_feature_per_image(self):
        import torch
        m = self.models()
        model = m.build_cmc_from_config({"model": MODEL})
        enc = model.get_encoder()          # layer-6 concat of both branches
        enc.eval()
        feats = enc(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, LAYER6_DIM))


class TestTheNCE(unittest.TestCase):
    def nce_mod(self):
        return load("cmc_nce", METHOD / "nce" / "__init__.py")

    @needs_deps
    def test_two_memory_banks_are_registered(self):
        import torch
        n = self.nce_mod()
        contrast = n.NCEAverage(feat_dim=MODEL["feat_dim"], n_data=6,
                                K=NCE["num_negatives"], T=NCE["temperature"],
                                momentum=NCE["nce_momentum"])
        state = contrast.state_dict()
        self.assertIn("memory_l", state)
        self.assertIn("memory_ab", state)
        self.assertEqual(tuple(state["memory_l"].shape), (6, MODEL["feat_dim"]))
        self.assertEqual(tuple(state["memory_ab"].shape), (6, MODEL["feat_dim"]))

    @needs_deps
    def test_alias_sampler_draws_valid_indices(self):
        import torch
        n = self.nce_mod()
        sampler = n.AliasMethod(torch.ones(6))
        draw = sampler.draw(20)
        self.assertEqual(tuple(draw.shape), (20,))
        self.assertTrue(int(draw.min()) >= 0 and int(draw.max()) < 6)

    @needs_deps
    def test_nce_loss_is_a_finite_scalar(self):
        import torch
        n = self.nce_mod()
        contrast = n.NCEAverage(feat_dim=MODEL["feat_dim"], n_data=6,
                                K=NCE["num_negatives"], T=NCE["temperature"],
                                momentum=NCE["nce_momentum"])
        crit = n.NCECriterion(6)
        feat_l = torch.nn.functional.normalize(torch.randn(2, MODEL["feat_dim"]),
                                               dim=1)
        feat_ab = torch.nn.functional.normalize(torch.randn(2, MODEL["feat_dim"]),
                                                dim=1)
        y = torch.tensor([0, 1])
        out_l, out_ab = contrast(feat_l, feat_ab, y)
        loss = crit(out_l) + crit(out_ab)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @needs_deps
    def test_the_update_moves_both_indexed_banks(self):
        import torch
        n = self.nce_mod()
        contrast = n.NCEAverage(feat_dim=MODEL["feat_dim"], n_data=6,
                                K=NCE["num_negatives"], T=NCE["temperature"],
                                momentum=NCE["nce_momentum"])
        y = torch.tensor([0, 1])
        before_l = contrast.memory_l.index_select(0, y).clone()
        before_ab = contrast.memory_ab.index_select(0, y).clone()
        feat_l = torch.nn.functional.normalize(torch.randn(2, MODEL["feat_dim"]),
                                               dim=1)
        feat_ab = torch.nn.functional.normalize(torch.randn(2, MODEL["feat_dim"]),
                                                dim=1)
        contrast(feat_l, feat_ab, y)
        after_l = contrast.memory_l.index_select(0, y)
        after_ab = contrast.memory_ab.index_select(0, y)
        self.assertFalse(torch.allclose(before_l, after_l),
                         "the L bank row did not move")
        self.assertFalse(torch.allclose(before_ab, after_ab),
                         "the ab bank row did not move")


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("cmc_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_a_lab_image_a_label_and_an_index(self):
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().CMCDataset(
            str(self.tmp / "data"), mode="train",
            image_size=MODEL["img_size"], crop_low=0.2)
        item = ds[0]
        self.assertEqual(len(item), 3)
        image, label, index = item
        self.assertEqual(tuple(image.shape),
                         (3, MODEL["img_size"], MODEL["img_size"]))
        self.assertEqual(index, 0)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_two_branches_come_out(self):
        got = adapter.extract_encoder({
            "encoder_l.conv1.0.weight": 1, "encoder_ab.conv1.0.weight": 2,
            "memory_l": 3, "memory_ab": 4, "params": 5})
        self.assertEqual(set(got),
                         {"encoder_l.conv1.0.weight", "encoder_ab.conv1.0.weight"})

    def test_the_memory_banks_are_left_out(self):
        got = adapter.extract_encoder({"encoder_l.fc8.weight": 1,
                                       "memory_l": 2, "memory_ab": 3})
        self.assertEqual(set(got), {"encoder_l.fc8.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"memory_l": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["feat_dim"], 32)
        self.assertEqual(built["nce"]["num_negatives"], 4)
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

    def test_nce_settings_are_not_part_of_the_probe(self):
        # temperature is a step-1 NCE knob; the probe (SGD) must reject it.
        cfg = self.eval_config(train={"temperature": 0.07})
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
        return load("cmc_trainer", METHOD / "train_pretrain_cmc.py")

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
        src = (METHOD / "train_pretrain_cmc.py").read_text()
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
        tree = ast.parse((METHOD / "train_pretrain_cmc.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native
# two-branch AlexNet CMC pretrain. Two ViT backbones (L=1ch, ab=2ch via
# in_chans) + two 3-layer projectors; cross-view NCE over two memory banks; the
# probe concatenates both branches' CLS features (2 * embed_dim). Tiny dims for a
# CPU smoke (batch_size > 1 for the projector BatchNorm1d).
VIT_MODEL_ARGS = {"feat_dim": 8, "hidden_dim": 16, "image_size": 32,
                  "patch_size": 16, "embed_dim": 16, "depth": 1, "num_heads": 2,
                  "mlp_ratio": 4.0, "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_TRAIN_TINY = {"arch": "vit", "feat_dim": 8, "hidden_dim": 16, "img_size": 32,
                  "patch_size": 16, "embed_dim": 16, "depth": 1, "num_heads": 2,
                  "mlp_ratio": 4.0, "drop_rate": 0.0, "attn_drop_rate": 0.0,
                  "temperature": 0.07, "nce_momentum": 0.5, "num_negatives": 4,
                  "crop_low": 0.2, "epochs": 2, "batch_size": 2, "num_workers": 0,
                  "lr": 6.0e-4, "weight_decay": 0.05, "warmup_epochs": 0,
                  "min_lr": 0.0, "save_at_epochs": [1, 2]}
CONCAT_DIM = 2 * VIT_MODEL_ARGS["embed_dim"]


class TestVitConfigTranslation(Base):
    def vit_config(self, train=None, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": dict(train if train is not None else VIT_TRAIN_TINY)}
        for k, v in over.items():
            cfg[k] = v
        return cfg

    def test_the_vit_step2_config_is_accepted(self):
        built = adapter.to_run_config(self.vit_config(), out=self.out)
        self.assertEqual(built["arch"], "vit")
        self.assertEqual(built["model"]["embed_dim"], 16)
        self.assertEqual(built["model"]["hidden_dim"], 16)
        self.assertEqual(built["nce"]["num_negatives"], 4)
        self.assertEqual(built["training"]["save_at_epochs"], [1, 2])

    def test_native_path_unchanged_when_arch_absent(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertNotIn("arch", built)

    def test_a_bad_arch_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"arch": "resnext"}),
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
                self.vit_config(train={**VIT_TRAIN_TINY, "momentum": 0.9}),
                out=self.out)
        self.assertIn("momentum", str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_cmc", METHOD / "models" / "vit_cmc.py")
        return vm.build_vit_cmc(**VIT_MODEL_ARGS)

    def _lab(self, torch, b=2):
        return torch.randn(b, 3, VIT_MODEL_ARGS["image_size"],
                           VIT_MODEL_ARGS["image_size"])

    @needs_timm
    def test_get_encoder_concatenates_both_branch_cls_features(self):
        import torch
        feats = self._model().get_encoder()(self._lab(torch))
        self.assertEqual(tuple(feats.shape), (2, CONCAT_DIM))

    @needs_timm
    def test_forward_gives_two_normalised_view_embeddings(self):
        import torch
        feat_l, feat_ab = self._model()(self._lab(torch))
        for f in (feat_l, feat_ab):
            self.assertEqual(tuple(f.shape), (2, VIT_MODEL_ARGS["feat_dim"]))
            norms = f.norm(dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms),
                                           atol=1e-4))

    @needs_timm
    def test_get_features_returns_both_branch_cls(self):
        import torch
        cls_l, cls_ab = self._model().get_features(self._lab(torch))
        for c in (cls_l, cls_ab):
            self.assertEqual(tuple(c.shape), (2, VIT_MODEL_ARGS["embed_dim"]))

    @needs_timm
    def test_encoder_pt_holds_only_the_two_backbones(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith(("encoder_l.", "encoder_ab."))
                            for k in got))
        self.assertFalse([k for k in got if k.startswith("proj_")])

    @needs_timm
    def test_load_encoder_round_trips_the_two_backbones(self):
        import torch
        saved = adapter.extract_encoder(self._model().state_dict())
        cfg = {"train": {"arch": "vit", "feat_dim": 8, "hidden_dim": 16,
                         "img_size": 32, "patch_size": 16, "embed_dim": 16,
                         "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                         "drop_rate": 0.0, "attn_drop_rate": 0.0}}
        loaded = adapter.load_encoder(saved, cfg).state_dict()
        pairs = 0
        for k, want in saved.items():
            got = loaded.get(k)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{k} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


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
                "encoder": str(encoder),
                "train": {"arch": "vit", "feat_dim": 8, "hidden_dim": 16,
                          "img_size": 32, "patch_size": 16, "embed_dim": 16,
                          "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                          "drop_rate": 0.0, "attn_drop_rate": 0.0, "epochs": 1,
                          "batch_size": 2, "num_workers": 0, "lr": 0.1,
                          "momentum": 0.9, "weight_decay": 0.0}}

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
    two-branch AlexNet's concatenated layer-6 feature (2 x 2048 = 4096-d) --
    raw, before the probe's normalise -- one row per val image, with honest
    meta. CMC's eval pipeline is CIE Lab, not RGB; the provider states so.

    The encoder.pt is built from the *shipped* linear_eval config's
    architecture (the provider reads that config), via the same
    `extract_encoder` filter the adapter writes with; random weights do not
    affect the shape-and-plumbing this proves. Modules load through `load`
    (`load_from`), which purges any other method's `adapter`/`models` first --
    the whole suite runs many methods in one interpreter.
    """

    FEAT_DIM = 4096  # concatenated fc6 of the L and ab branches (2 x 2048)

    def _shipped_config(self) -> dict:
        import yaml
        return yaml.safe_load(
            (METHOD / "configs" / "linear_eval.yaml").read_text())

    def _make_encoder(self, cfg: dict) -> Path:
        import torch
        models = load("cmc_models", METHOD / "models" / "__init__.py")
        trainer = load("cmc_trainer", METHOD / "train_pretrain_cmc.py")
        model = models.build_cmc_from_config(trainer.model_config(cfg["train"]))
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("cmc_feature_provider", METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_4096d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("12_cmc provider not yet present")
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
        self.assertEqual(feats.shape[1], self.FEAT_DIM,
                         "CMC feature is the concatenated fc6 (2 x 2048)")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], self.FEAT_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("12_cmc provider not yet present")
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
        self.assertEqual(feats.shape, (6, self.FEAT_DIM))
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
            self.skipTest("12_cmc provider not yet present")
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
        self.assertEqual(feats.shape, (6, self.FEAT_DIM))


if __name__ == "__main__":
    unittest.main()
