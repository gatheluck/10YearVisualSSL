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


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("cmc_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to train a step on a CPU: a narrow feat_dim, a 64px input (the
# fixed-6x6 fc6 is relaxed with an adaptive pool so a small smoke can run), a
# handful of negatives. The paper's 224px / feat_dim 128 / K 16384 live in the
# shipped config.
MODEL = {"feat_dim": 32, "img_size": 64}
NCE = {"temperature": 0.07, "nce_momentum": 0.5, "num_negatives": 4}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.03,
              "momentum": 0.9, "weight_decay": 1.0e-4,
              "lr_decay_epochs": [], "lr_decay_rate": 0.1, "crop_low": 0.2}
TRAIN = {**MODEL, **NCE, **STEP1_ONLY}
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
        return load("cmc_trainer", METHOD / "train_step1_cmc.py")

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
        src = (METHOD / "train_step1_cmc.py").read_text()
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
        tree = ast.parse((METHOD / "train_step1_cmc.py").read_text())
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
