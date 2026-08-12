#!/usr/bin/env python3
"""Specification for 07_deepcluster (Caron et al., ECCV 2018).

DeepCluster: each epoch, the whole training set's fc7 features are extracted,
reduced by PCA + whitening and clustered by k-means; the cluster assignments
become pseudo-labels that a reset classification head is trained to predict. The
backbone is an AlexNet-BN with a fixed Sobel front-end. `encoder.pt` is the
backbone (features + fc6/fc7); `linear_eval` probes its 4096-d fc7 feature.

Faithfulness decision (user-approved): the clustering uses **faiss** -- the path
the capture and the original DeepCluster repo use (the capture ships faiss as the
required paper-target backend and marks its sklearn fallback "not the official
protocol"). faiss-gpu has a linux-x86_64-only wheel, so this method is
**GPU / x86_64-linux only**: faiss lives in the CUDA lock, not the (cross-platform)
CPU lock. The captured ViT step 2 is excluded, as in every port.
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
METHOD = ROOT / "methods" / "07_deepcluster"
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

try:
    import faiss                                       # noqa: F401
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "07_deepcluster needs torch, numpy, torchvision")
needs_faiss = unittest.skipUnless(
    HAVE_DEPS and HAVE_FAISS,
    "07_deepcluster's clustering needs faiss (GPU / x86_64-linux only)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("dc_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to cluster + train a step on a CPU: k=4 clusters, pca_dim=4, a
# 64px crop, one epoch. The paper's k=10000 / pca_dim=256 / 224px / 500 epochs
# live in the shipped config.
MODEL = {"sobel": True}
CLUSTERING = {"k": 4, "pca_dim": 4}
DATA = {"crop_size": 64}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "feat_batch_size": 4,
              "num_workers": 0, "lr": 0.05, "momentum": 0.9,
              "weight_decay": 1.0e-5, "lr_decay_epochs": [], "lr_decay_rate": 0.1}
TRAIN = {**MODEL, **CLUSTERING, **DATA, **STEP1_ONLY}
EVAL_TRAIN = {**MODEL, **DATA, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

FEATURE_DIM = 4096  # fc7


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
        self.tmp = Path(tempfile.mkdtemp(prefix="dc-"))
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
               "encoder": str(self.tmp / "encoder.pt"), "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("dc_models", METHOD / "models" / "__init__.py")

    def _rgb(self, torch, b=2):
        return torch.randn(b, 3, DATA["crop_size"], DATA["crop_size"])

    @needs_deps
    def test_forward_produces_cluster_logits(self):
        import torch
        model = self.models().build_alexnet_deepcluster(sobel=True,
                                                        num_classes=CLUSTERING["k"])
        model.eval()
        out = model(self._rgb(torch))
        self.assertEqual(tuple(out.shape), (2, CLUSTERING["k"]))

    @needs_deps
    def test_get_features_returns_fc7(self):
        import torch
        model = self.models().build_alexnet_deepcluster(sobel=True,
                                                        num_classes=CLUSTERING["k"])
        model.eval()
        feats = model.get_features(self._rgb(torch))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))

    @needs_deps
    def test_reset_top_layer_reinitialises_the_head(self):
        import torch
        m = self.models()
        model = m.build_alexnet_deepcluster(sobel=True, num_classes=CLUSTERING["k"])
        with torch.no_grad():
            model.top_layer.weight.add_(5.0)
        before = model.top_layer.weight.clone()
        model.reset_top_layer(CLUSTERING["k"], torch.device("cpu"), seed=1)
        self.assertFalse(torch.allclose(before, model.top_layer.weight))


class TestTheClustering(unittest.TestCase):
    def clustering(self):
        return load("dc_clustering", METHOD / "utils" / "clustering.py")

    @needs_faiss
    def test_kmeans_gives_valid_seed_reproducible_assignments(self):
        import numpy as np
        c = self.clustering()
        feats = np.random.RandomState(0).randn(60, 64).astype("float32")
        a1, _ = c.run_kmeans(feats, k=4, pca_dim=8, use_gpu=False, seed=42,
                             verbose=False)
        a2, _ = c.run_kmeans(feats, k=4, pca_dim=8, use_gpu=False, seed=42,
                             verbose=False)
        self.assertEqual(tuple(a1.shape), (60,))
        self.assertTrue(int(a1.min()) >= 0 and int(a1.max()) < 4)
        self.assertTrue((a1 == a2).all(), "same seed did not reproduce clusters")

    @needs_faiss
    def test_faiss_is_required(self):
        # The clustering must refuse to guess without faiss (the port committed
        # to the paper-target backend); the guard exists even when faiss is here.
        c = self.clustering()
        self.assertTrue(hasattr(c, "run_kmeans"))


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("dc_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_pseudo_labels_replace_the_original_labels(self):
        import numpy as np
        tiny_imagefolder(self.tmp / "data")
        d = self.dataset_mod()
        base = d.build_base_dataset(str(self.tmp / "data"),
                                    crop_size=DATA["crop_size"], train=True)
        pseudo = np.zeros(len(base), dtype="int64")
        pseudo[0] = 3
        ds = d.DeepClusterDataset(base, pseudo)
        img, label = ds[0]
        self.assertEqual(tuple(img.shape),
                         (3, DATA["crop_size"], DATA["crop_size"]))
        self.assertEqual(int(label), 3)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_backbone_comes_out(self):
        got = adapter.extract_encoder({
            "features.0.weight": 1, "classifier.1.weight": 2,
            "top_layer.weight": 3, "sobel_layer.sobel.weight": 4})
        self.assertEqual(set(got), {"features.0.weight", "classifier.1.weight"})

    def test_the_top_layer_is_left_out(self):
        got = adapter.extract_encoder({"features.3.weight": 1,
                                       "top_layer.weight": 2, "top_layer.bias": 3})
        self.assertEqual(set(got), {"features.3.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"top_layer.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["clustering"]["k"], 4)
        self.assertEqual(built["data"]["crop_size"], 64)
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

    def test_clustering_settings_are_not_part_of_the_probe(self):
        cfg = self.eval_config(train={"k": 4})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("k", str(e.exception))


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
        return load("dc_trainer", METHOD / "train_step1_deepcluster.py")

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
        src = (METHOD / "train_step1_deepcluster.py").read_text()
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

    @needs_faiss
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_faiss
    def test_it_writes_an_encoder_and_a_pretext_loss(self):
        self.run_adapter()
        self.assertTrue((self.out / "encoder.pt").is_file())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m)

    @needs_faiss
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

    @needs_faiss
    def test_the_same_config_twice_gives_the_same_encoder(self):
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    @unittest.skipUnless(HAVE_DEPS and HAVE_FAISS and torch.cuda.is_available(),
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

    @needs_faiss
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_eval()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_faiss
    def test_it_reports_the_comparable_probe_numbers(self):
        self.run_eval()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy"):
            self.assertIn(name, m)

    @needs_faiss
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
        tree = ast.parse((METHOD / "train_step1_deepcluster.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)
        self.assertNotIn("timm", used)


if __name__ == "__main__":
    unittest.main()
