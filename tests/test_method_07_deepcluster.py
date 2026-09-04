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
CPU lock. The unified ViT-B/16 Step 2 (arch: vit) is also ported additively --
the same faiss k-means self-labelling on a from-scratch ViT-B/16, AdamW/cosine,
milestone checkpoints -- so its smoke is likewise gated on faiss (and timm).
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

try:
    import timm                                          # noqa: F401
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "07_deepcluster needs torch, numpy, torchvision")
needs_faiss = unittest.skipUnless(
    HAVE_DEPS and HAVE_FAISS,
    "07_deepcluster's clustering needs faiss (GPU / x86_64-linux only)")
needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("dc_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to cluster + train a step on a CPU: k=4 clusters, pca_dim=4, a
# 64px crop, one epoch. The paper's k=10000 / pca_dim=256 / 224px / 500 epochs
# live in the shipped config.
MODEL = {"sobel": True}
CLUSTERING = {"k": 4, "pca_dim": 4}
DATA = {"crop_size": 64}
PRETRAIN_ONLY = {"epochs": 1, "batch_size": 2, "feat_batch_size": 4,
              "num_workers": 0, "lr": 0.05, "momentum": 0.9,
              "weight_decay": 1.0e-5, "lr_decay_epochs": [], "lr_decay_rate": 0.1}
TRAIN = {**MODEL, **CLUSTERING, **DATA, **PRETRAIN_ONLY}
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
        return load("dc_trainer", METHOD / "train_pretrain_deepcluster.py")

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
        src = (METHOD / "train_pretrain_deepcluster.py").read_text()
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
        tree = ast.parse((METHOD / "train_pretrain_deepcluster.py").read_text())
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
# AlexNet-BN path. One from-scratch ViT-B/16 backbone + a reset-each-epoch head;
# the same per-epoch faiss k-means self-labelling, the unified AdamW/cosine
# recipe with milestone checkpoints. Tiny dims so a CPU smoke is cheap.
VIT_MODEL_ARGS = {"num_classes": 8, "image_size": 32, "patch_size": 16,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_MODEL_KNOBS = {"image_size": 32, "patch_size": 16, "embed_dim": 16,
                   "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                   "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_TRAIN_TINY = {"arch": "vit", **VIT_MODEL_KNOBS, "k": 4, "pca_dim": 4,
                  "epochs": 2, "batch_size": 2, "feat_batch_size": 4,
                  "num_workers": 0, "lr": 6.0e-4, "weight_decay": 0.05,
                  "warmup_epochs": 0, "min_lr": 0.0, "save_at_epochs": [1, 2]}
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
        self.assertEqual(built["clustering"]["k"], 4)
        self.assertEqual(built["clustering"]["pca_dim"], 4)
        self.assertEqual(built["training"]["save_at_epochs"], [1, 2])

    def test_the_native_path_has_no_top_level_arch(self):
        built = adapter.to_run_config(self.config(), self.out)
        self.assertNotIn("arch", built)

    def test_a_bad_arch_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.vit_config(train={**VIT_TRAIN_TINY, "arch": "alexnext"}),
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
        # momentum / lr_decay_rate belong to the SGD native path, not AdamW.
        for key in ("momentum", "lr_decay_rate", "sobel", "crop_size"):
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
                    adapter.to_run_config(
                        self.config(train={key: 1}), self.out)
                self.assertIn(key, str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_deepcluster", METHOD / "models" / "vit_deepcluster.py")
        return vm.build_vit_deepcluster(**VIT_MODEL_ARGS)

    @needs_timm
    def test_get_features_returns_the_cls_feature(self):
        import torch
        feats = self._model().get_features(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM_VIT))

    @needs_timm
    def test_get_features_accepts_the_before_final_relu_flag(self):
        # Signature parity with the AlexNet path, so the shared
        # extract_features_for_clustering reuses it; the flag is ignored.
        import torch
        m = self._model()
        x = torch.randn(2, 3, 32, 32)
        a = m.get_features(x, before_final_relu=True)
        b = m.get_features(x, before_final_relu=False)
        self.assertTrue(torch.equal(a, b))

    @needs_timm
    def test_forward_scores_every_image(self):
        import torch
        logits = self._model()(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, VIT_MODEL_ARGS["num_classes"]))

    @needs_timm
    def test_reset_top_layer_keeps_shape_and_reinitialises(self):
        import torch
        m = self._model()
        before = m.top_layer.weight.detach().clone()
        with torch.no_grad():
            m.top_layer.weight.add_(1.0)
        m.reset_top_layer(VIT_MODEL_ARGS["num_classes"], torch.device("cpu"),
                          seed=0)
        self.assertEqual(m.top_layer.out_features, VIT_MODEL_ARGS["num_classes"])
        self.assertFalse(torch.equal(m.top_layer.weight.detach(),
                                     before + 1.0))

    @needs_timm
    def test_encoder_pt_holds_only_the_backbone(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith("backbone.") for k in got))
        self.assertFalse([k for k in got if k.startswith("top_layer")])

    @needs_timm
    def test_load_encoder_round_trips_the_backbone(self):
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
        self.assertGreater(pairs, 0, "no saved weight reached the backbone")


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

    @needs_faiss
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
    4096-d AlexNet-BN fc7 backbone feature -- raw, before the probe's normalise
    -- one row per val image, with honest meta.

    Unlike the MoCo-style methods, the eval main feeds the loaded model
    directly to `extract_features` (no `get_encoder()`), and the provider
    mirrors that. The encoder.pt is built from the *shipped* linear_eval
    config's architecture (the provider reads that config), via the same
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
        models = load("dc_models", METHOD / "models" / "__init__.py")
        trainer = load("dc_trainer",
                       METHOD / "train_pretrain_deepcluster.py")
        model = models.build_alexnet_deepcluster(
            **trainer.model_config(cfg["train"]))
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("dc_feature_provider", METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_4096d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("07_deepcluster provider not yet present")
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
        self.assertEqual(feats.shape[1], FEATURE_DIM,
                         "AlexNet-BN fc7 feature is 4096-d")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], FEATURE_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("07_deepcluster provider not yet present")
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
        self.assertEqual(feats.shape, (6, FEATURE_DIM))
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
            self.skipTest("07_deepcluster provider not yet present")
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
        self.assertEqual(feats.shape, (6, FEATURE_DIM))


if __name__ == "__main__":
    unittest.main()
