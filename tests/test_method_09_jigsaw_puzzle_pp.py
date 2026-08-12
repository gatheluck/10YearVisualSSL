#!/usr/bin/env python3
"""Specification for 09_jigsaw_puzzle_pp (Noroozi et al., CVPR 2018; arXiv:1805.00385).

Jigsaw++ ("Boosting Self-Supervised Learning via Knowledge Transfer"). This port
covers **stage (a)**: the VGG16 jigsaw++ pretext task — an image is cut into a
3x3 grid, up to two tiles may be replaced with tiles from another image
(occlusions), the tiles are permuted by one of a fixed high-Hamming-distance
permutation set (701 in the paper), and a shared VGG16 encoder predicts which
permutation was applied. `encoder.pt` is that VGG16 encoder, and `linear_eval`
probes it.

The paper's **knowledge-transfer** stage is also ported (the `knowledge_transfer`
stage): cluster the VGG16 conv4 features with faiss k-means into pseudo-labels,
then train a standard AlexNet to classify them; `linear_eval arch=
alexnet_cluster_cls` probes that AlexNet. The clustering uses faiss, so that stage
is GPU / x86_64-linux only (faiss lives in the CUDA lock; step 1 + the default
`arch=vgg16` probe stay torch-only). The captured step 2 (ViT) is excluded, as in
every port.
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
METHOD = ROOT / "methods" / "09_jigsaw_puzzle_pp"
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
    HAVE_DEPS, "09_jigsaw_puzzle_pp needs torch, numpy, torchvision")

try:
    import faiss                                       # noqa: F401
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

needs_faiss = unittest.skipUnless(
    HAVE_DEPS and HAVE_FAISS,
    "the knowledge-transfer clustering needs faiss (GPU / x86_64-linux only)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("jigsaw_pp_adapter", METHOD / "adapter" / "__init__.py")

# A model + puzzle small enough to train a step on a CPU: 4 permutations and a
# 32px tile (VGG16's adaptive max-pool accepts any size, so the smoke stays
# cheap). The 3x3 grid is 3*32 + 2*0 = 96, so image_size must be >= 96.
MODEL = {"num_permutations": 4, "dropout": 0.0, "tile_size": 32, "tile_gap": 0,
         "image_size": 112, "grayscale_prob": 0.5, "max_occlusions": 2}
TRAIN = {**MODEL, "epochs": 1, "batch_size": 2, "num_workers": 0,
         "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0}
FEATURE_DIM = 1024

# Knowledge-transfer stage: cluster VGG16 conv4 features (k=4 for the smoke) and
# train an AlexNet. The paper's k=10000/2000 lives in the shipped config.
KT_TRAIN = {"num_clusters": 4, "image_size": 64, "dropout": 0.0,
            "epochs": 1, "batch_size": 2, "num_workers": 0,
            "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}
KT_EVAL_TRAIN = {"arch": "alexnet_cluster_cls", "dropout": 0.0, "image_size": 64,
                 "epochs": 2, "batch_size": 2, "num_workers": 0,
                 "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}
ALEXNET_FEATURE_DIM = 256 * 6 * 6  # AlexNet features + avgpool -> 9216


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "train" / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 256, (128, 128, 3), dtype="uint8")).save(
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
                base = np.full((128, 128, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (128, 128, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="jigpp-"))
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

    def kt_config(self, **over) -> dict:
        cfg = {"stage": "knowledge_transfer", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "encoder": str(self.tmp / "encoder.pt"), "train": dict(KT_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg

    def kt_eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "encoder": str(self.tmp / "encoder.pt"),
               "train": dict(KT_EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("jigsaw_pp_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_forward_predicts_over_the_permutation_set(self):
        import torch
        m = self.models()
        model = m.build_vgg16_jigsaw_pp_model(num_classes=4, dropout=0.0)
        tiles = torch.zeros(2, 9, 3, MODEL["tile_size"], MODEL["tile_size"])
        self.assertEqual(tuple(model(tiles).shape), (2, 4))

    @needs_deps
    def test_the_encoder_returns_one_feature_vector_per_image(self):
        import torch
        m = self.models()
        enc = m.build_vgg16_jigsaw_pp_model(num_classes=4).get_encoder()
        feats = enc(torch.zeros(2, 3, MODEL["tile_size"], MODEL["tile_size"]))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))

    @needs_deps
    def test_the_encoder_accepts_a_larger_tile(self):
        import torch
        m = self.models()
        enc = m.build_vgg16_jigsaw_pp_model(num_classes=4).get_encoder()
        feats = enc(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("jigsaw_pp_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_nine_tiles_and_a_permutation_label(self):
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().JigsawPPDataset(
            str(self.tmp / "data"), num_permutations=4,
            tile_size=MODEL["tile_size"], tile_gap=MODEL["tile_gap"],
            image_size=MODEL["image_size"])
        tiles, label = ds[0]
        self.assertEqual(tuple(tiles.shape),
                         (9, 3, MODEL["tile_size"], MODEL["tile_size"]))
        self.assertTrue(0 <= int(label) < 4)

    @needs_deps
    def test_the_permutation_set_is_deterministic_and_high_hamming(self):
        dm = self.dataset_mod()
        # 100, not a handful: two random 9-permutations rarely land within
        # Hamming < 3, so a small set can satisfy the floor by chance even with
        # no filter. At 100 the unfiltered first-N (seed 42) deterministically
        # contains a Hamming-2 pair, so this bites the dropped-filter mutant.
        a = dm.generate_jigsaw_pp_permutations(target_count=100, seed=42)
        b = dm.generate_jigsaw_pp_permutations(target_count=100, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 100)
        # Every pair is at least Hamming distance 3 apart (the paper's floor).
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                d = sum(x != y for x, y in zip(a[i], a[j]))
                self.assertGreaterEqual(d, 3)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_encoder_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.block1.0.weight": 1, "encoder.fc.1.weight": 2,
            "classifier.1.weight": 3, "classifier.4.weight": 4})
        self.assertEqual(set(got),
                         {"encoder.block1.0.weight", "encoder.fc.1.weight"})

    def test_the_classifier_is_left_out(self):
        got = adapter.extract_encoder({"encoder.block1.0.weight": 1,
                                       "classifier.1.weight": 2})
        self.assertNotIn("classifier.1.weight", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"classifier.1.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["num_permutations"], 4)
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
        return load("jigsaw_pp_trainer", METHOD / "train_step1_jigsaw_pp.py")

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
        src = (METHOD / "train_step1_jigsaw_pp.py").read_text()
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


# --- the knowledge-transfer stage -------------------------------------------
# Cluster the VGG16 conv4 features (faiss k-means) into pseudo-labels and train a
# standard AlexNet on them; `linear_eval arch=alexnet_cluster_cls` probes it.


class TestTheAlexNetModel(unittest.TestCase):
    def models(self):
        return load("jigsaw_pp_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_forward_predicts_over_the_clusters(self):
        import torch
        model = self.models().build_alexnet_cluster_cls_model(
            num_classes=7, dropout=0.0)
        self.assertEqual(tuple(model(torch.zeros(2, 3, 64, 64)).shape), (2, 7))

    @needs_deps
    def test_the_encoder_returns_one_feature_vector_per_image(self):
        import torch
        enc = self.models().build_alexnet_cluster_cls_model(
            num_classes=7).get_encoder()
        feats = enc(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(feats.shape), (2, ALEXNET_FEATURE_DIM))


class TestTheKTClustering(unittest.TestCase):
    def clustering(self):
        return load("jigsaw_pp_clustering", METHOD / "utils" / "clustering.py")

    @needs_deps
    def test_conv4_features_have_one_row_per_image(self):
        import torch
        models = self.clustering() and load(
            "jigsaw_pp_models", METHOD / "models" / "__init__.py")
        enc = models.build_vgg16_jigsaw_pp_model(num_classes=4).encoder
        loader = [(torch.zeros(5, 3, 64, 64), torch.zeros(5, dtype=torch.long))]
        feats = self.clustering().extract_conv4_features(
            enc, loader, torch.device("cpu"))
        self.assertEqual(feats.shape, (5, 8192))

    @needs_faiss
    def test_kmeans_is_reproducible_and_well_shaped(self):
        import numpy as np
        c = self.clustering()
        rng = np.random.RandomState(0)
        feats = rng.randn(40, 8192).astype("float32")
        a1, cent1 = c.run_kmeans(feats.copy(), num_clusters=4, seed=42,
                                 use_gpu=False, verbose=False)
        a2, _ = c.run_kmeans(feats.copy(), num_clusters=4, seed=42,
                             use_gpu=False, verbose=False)
        self.assertEqual(a1.shape, (40,))
        self.assertEqual(cent1.shape, (4, 8192))
        self.assertTrue((a1 == a2).all(), "same seed gave different clusters")
        self.assertTrue(0 <= int(a1.min()) and int(a1.max()) < 4)

    @needs_faiss
    def test_missing_faiss_is_refused_loudly(self):
        # The refusal path exists and names faiss; the capture rejects a CPU
        # fallback as inconsistent, so this must error, never silently degrade.
        import numpy as np
        from unittest import mock
        c = self.clustering()
        with mock.patch.object(c, "HAS_FAISS", False):
            with self.assertRaises(ImportError) as e:
                c.run_kmeans(np.zeros((4, 8), dtype="float32"), num_clusters=2,
                             seed=0, use_gpu=False, verbose=False)
            self.assertIn("faiss", str(e.exception).lower())


class TestKTConfigTranslation(Base):
    def test_kt_reaches_the_run_config(self):
        built = adapter.to_run_config(self.kt_config(), out=self.out)
        self.assertEqual(built["kt"]["num_clusters"], 4)
        self.assertEqual(built["training"]["epochs"], 1)

    def test_a_missing_kt_setting_is_refused_by_name(self):
        for key in KT_TRAIN:
            with self.subTest(key=key):
                cfg = self.kt_config()
                cfg["train"] = {k: v for k, v in KT_TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_kt_setting_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.kt_config(train={"nonsense": 1}),
                                  out=self.out)
        self.assertIn("nonsense", str(e.exception))

    def test_kt_requires_an_encoder(self):
        cfg = self.kt_config()
        del cfg["encoder"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("encoder", str(e.exception))


class TestKTEvalConfig(Base):
    def test_the_alexnet_arch_is_accepted(self):
        adapter.to_run_config(self.kt_eval_config(), out=self.out)

    def test_the_default_arch_is_vgg_and_kt_is_alexnet(self):
        self.assertEqual(adapter.eval_arch(EVAL_TRAIN), "vgg16")
        self.assertEqual(adapter.eval_arch(KT_EVAL_TRAIN),
                         "alexnet_cluster_cls")

    def test_an_unknown_arch_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.kt_eval_config(train={"arch": "resnet"}),
                                  out=self.out)
        self.assertIn("arch", str(e.exception))

    def test_a_vgg_probe_key_is_refused_for_the_alexnet_arch(self):
        cfg = self.kt_eval_config()
        cfg["train"]["tile_size"] = 32          # belongs to the vgg16 probe
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("tile_size", str(e.exception))


class TestExtractingTheAlexNetEncoder(unittest.TestCase):
    def test_only_the_conv_trunk_comes_out(self):
        got = adapter.extract_encoder(
            {"features.0.weight": 1, "features.3.bias": 2,
             "classifier.1.weight": 3, "classifier.6.weight": 4},
            adapter.ALEXNET_ENCODER_PREFIXES)
        self.assertEqual(set(got), {"features.0.weight", "features.3.bias"})

    def test_the_head_is_left_out(self):
        got = adapter.extract_encoder(
            {"features.0.weight": 1, "classifier.6.weight": 2},
            adapter.ALEXNET_ENCODER_PREFIXES)
        self.assertNotIn("classifier.6.weight", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError):
            adapter.extract_encoder({"classifier.6.weight": 1},
                                    adapter.ALEXNET_ENCODER_PREFIXES)


class TestAKnowledgeTransferSmoke(Base):
    def _adapter(self, cfg_dict, out):
        p = self.tmp / f"cfg_{out.name}.json"
        p.write_text(json.dumps(cfg_dict), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        return p, r

    def _step1_encoder(self):
        s1data = self.tmp / "s1data"
        tiny_imagefolder(s1data)
        s1cfg = {"stage": "pretrain", "seed": 0, "data_root": str(s1data),
                 "device": "cpu", "train": dict(TRAIN)}
        _, r = self._adapter(s1cfg, self.tmp / "s1out")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return self.tmp / "s1out" / "encoder.pt"

    def _kt_run(self):
        enc = self._step1_encoder()
        ktdata = self.tmp / "ktdata"
        tiny_imagefolder(ktdata)
        cfg = self.kt_config(data_root=str(ktdata), encoder=str(enc))
        p, r = self._adapter(cfg, self.tmp / "ktout")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        return p, self.tmp / "ktout"

    @needs_faiss
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, ktout = self._kt_run()
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(ktout), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_faiss
    def test_it_writes_an_alexnet_encoder_that_loads_back(self):
        _, ktout = self._kt_run()
        import torch
        self.assertTrue((ktout / "encoder.pt").is_file())
        saved = torch.load(ktout / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved)
        self.assertTrue(all(k.startswith("features.") for k in saved),
                        "encoder.pt should hold only the AlexNet conv trunk")
        model = adapter.load_encoder(saved, self.kt_eval_config())
        loaded = model.state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the AlexNet")

    @needs_faiss
    def test_the_alexnet_probe_reports_comparable_numbers(self):
        _, ktout = self._kt_run()
        split = self.tmp / "probe"
        tiny_split(split)
        cfg = self.kt_eval_config(data_root=str(split),
                                  encoder=str(ktout / "encoder.pt"))
        p, r = self._adapter(cfg, self.tmp / "probeout")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        m = json.loads(
            (self.tmp / "probeout" / "metrics.json").read_text())["metrics"]
        self.assertIn("best_linear_probe_top1_accuracy", m)
        self.assertFalse((self.tmp / "probeout" / "encoder.pt").exists())


class TestTheOriginalIsReferencedNotCopied(unittest.TestCase):
    def test_no_distributed_or_tensorboard_machinery_is_used(self):
        import ast
        tree = ast.parse((METHOD / "train_step1_jigsaw_pp.py").read_text())
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
