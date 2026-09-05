#!/usr/bin/env python3
"""Specification for 18_sela (Asano et al., ICLR 2020; arXiv:1911.05371).

SeLa (Self-Labelling via simultaneous clustering and representation learning),
the **ResNet** path. A ResNet backbone with `num_heads` linear prototype heads is
trained to predict pseudo-labels; the labels are (re)computed with **Sinkhorn-Knopp
optimal transport**, which forces a balanced (equipartitioned) assignment over K
clusters, and the network is trained with cross-entropy on the resulting hard
targets, averaged over the heads. Unlike DeepCluster, the heads are **not reset**
each epoch, and there is no Sobel front-end.

`encoder.pt` is the ResNet backbone (`backbone.*`); the prototype heads
(`top_layer.*`) are training machinery and are excluded. `linear_eval` probes the
backbone's 2048-d feature. The captured step 2 (ViT, which needs `timm`) is
excluded, as in every port.
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
METHOD = ROOT / "methods" / "18_sela"
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
    HAVE_DEPS, "18_sela needs torch, numpy, torchvision")

try:
    import timm                                        # noqa: F401
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("sela_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny K, two heads, a 32px input (the
# ResNet downsamples to 1x1, still valid), a short Sinkhorn schedule. The paper's
# K=3000 / 10 heads / 400 epochs / lambda 25 live in the shipped config.
MODEL = {"arch": "resnetv2", "image_size": 32}
CLUSTERING = {"k": 8, "num_heads": 2, "nopts": 2, "sinkhorn_max_iters": 50,
              "sinkhorn_tol": 0.1, "lambda": 25, "epsilon": 0.04}
PRETRAIN_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0,
              "learning_rate": 0.08, "momentum": 0.9, "weight_decay": 1.0e-5,
              "lr_schedule": "step", "lr_step_size": 150, "lr_gamma": 0.1,
              "temperature": 1.0, "assignment_mode": "hard"}
TRAIN = {**MODEL, **CLUSTERING, **PRETRAIN_ONLY}
EVAL_TRAIN = {"arch": "resnetv2", "image_size": 32, "epochs": 2, "batch_size": 2,
              "num_workers": 0, "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

FEATURE_DIM = 2048


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
        self.tmp = Path(tempfile.mkdtemp(prefix="sela-"))
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


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("sela_models", METHOD / "models" / "__init__.py")

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, MODEL["image_size"], MODEL["image_size"])

    @needs_deps
    def test_forward_returns_logits_per_head(self):
        import torch
        m = self.models()
        model = m.ResNetSeLa(num_classes=CLUSTERING["k"],
                             num_heads=CLUSTERING["num_heads"],
                             arch=MODEL["arch"])
        out = model(self._batch(torch))
        self.assertEqual(tuple(out.shape),
                         (2, CLUSTERING["num_heads"], CLUSTERING["k"]))

    @needs_deps
    def test_a_single_head_returns_a_flat_logit_matrix(self):
        import torch
        m = self.models()
        model = m.ResNetSeLa(num_classes=CLUSTERING["k"], num_heads=1,
                             arch=MODEL["arch"])
        out = model(self._batch(torch))
        self.assertEqual(tuple(out.shape), (2, CLUSTERING["k"]))

    @needs_deps
    def test_get_features_returns_the_backbone_feature(self):
        import torch
        m = self.models()
        model = m.ResNetSeLa(num_classes=CLUSTERING["k"],
                             num_heads=CLUSTERING["num_heads"],
                             arch=MODEL["arch"])
        feats = model.get_features(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))


class TestTheSinkhorn(unittest.TestCase):
    def sinkhorn(self):
        return load("sela_sinkhorn", METHOD / "utils" / "__init__.py")

    @needs_deps
    def test_it_produces_a_balanced_assignment(self):
        # Sinkhorn-Knopp enforces equipartition: with N samples and K clusters
        # the column masses are all ~N/K (up to tolerance). A plain argmax over
        # random scores would not be balanced.
        import torch
        sk = self.sinkhorn()
        torch.manual_seed(0)
        N, K = 60, 6
        scores = torch.randn(N, K)
        Q, info = sk.sinkhorn_knopp(scores, n_iters=200, temperature=1.0,
                                    lamb=1.0, tol=1e-3)
        self.assertEqual(tuple(Q.shape), (N, K))
        col = Q.sum(dim=0)                       # column masses
        self.assertTrue(torch.allclose(col, col.mean().expand_as(col),
                                       atol=1e-3),
                        "the assignment is not equipartitioned")

    @needs_deps
    def test_hard_assignments_are_valid_labels_and_deterministic(self):
        import torch
        sk = self.sinkhorn()
        models = load("sela_models", METHOD / "models" / "__init__.py")
        model = models.ResNetSeLa(num_classes=CLUSTERING["k"],
                                  num_heads=CLUSTERING["num_heads"],
                                  arch=MODEL["arch"])
        model.eval()
        n = 6
        imgs = torch.randn(n, 3, MODEL["image_size"], MODEL["image_size"])

        class _FakeLoader(list):
            """A loader stand-in: an iterable of batches with a .dataset whose
            length compute_hard_sinkhorn_assignments reads."""

        # Indexed loader shape: (images, labels, indices), shuffled order.
        loader = _FakeLoader([
            (imgs[[2, 0]], torch.zeros(2), torch.tensor([2, 0])),
            (imgs[[1, 3]], torch.zeros(2), torch.tensor([1, 3])),
            (imgs[[5, 4]], torch.zeros(2), torch.tensor([5, 4]))])
        loader.dataset = list(range(n))          # compute_* reads len(...)
        a = sk.compute_hard_sinkhorn_assignments(
            model, loader, torch.device("cpu"),
            num_heads=CLUSTERING["num_heads"], n_iters=50,
            temperature=1.0, lamb=25, tol=0.1, verbose=False)
        self.assertEqual(tuple(a.shape), (n, CLUSTERING["num_heads"]))
        self.assertTrue(int(a.min()) >= 0 and int(a.max()) < CLUSTERING["k"])
        b = sk.compute_hard_sinkhorn_assignments(
            model, loader, torch.device("cpu"),
            num_heads=CLUSTERING["num_heads"], n_iters=50,
            temperature=1.0, lamb=25, tol=0.1, verbose=False)
        self.assertTrue(torch.equal(a, b), "assignment is not deterministic")


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("sela_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_indexed_item_is_image_label_and_index(self):
        tiny_imagefolder(self.tmp / "data")
        dm = self.dataset_mod()
        ds = dm.IndexedImageFolder(
            str(self.tmp / "data"),
            transform=dm.get_sela_train_transform(MODEL["image_size"]))
        img, label, idx = ds[3]
        self.assertEqual(tuple(img.shape),
                         (3, MODEL["image_size"], MODEL["image_size"]))
        self.assertEqual(int(idx), 3)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_backbone_comes_out(self):
        got = adapter.extract_encoder({
            "backbone.features.0.weight": 1, "backbone.0.weight": 2,
            "top_layer.0.weight": 3, "top_layer.1.bias": 4})
        self.assertEqual(set(got),
                         {"backbone.features.0.weight", "backbone.0.weight"})

    def test_the_prototype_heads_are_left_out(self):
        got = adapter.extract_encoder({"backbone.features.0.weight": 1,
                                       "top_layer.0.weight": 2})
        self.assertNotIn("top_layer.0.weight", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"top_layer.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["clustering"]["k"], 8)
        self.assertEqual(built["clustering"]["num_heads"], 2)
        self.assertEqual(built["model"]["arch"], "resnetv2")
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
        # k is a step-1 clustering knob; the probe must reject it as unknown.
        cfg = self.eval_config(train={"k": 8})
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
        return load("sela_trainer", METHOD / "train_pretrain_sela.py")

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
        src = (METHOD / "train_pretrain_sela.py").read_text()
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
        tree = ast.parse((METHOD / "train_pretrain_sela.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("DataParallel", used)
        self.assertNotIn("SummaryWriter", used)


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native
# ResNetv2 SeLa pretrain. One ViT backbone + a single linear prototype head
# (top_layer); per-epoch Sinkhorn self-labelling, cross-entropy on the hard
# assignments. Tiny dims for a CPU smoke.
VIT_MODEL_ARGS = {"num_classes": 8, "image_size": 32, "patch_size": 16,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_TRAIN_TINY = {"arch": "vit", "k": 8, "image_size": 32, "patch_size": 16,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "drop_rate": 0.0, "attn_drop_rate": 0.0, "sinkhorn_iters": 3,
                  "sinkhorn_tol": 0.1, "lambda": 25, "epsilon": 0.05,
                  "temperature": 1.0, "epochs": 2, "batch_size": 2,
                  "num_workers": 0, "lr": 6.0e-4, "weight_decay": 0.05,
                  "warmup_epochs": 0, "min_lr": 0.0, "save_at_epochs": [1, 2]}


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
        self.assertEqual(built["clustering"]["k"], 8)
        self.assertEqual(built["training"]["save_at_epochs"], [1, 2])

    def test_native_path_has_no_top_level_arch(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertNotIn("arch", built)
        self.assertEqual(built["model"]["arch"], "resnetv2")

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
        vm = load("vit_sela", METHOD / "models" / "vit_sela.py")
        return vm.build_vit_sela(**VIT_MODEL_ARGS)

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, VIT_MODEL_ARGS["image_size"],
                           VIT_MODEL_ARGS["image_size"])

    @needs_timm
    def test_get_features_returns_the_cls_feature(self):
        import torch
        feats = self._model().get_features(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, VIT_MODEL_ARGS["embed_dim"]))

    @needs_timm
    def test_forward_returns_prototype_logits(self):
        import torch
        logits = self._model()(self._batch(torch))
        self.assertEqual(tuple(logits.shape), (2, VIT_MODEL_ARGS["num_classes"]))

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
        cfg = {"train": {"arch": "vit", "image_size": 32, "patch_size": 16,
                         "embed_dim": 16, "depth": 1, "num_heads": 2,
                         "mlp_ratio": 4.0, "drop_rate": 0.0,
                         "attn_drop_rate": 0.0}}
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
                "train": {"arch": "vit", "image_size": 32, "patch_size": 16,
                          "embed_dim": 16, "depth": 1, "num_heads": 2,
                          "mlp_ratio": 4.0, "drop_rate": 0.0,
                          "attn_drop_rate": 0.0, "epochs": 1, "batch_size": 2,
                          "num_workers": 0, "lr": 0.1, "momentum": 0.9,
                          "weight_decay": 0.0}}

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
    2048-d ResNet backbone feature -- raw, before the probe's normalise -- one
    row per val image, with honest meta.

    The encoder.pt is built from the *shipped* linear_eval config's
    architecture (the provider reads that config), via the same
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
        models = load("sela_models", METHOD / "models" / "__init__.py")
        trainer = load("sela_trainer", METHOD / "train_pretrain_sela.py")
        model = models.ResNetSeLa(num_classes=1, num_heads=1,
                                  **trainer.model_config(cfg["train"]))
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("sela_feature_provider", METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_2048d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("18_sela provider not yet present")
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
                         "ResNet backbone feature is 2048-d")
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
            self.skipTest("18_sela provider not yet present")
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
            self.skipTest("18_sela provider not yet present")
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
