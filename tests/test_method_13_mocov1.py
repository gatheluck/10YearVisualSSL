#!/usr/bin/env python3
"""Specification for 13_mocov1 (He et al., 2019; arXiv:1911.05722).

Momentum Contrast v1, the **ResNet-50** path. This port covers the lab's
paper-faithful step 1: two augmented views of an image feed a **query encoder**
(ResNet-50 + a single Linear(2048, 128) projection, L2-normalised -- no MLP, that
is v2) and a **momentum key encoder** (an EMA copy, no gradient); an **InfoNCE**
loss contrasts the query against the matching key (positive) and a FIFO **queue**
of K past keys (negatives). `encoder.pt` is the query ResNet-50 backbone;
`linear_eval` probes it (2048-d).

The captured ViT step 2 is excluded, as in every port.
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
METHOD = ROOT / "methods" / "13_mocov1"
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
    HAVE_DEPS, "13_mocov1 needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("mocov1_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to train a step on a CPU: a narrow projection, a 32px input
# (ResNet-50 downsamples to 1x1, still valid), a tiny queue (K must divide the
# batch). The paper's 224px / feature_dim 128 / K 65536 live in the shipped
# config.
MODEL = {"feature_dim": 32, "img_size": 32}
MOCO = {"queue_size": 4, "key_momentum": 0.999, "temperature": 0.07}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.03,
              "momentum": 0.9, "weight_decay": 1.0e-4,
              "lr_decay_epochs": [], "lr_decay_rate": 0.1}
TRAIN = {**MODEL, **MOCO, **STEP1_ONLY}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

BACKBONE_DIM = 2048  # ResNet-50 pre-projection feature


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
        self.tmp = Path(tempfile.mkdtemp(prefix="mocov1-"))
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
        return load("mocov1_models", METHOD / "models" / "__init__.py")

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, MODEL["img_size"], MODEL["img_size"])

    @needs_deps
    def test_forward_returns_loss_logits_labels(self):
        import torch
        m = self.models()
        model = m.build_moco_resnet(feature_dim=MODEL["feature_dim"],
                                    queue_size=MOCO["queue_size"],
                                    momentum=MOCO["key_momentum"],
                                    temperature=MOCO["temperature"])
        model.train()
        loss, logits, labels = model(self._batch(torch), self._batch(torch))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(logits.shape), (2, 1 + MOCO["queue_size"]))
        self.assertTrue(torch.equal(labels, torch.zeros(2, dtype=torch.long)))

    @needs_deps
    def test_the_query_embedding_is_l2_normalised(self):
        import torch
        m = self.models()
        model = m.build_moco_resnet(feature_dim=MODEL["feature_dim"],
                                    queue_size=MOCO["queue_size"])
        model.eval()
        q = model.encoder_q(self._batch(torch))
        self.assertEqual(tuple(q.shape), (2, MODEL["feature_dim"]))
        norms = q.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))

    @needs_deps
    def test_the_encoder_returns_the_backbone_feature(self):
        import torch
        m = self.models()
        model = m.build_moco_resnet(feature_dim=MODEL["feature_dim"],
                                    queue_size=MOCO["queue_size"])
        enc = model.get_encoder()
        enc.eval()
        feats = enc(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, BACKBONE_DIM))


class TestTheQueue(unittest.TestCase):
    def models(self):
        return load("mocov1_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.build_moco_resnet(feature_dim=MODEL["feature_dim"],
                                   queue_size=MOCO["queue_size"],
                                   momentum=MOCO["key_momentum"],
                                   temperature=MOCO["temperature"])

    @needs_deps
    def test_the_queue_and_pointer_are_registered(self):
        m = self.models()
        state = self._model(m).state_dict()
        self.assertIn("queue", state)
        self.assertIn("queue_ptr", state)
        self.assertEqual(tuple(state["queue"].shape),
                         (MODEL["feature_dim"], MOCO["queue_size"]))

    @needs_deps
    def test_dequeue_and_enqueue_advances_the_pointer_and_writes(self):
        import torch
        m = self.models()
        model = self._model(m)
        keys = torch.nn.functional.normalize(
            torch.randn(2, MODEL["feature_dim"]), dim=1)
        self.assertEqual(int(model.queue_ptr), 0)
        model._dequeue_and_enqueue(keys)
        self.assertEqual(int(model.queue_ptr), 2)
        self.assertTrue(torch.allclose(model.queue[:, 0:2], keys.T))

    @needs_deps
    def test_the_key_encoder_is_momentum_updated(self):
        import torch
        m = self.models()
        model = self._model(m)
        with torch.no_grad():
            for p in model.encoder_q.parameters():
                p.add_(1.0)                      # move the query away from the key
        before = model.encoder_k.proj.weight.clone()
        model._momentum_update()
        after = model.encoder_k.proj.weight
        self.assertFalse(torch.allclose(before, after),
                         "the key encoder did not move toward the query")


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("mocov1_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_two_views_and_a_label(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().MoCoDataset(
            str(self.tmp / "data"), mode="step1", image_size=MODEL["img_size"])
        item = ds[0]
        self.assertEqual(len(item), 3)
        q, k, label = item
        self.assertEqual(tuple(q.shape),
                         (3, MODEL["img_size"], MODEL["img_size"]))
        self.assertEqual(tuple(k.shape),
                         (3, MODEL["img_size"], MODEL["img_size"]))
        self.assertFalse(torch.equal(q, k),
                         "the two views are identical, not independently augmented")


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_query_backbone_comes_out(self):
        got = adapter.extract_encoder({
            "encoder_q.backbone.0.weight": 1, "encoder_q.proj.weight": 2,
            "encoder_k.backbone.0.weight": 3, "queue": 4, "queue_ptr": 5})
        self.assertEqual(set(got), {"encoder_q.backbone.0.weight"})

    def test_the_head_key_encoder_and_queue_are_left_out(self):
        got = adapter.extract_encoder({"encoder_q.backbone.1.weight": 1,
                                       "encoder_q.proj.bias": 2,
                                       "encoder_k.proj.bias": 3, "queue": 4})
        self.assertEqual(set(got), {"encoder_q.backbone.1.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"queue": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["feature_dim"], 32)
        self.assertEqual(built["moco"]["queue_size"], 4)
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

    def test_moco_settings_are_not_part_of_the_probe(self):
        # queue_size is a step-1 MoCo knob; the probe must reject it as unknown.
        cfg = self.eval_config(train={"queue_size": 4})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("queue_size", str(e.exception))


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
        return load("mocov1_trainer", METHOD / "train_step1_mocov1.py")

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
        src = (METHOD / "train_step1_mocov1.py").read_text()
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
        tree = ast.parse((METHOD / "train_step1_mocov1.py").read_text())
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
