#!/usr/bin/env python3
"""Specification for image_gpt (Chen et al., 2020; arXiv:2006.14671).

iGPT pretrains a GPT on **pixels**: an image is quantised to a sequence of
colour-cluster tokens and a causal transformer is trained to predict the next
token. The representation is a middle transformer layer, mean-pooled -- and,
unlike the generative ports var and mar, that representation is the model this
port trains, so `linear_eval` reads the trained `encoder.pt` and its number is a
genuine, comparable linear probe.

It is a **self-contained re-implementation**, ported from the lab's ARSSL inline
model (`src/models/train_igpt_scratch.py`), the same treatment the six official
methods got. The colour quantiser is a deterministic k-means implemented here in
numpy, so the dependency stack stays torch/torchvision/numpy and a run is
reproducible; the lab used sklearn's MiniBatchKMeans, whose saved clusters are
not in the capture, so exact clusters cannot be reproduced regardless.

Two paths are passed at run time; the output is fixed at --out. The hermetic
smoke uses a tiny model, a few fabricated images and a handful of clusters, so
nothing is downloaded and it runs on a CPU.
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
METHOD = ROOT / "methods" / "image_gpt"
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
    HAVE_DEPS, "image_gpt needs torch, numpy and torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("igpt_adapter", METHOD / "adapter" / "__init__.py")

# A model small enough to train a step on a CPU in a moment. img_size 8 -> a
# 64-token sequence; a handful of colour clusters; two tiny transformer blocks.
MODEL = {"vocab_size": 8, "img_size": 8, "n_layer": 2, "n_head": 2,
         "n_embd": 32}
TRAIN = {**MODEL, "epochs": 1, "batch_size": 2, "num_workers": 0,
         "lr": 3.0e-4, "grad_clip": 1.0}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    """A minimal ImageFolder of fabricated images under train/ -- no download."""
    import numpy as np
    from PIL import Image
    cls = root / "train" / "class0"
    cls.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype="uint8")).save(
            cls / f"{i}.png")
    return root


def tiny_split(root: Path, per: int = 3) -> Path:
    """A labelled ImageFolder with train/ and val/, two classes each, colour
    biased so the probe can beat chance."""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        for label, cls in enumerate(("c0", "c1")):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per):
                base = np.full((32, 32, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (32, 32, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="igpt-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(TRAIN)}
        for k, v in over.items():
            cfg["train"] = {**cfg["train"], **v} if k == "train" and v else cfg["train"]
            if k != "train":
                cfg[k] = v
        return cfg

    def eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "encoder": str(self.tmp / "encoder.pt"),
               "clusters": str(self.tmp / "clusters.npy"),
               "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            cfg["train"] = {**cfg["train"], **v} if k == "train" and v else cfg["train"]
            if k != "train":
                cfg[k] = v
        return cfg


# ─────────────────────────────────────────────────────────────────────────
# Colour quantisation (deterministic k-means)
# ─────────────────────────────────────────────────────────────────────────
class TestTheColourQuantiser(unittest.TestCase):
    def quant(self):
        return load("igpt_quantize", METHOD / "quantize.py")

    @needs_deps
    def test_kmeans_is_deterministic(self):
        import numpy as np
        q = self.quant()
        px = np.random.RandomState(1).rand(200, 3).astype("float32")
        a = q.kmeans(px, n_clusters=8, seed=0)
        b = q.kmeans(px, n_clusters=8, seed=0)
        self.assertTrue(np.array_equal(a, b), "same seed gave different clusters")
        self.assertEqual(a.shape, (8, 3))

    @needs_deps
    def test_quantise_maps_images_to_token_indices(self):
        import numpy as np
        import torch
        q = self.quant()
        clusters = np.random.RandomState(2).rand(8, 3).astype("float32")
        imgs = torch.rand(2, 3, 8, 8)
        tokens = q.quantize_images(imgs, clusters)
        self.assertEqual(tuple(tokens.shape), (2, 64))     # [B, H*W]
        self.assertEqual(tokens.dtype, torch.long)
        self.assertGreaterEqual(int(tokens.min()), 0)
        self.assertLess(int(tokens.max()), 8)              # in [0, n_clusters)


# ─────────────────────────────────────────────────────────────────────────
# The model and its representation
# ─────────────────────────────────────────────────────────────────────────
class TestTheModel(unittest.TestCase):
    def models(self):
        return load("igpt_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_extract_features_returns_one_vector_per_image(self):
        import torch
        m = self.models()
        model = m.build_igpt(**MODEL)
        tokens = torch.zeros(2, MODEL["img_size"] ** 2, dtype=torch.long)
        feats = model.extract_features(tokens)
        self.assertEqual(tuple(feats.shape), (2, MODEL["n_embd"]))

    @needs_deps
    def test_forward_predicts_over_the_colour_vocabulary(self):
        import torch
        m = self.models()
        model = m.build_igpt(**MODEL)
        tokens = torch.zeros(2, 10, dtype=torch.long)
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 10, MODEL["vocab_size"]))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_representation_side_comes_out(self):
        got = adapter.extract_encoder({
            "token_embed.weight": 1, "pos_embed.weight": 2,
            "blocks.0.ln1.weight": 3, "ln_f.weight": 4,
            "head.weight": 5})
        self.assertEqual(set(got), {"token_embed.weight", "pos_embed.weight",
                                    "blocks.0.ln1.weight", "ln_f.weight"})

    def test_the_generative_head_is_left_out(self):
        got = adapter.extract_encoder({"blocks.0.w": 1, "head.weight": 2})
        self.assertNotIn("head.weight", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"head.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


# ─────────────────────────────────────────────────────────────────────────
# Config translation
# ─────────────────────────────────────────────────────────────────────────
class TestConfigTranslation(Base):
    def test_every_step1_setting_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["training"]["epochs"], 1)
        self.assertEqual(built["model"]["n_embd"], 32)
        self.assertEqual(built["seed"], 0)

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
            adapter.to_run_config(self.config(train={"momentum": 0.9}),
                                  out=self.out)
        self.assertIn("momentum", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(stage="step2"), out=self.out)

    def test_a_config_that_sets_output_is_refused(self):
        cfg = self.config()
        cfg["output"] = {"checkpoint_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))


class TestLinearEvalConfig(Base):
    def test_linear_eval_is_accepted(self):
        adapter.to_run_config(self.eval_config(), out=self.out)

    def test_a_missing_eval_setting_is_refused_by_name(self):
        for key in EVAL_TRAIN:
            with self.subTest(key=key):
                cfg = self.eval_config()
                cfg["train"] = {k: v for k, v in EVAL_TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_the_encoder_and_clusters_must_be_named(self):
        for key in ("encoder", "clusters"):
            with self.subTest(key=key):
                cfg = self.eval_config()
                del cfg[key]
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_a_step1_only_key_is_not_read_by_eval(self):
        self.assertNotIn("grad_clip", EVAL_TRAIN)
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.eval_config(train={"grad_clip": 1.0}),
                                  out=self.out)
        self.assertIn("grad_clip", str(e.exception))


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
        for target in adapter.PRETRAIN_METRIC_NAMES.values():
            if target is not None:
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)
        self.assertEqual(adapter.PRETRAIN_METRIC_NAMES["final_loss"],
                         "final_pretext_loss")

    def test_eval_maps_the_comparable_probe_numbers(self):
        mapped = set(adapter.LINEAR_EVAL_METRIC_NAMES.values())
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, mapped)
            self.assertEqual(adapterlib.METRIC_VOCABULARY[name],
                             adapterlib.COMPARABLE)


# ─────────────────────────────────────────────────────────────────────────
# The device is resolved (referenced by the device mutation spec)
# ─────────────────────────────────────────────────────────────────────────
class TestTheDeviceIsResolved(Base):
    def trainer(self):
        return load("igpt_trainer", METHOD / "train_pretrain_igpt.py")

    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        t = self.trainer()
        with mock.patch.object(t.torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                t.resolve_device("cuda", 0)
            self.assertEqual(t.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(t.resolve_device("auto", 0).type, "cpu")

    def test_run_resolves_the_device_rather_than_sniffing_it(self):
        import ast
        src = (METHOD / "train_pretrain_igpt.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


# ─────────────────────────────────────────────────────────────────────────
# End-to-end smokes through the adapter and contract-test
# ─────────────────────────────────────────────────────────────────────────
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
    def test_it_writes_an_encoder_and_the_clusters(self):
        self.run_adapter()
        self.assertTrue((self.out / "encoder.pt").is_file())
        self.assertTrue((self.out / "clusters.npy").is_file(),
                        "the colour clusters are not carried over for the probe")

    @needs_deps
    def test_the_loss_is_recorded_as_a_pretext_number(self):
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m)

    @needs_deps
    def test_the_encoder_pt_it_wrote_loads_back(self):
        self.run_adapter()
        import torch
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved)
        # Several methods define a package called `models`, and only one can be
        # in sys.modules at a time. The adapter imports its own lazily, so put
        # the right one there first -- the same isolation the rest of the suite
        # uses. Without this the round trip errors when run in the full suite.
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
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty after a CUDA run")


class TestALinearEvalSmoke(Base):
    def _step1(self):
        """A step-1 run whose encoder.pt and clusters the probe will read."""
        tiny_split(self.tmp / "data")           # train/ + val/ for the probe
        step1_data = self.tmp / "s1data"
        tiny_imagefolder(step1_data)
        s1cfg = {"stage": "pretrain", "seed": 0, "data_root": str(step1_data),
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
        cfg = self.eval_config(encoder=str(s1out / "encoder.pt"),
                               clusters=str(s1out / "clusters.npy"), **over)
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
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, m)

    @needs_deps
    def test_it_produces_no_encoder_and_says_so(self):
        self.run_eval()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_the_probe_runs_on_cuda(self):
        # _step1 forces cpu; only the probe stage is exercised on cuda here, by
        # re-reading its encoder and probing on the GPU. The device invariant
        # (docs/GPU.md section 4).
        cfg, r = self.run_eval(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


class TestTheOriginalIsReferencedNotCopied(unittest.TestCase):
    def test_the_body_lives_in_run_and_main_only_parses(self):
        import ast
        src = (METHOD / "train_pretrain_igpt.py").read_text()
        top = {n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef)}
        for fn in ("run", "main", "build_parser"):
            self.assertIn(fn, top)

    def test_no_distributed_machinery_is_used(self):
        """The lab wrapper trains under DDP + torch.cuda.amp; the port owns a
        single-process loop instead. Checked against the code's identifiers
        (AST), not the source text -- the docstring names them to say they are
        avoided, and a substring search would match that prose."""
        import ast
        tree = ast.parse((METHOD / "train_pretrain_igpt.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("init_distributed_mode", used)


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. It reuses this method's
    own encoder loader and eval pipeline, so the check is that it returns the
    raw iGPT representation (a middle transformer layer, mean-pooled, n_embd
    wide) -- before the probe's normalise -- one row per val image, with honest
    meta.

    iGPT reads colour tokens, so an image must be quantised with the clusters
    the model was trained on. The adapter writes those as `clusters.npy` beside
    `encoder.pt`; the provider reads them from there. The encoder.pt is built
    from the *shipped* linear_eval config's architecture (the provider reads
    that config), via the same `extract_encoder` filter the adapter writes with,
    and a matching `clusters.npy` is written beside it; random weights and
    clusters do not affect the shape-and-plumbing this proves. Modules load
    through `load` (`load_from`), which purges any other method's
    `adapter`/`models` first -- the whole suite runs many methods in one
    interpreter.
    """

    def _shipped_config(self) -> dict:
        import yaml
        return yaml.safe_load(
            (METHOD / "configs" / "linear_eval.yaml").read_text())

    def _make_encoder(self, cfg: dict) -> Path:
        import numpy as np
        import torch
        models = load("igpt_provider_models", METHOD / "models" / "__init__.py")
        trainer = load("igpt_provider_trainer",
                       METHOD / "train_pretrain_igpt.py")
        model = models.build_igpt(**trainer.model_kwargs(cfg["train"]))
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        # The colour clusters the model was trained on, written beside
        # encoder.pt exactly as the adapter does; [vocab_size, 3] float32 in the
        # [0, 1] ToTensor range, so the provider can quantise with them.
        n = int(cfg["train"]["vocab_size"])
        clusters = np.random.RandomState(0).rand(n, 3).astype("float32")
        np.save(self.tmp / "clusters.npy", clusters)
        return encoder_pt

    def _provider(self):
        return load("igpt_feature_provider", METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_nembd_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("image_gpt provider not yet present")
        import numpy as np
        data_root = tiny_split(self.tmp / "data")
        cfg = self._shipped_config()
        feat_dim = int(cfg["train"]["n_embd"])
        encoder_pt = self._make_encoder(cfg)

        prov = self._provider()
        feats, labels, meta = prov.extract_val_features(
            encoder_path=str(encoder_pt), data_root=str(data_root),
            split="val", device="cpu", batch_size=2, num_workers=0)

        feats = np.asarray(feats)
        self.assertEqual(feats.ndim, 2)
        self.assertEqual(feats.shape[0], 6, "6 val images expected")
        self.assertEqual(feats.shape[1], feat_dim,
                         "iGPT feature is n_embd wide")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], feat_dim)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("image_gpt provider not yet present")
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        data_root = tiny_split(self.tmp / "data")
        cfg = self._shipped_config()
        feat_dim = int(cfg["train"]["n_embd"])
        encoder_pt = self._make_encoder(cfg)

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
        self.assertEqual(feats.shape, (6, feat_dim))
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
            self.skipTest("image_gpt provider not yet present")
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        data_root = tiny_split(self.tmp / "data")
        cfg = self._shipped_config()
        feat_dim = int(cfg["train"]["n_embd"])
        encoder_pt = self._make_encoder(cfg)
        out = self.tmp / "features"
        manifest = driver.run(
            METHOD.parent, data_root=str(data_root), split="val", out=out,
            encoders={METHOD.name: str(encoder_pt)}, encoders_root=None,
            device="cpu", batch_size=2, num_workers=0,
            venvs_root=ROOT / ".venvs")

        rec = {r["method"]: r for r in manifest["records"]}[METHOD.name]
        self.assertEqual(rec["status"], "ok", rec.get("reason", ""))
        feats = np.load(out / METHOD.name / "features.npy")
        self.assertEqual(feats.shape, (6, feat_dim))


if __name__ == "__main__":
    unittest.main()
