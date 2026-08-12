#!/usr/bin/env python3
"""Specification for 29_ijepa (Assran et al., 2023; arXiv:2301.08243).

I-JEPA: joint-embedding predictive architecture. A context encoder (ViT) sees a
large context block of patches; a narrow predictor predicts the representations of
several masked target blocks; the targets come from an EMA **target encoder** (a
momentum copy of the context encoder), and the loss is a smooth-L1 in latent
space. No pixel reconstruction, no hand-crafted augmentation invariances. I-JEPA
ships its own ViT (NOT timm) and trains from scratch on ImageNet, so the run is
torch-only and hermetic.

`encoder.pt` is the **target** ViT encoder (`target_encoder.*`, the prefix
stripped so it loads into a plain VisionTransformer; the capture's own linear eval
uses the target encoder). The context encoder, the predictor and the mask token
are training machinery and are excluded. `linear_eval` probes the target encoder's
mean-pooled patch tokens (embed_dim). The captured step 2 (ViT-B) is excluded, as
in every port.
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
METHOD = ROOT / "methods" / "29_ijepa"
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
    HAVE_DEPS, "29_ijepa needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("ijepa_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny ViT at a 32px input, patch 8 (a 4x4
# = 16 token grid, embed_dim 48), a narrow predictor, 2 target blocks. The paper's
# ViT-H/14 / 224px / 300 epochs live in the shipped config.
IMG = 32
PATCH = 8
GRID = IMG // PATCH          # 4
N = GRID * GRID              # 16
EMBED_DIM = 48               # vit_tiny embed_dim

MODEL = {"name": "vit_tiny", "img_size": IMG, "patch_size": PATCH}
PREDICTOR = {"pred_dim": 32, "pred_depth": 2}
DATA = {"augmentation": "step1", "use_horizontal_flip": False}
MASKING = {"num_enc_masks": 1, "num_pred_masks": 2, "allow_overlap": False,
           "min_keep": 4, "enc_mask_scale": [0.85, 1.0],
           "enc_mask_aspect": [1.0, 1.0], "pred_mask_scale": [0.15, 0.25],
           "pred_mask_aspect": [0.75, 1.5]}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1.0e-3,
              "start_lr": 2.0e-4, "final_lr": 1.0e-6, "weight_decay": 0.04,
              "final_wd": 0.4, "warmup_epochs": 0, "clip_grad": 10.0,
              "ipe_scale": 1.0, "beta1": 0.9, "beta2": 0.95,
              "start_ema": 0.996, "final_ema": 1.0}
TRAIN = {**MODEL, **PREDICTOR, **DATA, **MASKING, **STEP1_ONLY}
EVAL_TRAIN = {"name": "vit_tiny", "img_size": IMG, "patch_size": PATCH,
              "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}


def tiny_imagefolder(root: Path, n: int = 6) -> Path:
    import numpy as np
    from PIL import Image
    cls = root / "class0"
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
        self.tmp = Path(tempfile.mkdtemp(prefix="ijepa-"))
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
        return load("ijepa_models", METHOD / "models" / "__init__.py")

    def _encoder(self, m):
        return m.build_ijepa_encoder("vit_tiny", img_size=IMG, patch_size=PATCH)

    def _predictor(self, m):
        return m.build_ijepa_predictor("vit_tiny", img_size=IMG, patch_size=PATCH,
                                       pred_dim=PREDICTOR["pred_dim"],
                                       pred_depth=PREDICTOR["pred_depth"])

    @needs_deps
    def test_full_forward_returns_all_tokens(self):
        import torch
        enc = self._encoder(self.models())
        out = enc(torch.randn(2, 3, IMG, IMG))
        self.assertEqual(tuple(out.shape), (2, N, EMBED_DIM))

    @needs_deps
    def test_masked_forward_gathers_only_the_context(self):
        import torch
        enc = self._encoder(self.models())
        ctx_ids = torch.arange(6).unsqueeze(0).expand(2, -1)
        out = enc(torch.randn(2, 3, IMG, IMG), ctx_ids)
        self.assertEqual(tuple(out.shape), (2, 6, EMBED_DIM))

    @needs_deps
    def test_forward_features_mean_pools(self):
        import torch
        enc = self._encoder(self.models())
        feats = enc.forward_features(torch.randn(2, 3, IMG, IMG))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))

    @needs_deps
    def test_the_predictor_predicts_target_representations(self):
        import torch
        m = self.models()
        enc, pred = self._encoder(m), self._predictor(m)
        ctx_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        tgt_ids = torch.arange(8, 12).unsqueeze(0).expand(2, -1)
        z_ctx = enc(torch.randn(2, 3, IMG, IMG), ctx_ids)
        z_pred = pred(z_ctx, ctx_ids, tgt_ids)
        # predictor projects back to encoder_dim, one prediction per target patch
        self.assertEqual(tuple(z_pred.shape), (2, 4, EMBED_DIM))


class TestTheEmaUpdate(unittest.TestCase):
    def trainer(self):
        return load("ijepa_trainer", METHOD / "train_step1_ijepa.py")

    @needs_deps
    def test_ema_update_moves_the_target_toward_the_source(self):
        import torch
        t = self.trainer()
        m = load("ijepa_models", METHOD / "models" / "__init__.py")
        source = m.build_ijepa_encoder("vit_tiny", img_size=IMG, patch_size=PATCH)
        target = m.build_ijepa_encoder("vit_tiny", img_size=IMG, patch_size=PATCH)
        with torch.no_grad():
            for p in source.parameters():
                p.fill_(10.0)
            for p in target.parameters():
                p.fill_(0.0)
        t.ema_update(target, source, 0.9)
        after = next(iter(target.parameters()))
        self.assertGreater(after.mean().item(), 0.5,
                           "the EMA update did not move the target toward source")
        self.assertLess(after.mean().item(), 5.0,
                        "the target jumped to the source instead of an EMA step")


class TestTheSchedulers(unittest.TestCase):
    def utils(self):
        return load("ijepa_utils", METHOD / "utils" / "__init__.py")

    @needs_deps
    def test_cosine_warms_up_then_decays(self):
        u = self.utils()
        sched = u.cosine_scheduler(base_value=1.0, final_value=0.0,
                                   total_steps=100, warmup_steps=10,
                                   start_warmup_value=0.1)
        self.assertEqual(len(sched), 100)
        self.assertAlmostEqual(float(sched[0]), 0.1, places=4)
        self.assertAlmostEqual(float(sched[9]), 1.0, places=4)   # peak
        self.assertLess(float(sched[-1]), 0.05)                  # decayed
        self.assertLess(float(sched[0]), float(sched[9]))        # warmup rose

    @needs_deps
    def test_ema_momentum_rises_toward_final(self):
        u = self.utils()
        sched = u.ema_scheduler(base_value=0.996, final_value=1.0,
                                total_steps=100)
        self.assertAlmostEqual(float(sched[0]), 0.996, places=4)
        self.assertLess(float(sched[0]), float(sched[-1]))
        self.assertLessEqual(float(sched[-1]), 1.0 + 1e-5)


class TestTheMasking(unittest.TestCase):
    def masks(self):
        return load("ijepa_masks", METHOD / "masks" / "__init__.py")

    @needs_deps
    def test_the_collator_returns_context_and_target_masks(self):
        import torch
        collator = self.masks().MultiBlockMaskCollator(
            img_size=IMG, patch_size=PATCH, num_pred_masks=MASKING["num_pred_masks"],
            min_keep=MASKING["min_keep"], enc_mask_scale=tuple(MASKING["enc_mask_scale"]),
            pred_mask_scale=tuple(MASKING["pred_mask_scale"]))
        batch = [(torch.randn(3, IMG, IMG), 0), (torch.randn(3, IMG, IMG), 1)]
        images, labels, enc_masks, pred_masks = collator(batch)
        self.assertEqual(tuple(images.shape), (2, 3, IMG, IMG))
        self.assertEqual(enc_masks.shape[0], 2)
        self.assertLessEqual(int(enc_masks.max()), N - 1)
        self.assertEqual(len(pred_masks), MASKING["num_pred_masks"])
        for pm in pred_masks:
            self.assertEqual(pm.shape[0], 2)
            self.assertLessEqual(int(pm.max()), N - 1)


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_target_encoder_comes_out(self):
        got = adapter.extract_encoder({
            "target_encoder.pos_embed": 1,
            "target_encoder.blocks.0.norm1.weight": 2,
            "encoder.pos_embed": 3,
            "predictor.mask_token": 4})
        self.assertEqual(set(got), {"pos_embed", "blocks.0.norm1.weight"})

    def test_the_context_encoder_and_predictor_are_left_out(self):
        got = adapter.extract_encoder({"target_encoder.norm.weight": 1,
                                       "encoder.norm.weight": 2,
                                       "predictor.pos_embed": 3})
        self.assertEqual(set(got), {"norm.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"predictor.mask_token": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["name"], "vit_tiny")
        self.assertEqual(built["masking"]["num_pred_masks"], 2)
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
        cfg = self.eval_config(train={"pred_dim": 32})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("pred_dim", str(e.exception))


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
        return load("ijepa_trainer", METHOD / "train_step1_ijepa.py")

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
        src = (METHOD / "train_step1_ijepa.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


class TestAStep1Smoke(Base):
    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data" / "train")
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
        tiny_imagefolder(s1data / "train")
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
        tree = ast.parse((METHOD / "train_step1_ijepa.py").read_text())
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
