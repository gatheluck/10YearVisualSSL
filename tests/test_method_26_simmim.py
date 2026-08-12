#!/usr/bin/env python3
"""Specification for 26_simmim (Xie et al., 2022; arXiv:2111.09886).

SimMIM: masked image modeling with a Swin-B encoder. Images are patch-embedded by
Swin; a random block of patch tokens is replaced by a learned mask token; the full
grid is encoded; a lightweight Conv + PixelShuffle decoder reconstructs pixels;
an L1 loss is taken only on the masked pixels. SimMIM's step 1 is genuinely
Swin-based -- `timm` supplies the SwinTransformer -- but the Swin is built from
scratch (no pretrained download), so the run stays hermetic.

`encoder.pt` is the bare Swin encoder (`encoder.*`, the prefix stripped so it
loads into a plain timm SwinTransformer); the learned mask token and the decoder
are training machinery and are excluded. `linear_eval` probes the Swin's
mean-pooled features (encoder_dim). The captured step 2 (ViT) is excluded, as in
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
METHOD = ROOT / "methods" / "26_simmim"
BIN = ROOT / "bin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import numpy                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import timm                                        # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "26_simmim needs torch, numpy, torchvision, timm")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("simmim_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny 2-stage Swin at a 16px input
# (patch2 -> an 8x8 token grid; two stages -> a 4x4 final grid, encoder_dim 32),
# 4px mask units. The paper's Swin-B / 192px / out 1024 / 800 epochs live in the
# shipped config.
IMG = 16
PATCH = 2
WINDOW = 4
EMBED = 16
DEPTHS = [2, 2]
NUM_HEADS = [2, 4]
MASK_PATCH = 4
ENCODER_DIM = 32  # EMBED * 2**(len(DEPTHS)-1)

MODEL = {"img_size": IMG, "patch_size": PATCH, "window_size": WINDOW,
         "embed_dim": EMBED, "depths": DEPTHS, "num_heads": NUM_HEADS,
         "mask_patch_size": MASK_PATCH, "drop_path_rate": 0.0}
DATA = {"mask_ratio": 0.6}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1.0e-3,
              "scale_lr_by_global_batch": False, "lr_reference_batch_size": 512,
              "betas": [0.9, 0.999], "weight_decay": 0.05, "warmup_epochs": 0,
              "warmup_lr": 0.0, "clip_grad": 5.0, "lr_gamma": 0.1,
              "lr_multisteps": []}
TRAIN = {**MODEL, **DATA, **STEP1_ONLY}
EVAL_MODEL = {"img_size": IMG, "patch_size": PATCH, "window_size": WINDOW,
              "embed_dim": EMBED, "depths": DEPTHS, "num_heads": NUM_HEADS}
EVAL_TRAIN = {**EVAL_MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}


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
        self.tmp = Path(tempfile.mkdtemp(prefix="simmim-"))
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
        return load("simmim_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.build_simmim_swinb(
            img_size=IMG, patch_size=PATCH, window_size=WINDOW, embed_dim=EMBED,
            depths=tuple(DEPTHS), num_heads=tuple(NUM_HEADS),
            mask_patch_size=MASK_PATCH, drop_path_rate=0.0)

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, IMG, IMG)

    def _mask(self, torch, b=2, ratio=0.6):
        grid = IMG // PATCH
        m = (torch.rand(b, grid, grid) < ratio).float()
        return m

    @needs_deps
    def test_forward_returns_a_finite_scalar_loss_and_a_reconstruction(self):
        import torch
        model = self._model(self.models())
        model.train()
        loss, pred = model(self._batch(torch), self._mask(torch))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(pred.shape), (2, 3, IMG, IMG))

    @needs_deps
    def test_the_loss_is_taken_only_on_masked_pixels(self):
        import torch
        model = self._model(self.models())
        model.eval()
        x = self._batch(torch)
        zeros = torch.zeros(2, IMG // PATCH, IMG // PATCH)
        ones = torch.ones(2, IMG // PATCH, IMG // PATCH)
        with torch.no_grad():
            loss_none, _ = model(x, zeros)
            loss_all, _ = model(x, ones)
        self.assertAlmostEqual(loss_none.item(), 0.0, places=6,
                               msg="loss is nonzero with no masked pixels")
        self.assertGreater(loss_all.item(), 0.0)

    @needs_deps
    def test_masked_tokens_use_the_mask_token(self):
        # With every patch masked, all tokens become the mask token, so changing
        # the mask token changes the reconstruction (and the loss). If the mask
        # token is not applied, the loss is independent of it.
        import torch
        model = self._model(self.models())
        model.eval()
        x = self._batch(torch)
        ones = torch.ones(2, IMG // PATCH, IMG // PATCH)
        with torch.no_grad():
            loss1, _ = model(x, ones)
            model.mask_token.zero_().add_(5.0)
            loss2, _ = model(x, ones)
        self.assertFalse(torch.allclose(loss1, loss2),
                         "changing the mask token did not change the fully-masked "
                         "reconstruction -- the mask token is not being applied")

    @needs_deps
    def test_encode_global_returns_pooled_features(self):
        import torch
        model = self._model(self.models())
        feats = model.encode_global(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, ENCODER_DIM))

    @needs_deps
    def test_get_encoder_is_a_usable_backbone(self):
        import torch
        model = self._model(self.models())
        enc = model.get_encoder()
        ff = enc.forward_features(self._batch(torch))
        self.assertEqual(ff.shape[0], 2)
        self.assertEqual(ff.shape[-1], ENCODER_DIM)


class TestTheMaskGenerator(unittest.TestCase):
    def dataset_mod(self):
        return load("simmim_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_the_patch_grid_mask_has_the_right_shape_and_ratio(self):
        import numpy as np
        np.random.seed(0)
        gen = self.dataset_mod().MaskGenerator(
            input_size=IMG, mask_patch_size=MASK_PATCH, model_patch_size=PATCH,
            mask_ratio=0.6)
        mask = gen()
        grid = IMG // PATCH
        self.assertEqual(mask.shape, (grid, grid))
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))
        # rand_size = 16/4 = 4 -> 16 units, ceil(16*0.6)=10 masked, each a 2x2
        # patch block: 10*4 / 64 = 0.625 of the patch grid.
        self.assertAlmostEqual(float(mask.mean()), 40.0 / 64.0, places=6)


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("simmim_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_an_image_a_patch_grid_mask_and_a_label(self):
        tiny_imagefolder(self.tmp / "data" / "train")
        ds = self.dataset_mod().SimMIMDataset(
            str(self.tmp / "data" / "train"), img_size=IMG,
            mask_patch_size=MASK_PATCH, mask_ratio=0.6, model_patch_size=PATCH,
            return_pixel_mask=False)
        img, mask, label = ds[0]
        self.assertEqual(tuple(img.shape), (3, IMG, IMG))
        self.assertEqual(tuple(mask.shape), (IMG // PATCH, IMG // PATCH))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_swin_encoder_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.patch_embed.proj.weight": 1,
            "encoder.layers.0.blocks.0.norm1.weight": 2,
            "mask_token": 3, "decoder.0.weight": 4})
        self.assertEqual(set(got),
                         {"patch_embed.proj.weight",
                          "layers.0.blocks.0.norm1.weight"})

    def test_the_mask_token_and_decoder_are_left_out(self):
        got = adapter.extract_encoder({"encoder.norm.weight": 1,
                                       "mask_token": 2, "decoder.0.bias": 3})
        self.assertEqual(set(got), {"norm.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"decoder.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["embed_dim"], EMBED)
        self.assertEqual(built["data"]["mask_ratio"], 0.6)
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
        cfg = self.eval_config(train={"mask_ratio": 0.6})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("mask_ratio", str(e.exception))


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
        return load("simmim_trainer", METHOD / "train_step1_simmim.py")

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
        src = (METHOD / "train_step1_simmim.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


class TestTheSchedule(unittest.TestCase):
    def trainer(self):
        return load("simmim_trainer", METHOD / "train_step1_simmim.py")

    @needs_deps
    def test_warmup_rises_then_multistep_decays(self):
        t = self.trainer()
        cfg = {"training": {"lr": 1.0, "warmup_epochs": 2, "warmup_lr": 0.0,
                            "lr_gamma": 0.1, "lr_multisteps": [5],
                            "scale_lr_by_global_batch": False}}
        lr_start = t.get_lr_at_update(0, 1, cfg)     # warmup start
        lr_peak = t.get_lr_at_update(2, 1, cfg)      # end of warmup
        lr_after = t.get_lr_at_update(6, 1, cfg)     # past milestone 5
        self.assertLess(lr_start, lr_peak, "warmup did not raise the LR")
        self.assertAlmostEqual(lr_peak, 1.0, places=6)
        self.assertLess(lr_after, lr_peak, "the multistep decay did not fire")


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
        tree = ast.parse((METHOD / "train_step1_simmim.py").read_text())
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
