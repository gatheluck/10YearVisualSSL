#!/usr/bin/env python3
"""Specification for 14_simclrv1 (Chen et al., 2020; arXiv:2002.05709).

SimCLR v1, the **ResNet-50** path. This port covers the lab's paper-faithful
step 1: two independently-augmented views of an image feed a shared ResNet-50
encoder and a 2-layer MLP projection head (2048->2048->out_dim, BN + ReLU,
L2-normalised output); the **NT-Xent** loss pulls the two views of an image
together and pushes every other image in the batch apart, optimised with
**LARS**. `encoder.pt` is the ResNet-50 backbone; `linear_eval` probes it
(2048-d).

The captured ViT step 2 (which needs `timm`) is excluded, as in every port.
"""

from __future__ import annotations

import hashlib
import json
import math
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
METHOD = ROOT / "methods" / "14_simclrv1"
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
    HAVE_DEPS, "14_simclrv1 needs torch, numpy, torchvision")

try:
    import timm                                        # noqa: F401
    HAVE_TIMM = HAVE_DEPS
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("simclr_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to train a step on a CPU: a narrow projection and a 32px input
# (ResNet-50 downsamples to 1x1, still valid). The paper's 224px / out_dim 128 /
# LARS lr 4.8 / 1000 epochs live in the shipped config.
MODEL = {"out_dim": 32, "img_size": 32}
LOSS = {"temperature": 0.1}
DATA = {"color_jitter_strength": 1.0}
PRETRAIN_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.5,
              "momentum": 0.9, "weight_decay": 1.0e-6, "eta": 0.001,
              "warmup_epochs": 0}
TRAIN = {**MODEL, **LOSS, **DATA, **PRETRAIN_ONLY}
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
        self.tmp = Path(tempfile.mkdtemp(prefix="simclr-"))
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
        return load("simclr_models", METHOD / "models" / "__init__.py")

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, MODEL["img_size"], MODEL["img_size"])

    @needs_deps
    def test_forward_projects_and_l2_normalises(self):
        import torch
        m = self.models()
        model = m.build_resnet_simclr(out_dim=MODEL["out_dim"])
        model.eval()
        z = model(self._batch(torch))
        self.assertEqual(tuple(z.shape), (2, MODEL["out_dim"]))
        norms = z.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4),
                        "the projection is not L2-normalised")

    @needs_deps
    def test_the_encoder_returns_the_backbone_feature(self):
        import torch
        m = self.models()
        enc = m.build_resnet_simclr(out_dim=MODEL["out_dim"]).get_encoder()
        enc.eval()
        feats = enc(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, BACKBONE_DIM))


class TestTheLoss(unittest.TestCase):
    def loss_mod(self):
        return load("simclr_loss", METHOD / "loss" / "__init__.py")

    @needs_deps
    def test_it_is_a_finite_scalar(self):
        import torch
        crit = self.loss_mod().NTXentLoss(temperature=LOSS["temperature"])
        z1 = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        z2 = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
        loss = crit(z1, z2)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @needs_deps
    def test_it_prefers_the_matched_positive(self):
        import torch
        crit = self.loss_mod().NTXentLoss(temperature=LOSS["temperature"])
        z = torch.eye(3)[:2]                 # two orthonormal directions e0, e1
        aligned = crit(z, z.clone())          # view2 row i matches view1 row i
        shuffled = crit(z, z.flip(0))         # positives swapped -> mismatched
        self.assertLess(aligned.item(), shuffled.item(),
                        "the loss does not pair view i with view i")

    @needs_deps
    def test_it_excludes_self_similarity(self):
        import torch
        # All four rows identical: every off-diagonal similarity is equal, so the
        # positive competes only against the 2N-2 other views (not itself). The
        # loss is then exactly ln(2N-1); if the diagonal self-term is not masked
        # it competes too and the loss becomes ln(2N).
        crit = self.loss_mod().NTXentLoss(temperature=LOSS["temperature"])
        z = torch.nn.functional.normalize(torch.ones(2, 4), dim=1)
        loss = crit(z, z.clone())
        self.assertAlmostEqual(loss.item(), math.log(3), places=4)

    @needs_deps
    def test_the_self_term_is_excluded_not_merely_set_low(self):
        # At a large temperature the positive no longer dominates, so the exact
        # loss depends on whether the diagonal self-term is *excluded* (-inf) or
        # left in at some value. Two orthonormal directions, both views equal:
        # each row competes its positive (sim 1/tau=0.1) against two zero
        # negatives, self excluded, so every row's loss is
        #   -log( e^0.1 / (2*e^0 + e^0.1) ) = 1.03307 (fp32)
        # Leaving the self-term in (unmasked at 0.1 -> ~1.34, or set to 0 -> ~1.31)
        # changes this well outside the tolerance.
        import torch
        crit = self.loss_mod().NTXentLoss(temperature=10.0)
        z = torch.eye(2)                      # rows e0, e1 (orthonormal)
        loss = crit(z, z.clone())
        self.assertAlmostEqual(loss.item(), 1.0330688, places=5)


class TestTheOptimizer(unittest.TestCase):
    def optim_mod(self):
        return load("simclr_optim", METHOD / "optim" / "__init__.py")

    @needs_deps
    def test_a_step_updates_the_parameters(self):
        import torch
        p = torch.nn.Parameter(torch.ones(4, 4))
        opt = self.optim_mod().LARS([p], lr=0.1, momentum=0.9,
                                    weight_decay=1e-6, eta=0.001)
        (p.sum()).backward()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.allclose(before, p.detach()),
                         "LARS.step did not move the parameter")


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("simclr_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_two_views_and_a_label(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().SimCLRDataset(
            str(self.tmp / "data"), image_size=MODEL["img_size"])
        item = ds[0]
        self.assertEqual(len(item), 3)
        v1, v2, label = item
        self.assertEqual(tuple(v1.shape),
                         (3, MODEL["img_size"], MODEL["img_size"]))
        self.assertEqual(tuple(v2.shape),
                         (3, MODEL["img_size"], MODEL["img_size"]))
        self.assertFalse(torch.equal(v1, v2),
                         "the two views are identical, not independently augmented")


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_backbone_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.0.weight": 1, "encoder.4.0.conv1.weight": 2,
            "projector.0.weight": 3, "projector.1.weight": 4})
        self.assertEqual(set(got),
                         {"encoder.0.weight", "encoder.4.0.conv1.weight"})

    def test_the_projection_head_is_left_out(self):
        got = adapter.extract_encoder({"encoder.0.weight": 1,
                                       "projector.0.weight": 2})
        self.assertNotIn("projector.0.weight", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"projector.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["out_dim"], 32)
        self.assertEqual(built["loss"]["temperature"], 0.1)
        self.assertEqual(built["training"]["epochs"], 1)
        self.assertEqual(built["training"]["eta"], 0.001)

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
        # temperature is a step-1 NT-Xent knob; the probe must reject it.
        cfg = self.eval_config(train={"temperature": 0.1})
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
        return load("simclr_trainer", METHOD / "train_pretrain_simclr.py")

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
        src = (METHOD / "train_pretrain_simclr.py").read_text()
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
        tree = ast.parse((METHOD / "train_pretrain_simclr.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native
# ResNet-50/LARS pretrain. NT-Xent on two views; tiny dims for a CPU smoke.
VIT_MODEL_ARGS = {"out_dim": 8, "image_size": 32, "patch_size": 16,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_TRAIN_TINY = {"arch": "vit", "out_dim": 8, "img_size": 32, "patch_size": 16,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "drop_rate": 0.0, "attn_drop_rate": 0.0, "temperature": 0.07,
                  "color_jitter_strength": 1.0, "epochs": 2, "batch_size": 2,
                  "num_workers": 0, "lr": 0.0006, "weight_decay": 0.05,
                  "warmup_epochs": 0, "min_lr": 1.0e-6, "clip_grad": 1.0,
                  "save_at_epochs": [1, 2]}


class TestVitConfigTranslation(Base):
    def vit_config(self, train=None, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": dict(train if train is not None else VIT_TRAIN_TINY)}
        cfg.update(over)
        return cfg

    def test_the_vit_step2_config_is_accepted(self):
        built = adapter.to_run_config(self.vit_config(), out=self.out)
        self.assertEqual(built["arch"], "vit")
        self.assertEqual(built["model"]["embed_dim"], 16)
        self.assertEqual(built["loss"]["temperature"], 0.07)
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
                self.vit_config(train={**VIT_TRAIN_TINY, "eta": 0.001}),
                out=self.out)
        self.assertIn("eta", str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vs = load("vit_simclr", METHOD / "models" / "vit_simclr.py")
        return vs.build_vit_simclr(**VIT_MODEL_ARGS)

    @needs_timm
    def test_the_encoder_returns_the_cls_feature(self):
        import torch
        feats = self._model().get_encoder()(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(feats.shape), (2, VIT_MODEL_ARGS["embed_dim"]))

    @needs_timm
    def test_forward_returns_a_normalised_projection(self):
        import torch
        z = self._model()(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(z.shape), (2, VIT_MODEL_ARGS["out_dim"]))

    @needs_timm
    def test_encoder_pt_holds_only_the_backbone(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith("encoder.") for k in got))
        self.assertFalse([k for k in got if k.startswith("projector")])

    @needs_timm
    def test_load_encoder_round_trips_the_vit_weights(self):
        import torch
        saved = adapter.extract_encoder(self._model().state_dict())
        cfg = {"train": {"arch": "vit", **VIT_MODEL_ARGS, "img_size": 32}}
        # build_vit_simclr uses image_size; the adapter maps img_size->image_size
        cfg["train"].pop("image_size")
        loaded = adapter.load_encoder(saved, cfg).state_dict()
        pairs = 0
        for k, want in saved.items():
            got = loaded.get(k)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{k} came back changed")
        self.assertGreater(pairs, 0)


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
                "train": {"arch": "vit", "out_dim": 8, "img_size": 32,
                          "patch_size": 16, "embed_dim": 16, "depth": 1,
                          "num_heads": 2, "mlp_ratio": 4.0, "drop_rate": 0.0,
                          "attn_drop_rate": 0.0, "epochs": 1, "batch_size": 2,
                          "num_workers": 0, "lr": 0.01, "momentum": 0.9,
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


if __name__ == "__main__":
    unittest.main()
