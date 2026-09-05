#!/usr/bin/env python3
"""Specification for 06_rotation_prediction (Gidaris et al., ICLR 2018; arXiv:1803.07728).

Rotation prediction: an image is rotated by one of {0, 90, 180, 270} degrees and
an AlexNet-BN predicts which rotation was applied (a 4-class pretext). A
self-contained re-implementation (the lab's own code, torch/torchvision), on the
existing step1 -> encoder.pt -> linear_probe contract; `encoder.pt` is the
AlexNet-BN feature extractor, and `linear_eval` probes it. The captured step 2
(ViT) is excluded, as in every port.
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
METHOD = ROOT / "methods" / "06_rotation_prediction"
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
    HAVE_DEPS, "06_rotation_prediction needs torch, numpy, torchvision")

try:
    import timm                                        # noqa: F401
    HAVE_TIMM = HAVE_DEPS
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("rotation_adapter", METHOD / "adapter" / "__init__.py")

# A model + images small enough to train a step on a CPU. num_classes is fixed at
# 4 (the four rotations); a 64px image is well below the paper's 224 but the
# encoder's adaptive pool accepts any size, so the smoke stays cheap.
MODEL = {"num_classes": 4, "image_size": 64}
TRAIN = {**MODEL, "epochs": 1, "batch_size": 2, "num_workers": 0,
         "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}
EVAL_TRAIN = {**MODEL, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0}
FEATURE_DIM = 4096


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
        self.tmp = Path(tempfile.mkdtemp(prefix="rot-"))
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
        return load("rotation_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_forward_predicts_over_the_four_rotations(self):
        import torch
        m = self.models()
        model = m.build_alexnet_rotation_model(num_classes=4)
        x = torch.zeros(2, 3, MODEL["image_size"], MODEL["image_size"])
        self.assertEqual(tuple(model(x).shape), (2, 4))

    @needs_deps
    def test_the_encoder_returns_one_feature_vector_per_image(self):
        import torch
        m = self.models()
        enc = m.build_alexnet_rotation_model(num_classes=4).get_encoder()
        enc.eval()
        feats = enc(torch.zeros(2, 3, MODEL["image_size"], MODEL["image_size"]))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))

    @needs_deps
    def test_the_encoder_accepts_the_papers_input_size(self):
        import torch
        m = self.models()
        enc = m.build_alexnet_rotation_model(num_classes=4).get_encoder()
        enc.eval()
        feats = enc(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM))


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("rotation_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_four_rotations_and_their_labels(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().RotationDataset(str(self.tmp / "data"))
        rotated, labels = ds[0]
        self.assertEqual(rotated.shape[0], 4)
        self.assertEqual(rotated.shape[1], 3)
        self.assertTrue(torch.equal(labels, torch.tensor([0, 1, 2, 3])))

    @needs_deps
    def test_the_rotations_are_the_four_right_angles(self):
        # A positive control on the geometry: the port's 90/180/270 rotations
        # must equal torch.rot90 of the 0-degree copy (counter-clockwise), so a
        # wrong flip/transpose is caught, not just a shape.
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().RotationDataset(str(self.tmp / "data"),
                                                normalize=False)
        rotated, _ = ds[0]
        base = rotated[0]
        for k in (1, 2, 3):
            self.assertTrue(
                torch.allclose(rotated[k], torch.rot90(base, k, dims=[1, 2])),
                f"rotation index {k} is not a {90 * k} degree turn")

    @needs_deps
    def test_the_collate_flattens_images_and_labels(self):
        import torch
        dm = self.dataset_mod()
        batch = [(torch.zeros(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])),
                 (torch.ones(4, 3, 8, 8), torch.tensor([0, 1, 2, 3]))]
        imgs, labels = dm.rotation_collate(batch)
        self.assertEqual(tuple(imgs.shape), (8, 3, 8, 8))
        self.assertTrue(torch.equal(labels, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_encoder_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.features.0.weight": 1, "encoder.fc_block.1.weight": 2,
            "classifier.weight": 3, "classifier.bias": 4})
        self.assertEqual(set(got),
                         {"encoder.features.0.weight", "encoder.fc_block.1.weight"})

    def test_the_classifier_is_left_out(self):
        got = adapter.extract_encoder({"encoder.features.0.weight": 1,
                                       "classifier.weight": 2})
        self.assertNotIn("classifier.weight", got)

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"classifier.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())


# --- Step 2: the unified ViT-B/16 path, added additively alongside the native
# AlexNet pretrain. Selected by `arch: vit` in the config; the native path
# (no arch / arch=alexnet) must be untouched. Faithful to the capture's
# train_step2_vit.py: ViT-B/16 from scratch, AdamW + cosine/warmup, 4-way
# rotation CE on the CLS token, checkpoints at save_at_epochs.
VIT_MODEL = {"arch": "vit", "num_classes": 4, "image_size": 224,
             "patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12,
             "mlp_ratio": 4.0, "drop_rate": 0.1, "attn_drop_rate": 0.0}
VIT_TRAIN = {**VIT_MODEL, "epochs": 300, "batch_size": 1024, "num_workers": 8,
             "lr": 0.0006, "weight_decay": 0.05, "betas": [0.9, 0.999],
             "warmup_epochs": 10, "min_lr": 1.0e-6, "clip_grad": 1.0,
             "save_at_epochs": [100, 200, 300]}
# Tiny ViT for CPU smoke: real arch shape, minimal dims.
VIT_TRAIN_TINY = {**VIT_TRAIN, "image_size": 32, "patch_size": 16,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "epochs": 2,
                  "batch_size": 2, "num_workers": 0, "warmup_epochs": 0,
                  "save_at_epochs": [1, 2]}


class TestVitConfigTranslation(Base):
    def vit_config(self, train=None, **over) -> dict:
        # Replace train entirely (the native Base.config would merge onto the
        # SGD TRAIN and drag `momentum` into the AdamW path).
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": dict(train if train is not None else VIT_TRAIN)}
        cfg.update(over)
        return cfg

    def test_the_vit_step2_config_is_accepted(self):
        built = adapter.to_run_config(self.vit_config(), out=self.out)
        self.assertEqual(built["arch"], "vit")
        self.assertEqual(built["model"]["embed_dim"], 768)
        self.assertEqual(built["training"]["epochs"], 300)
        self.assertEqual(built["training"]["save_at_epochs"], [100, 200, 300])

    def test_the_native_path_is_unchanged_when_arch_is_absent(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertNotIn("arch", built)          # native run-config shape intact
        self.assertEqual(built["model"]["num_classes"], 4)

    def test_a_bad_arch_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"arch": "resnet"}),
                                  out=self.out)
        self.assertIn("arch", str(e.exception))

    def test_a_missing_vit_setting_is_refused_by_name(self):
        for key in VIT_TRAIN:
            if key == "arch":
                continue
            with self.subTest(key=key):
                t = {k: v for k, v in VIT_TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.vit_config(train=t), out=self.out)
                self.assertIn(key, str(e.exception))

    def test_native_knobs_do_not_leak_into_the_vit_path(self):
        """`momentum` is an SGD (native) knob; the ViT path uses AdamW betas, so
        a momentum here is a config claiming an effect it never had."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.vit_config(train={**VIT_TRAIN, "momentum": 0.9}),
                out=self.out)
        self.assertIn("momentum", str(e.exception))

    def test_vit_knobs_do_not_leak_into_the_native_path(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"embed_dim": 768}),
                                  out=self.out)
        self.assertIn("embed_dim", str(e.exception))


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["num_classes"], 4)
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
        return load("rotation_trainer", METHOD / "train_pretrain_rotation.py")

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
        src = (METHOD / "train_pretrain_rotation.py").read_text()
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
        tree = ast.parse((METHOD / "train_pretrain_rotation.py").read_text())
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        self.assertNotIn("DistributedDataParallel", used)
        self.assertNotIn("SummaryWriter", used)


VIT_MODEL_ARGS = {"num_classes": 4, "image_size": 32, "patch_size": 16,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "drop_rate": 0.1, "attn_drop_rate": 0.0}


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_rotation", METHOD / "models" / "vit_rotation.py")
        return vm.build_vit_rotation_model(**VIT_MODEL_ARGS)

    @needs_timm
    def test_the_encoder_returns_the_cls_feature_per_image(self):
        import torch
        feats = self._model().get_encoder()(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(feats.shape), (2, VIT_MODEL_ARGS["embed_dim"]))

    @needs_timm
    def test_forward_predicts_over_the_four_rotations(self):
        import torch
        self.assertEqual(tuple(self._model()(torch.zeros(2, 3, 32, 32)).shape),
                         (2, 4))

    @needs_timm
    def test_encoder_pt_holds_only_the_backbone(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith("encoder.") for k in got))
        self.assertFalse([k for k in got if k.startswith("head")],
                         "the rotation head is training machinery, not encoder.pt")

    @needs_timm
    def test_load_encoder_round_trips_the_vit_weights(self):
        import torch
        saved = adapter.extract_encoder(self._model().state_dict())
        cfg = {"train": {"arch": "vit", **VIT_MODEL_ARGS}}
        loaded = adapter.load_encoder(saved, cfg).state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


class TestAVitStep2Smoke(Base):
    """The Step-2 pipeline end to end: unified ViT-B/16 (tiny) pretrain ->
    milestone encoders -> ImageNet-style linear probe -> contract-test. The
    native AlexNet pretrain path is untouched (its own smoke still runs)."""

    def _adapter(self, cfg_dict, out):
        cfg = self.tmp / (out.name + ".json")
        cfg.write_text(json.dumps(cfg_dict), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(out)], cwd=METHOD, env=env,
            capture_output=True, text=True)
        return cfg, r

    def _pretrain_cfg(self):
        return {"stage": "pretrain", "seed": 0,
                "data_root": str(self.tmp / "pre"), "device": "cpu",
                "train": dict(VIT_TRAIN_TINY)}

    def _eval_cfg(self, encoder):
        return {"stage": "linear_eval", "seed": 0,
                "data_root": str(self.tmp / "eval"), "device": "cpu",
                "encoder": str(encoder),
                "train": {"arch": "vit", **VIT_MODEL_ARGS, "epochs": 1,
                          "batch_size": 2, "num_workers": 0, "lr": 0.01,
                          "momentum": 0.9, "weight_decay": 0.0}}

    @needs_timm
    def test_pretrain_writes_milestone_encoders_then_probe_passes_contract(self):
        tiny_imagefolder(self.tmp / "pre")
        pre = self.tmp / "pre_out"
        _, r = self._adapter(self._pretrain_cfg(), pre)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((pre / "encoder.pt").is_file())
        for n in (1, 2):                                   # save_at_epochs=[1,2]
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
        self.assertFalse((ev / "encoder.pt").exists(),
                         "linear_eval must not write an encoder")


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. It reuses this method's
    own encoder loader and eval pipeline, so the check is that it returns the
    4096-d AlexNet-BN backbone feature -- raw, before the probe's normalise --
    one row per val image, with honest meta.

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
        models = load("rotation_models", METHOD / "models" / "__init__.py")
        trainer = load("rotation_trainer",
                       METHOD / "train_pretrain_rotation.py")
        model = models.build_alexnet_rotation_model(
            **trainer.model_kwargs(cfg["train"]))
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("rotation_feature_provider",
                    METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_4096d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("06_rotation_prediction provider not yet present")
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
                         "AlexNet-BN feature is 4096-d")
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
            self.skipTest("06_rotation_prediction provider not yet present")
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
            self.skipTest("06_rotation_prediction provider not yet present")
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
