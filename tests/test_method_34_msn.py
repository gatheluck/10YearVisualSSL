#!/usr/bin/env python3
"""Specification for 34_msn (MSN; Assran et al., 2022; arXiv:2204.07141).

Masked Siamese Networks: an anchor ViT sees patch-dropped multi-crop views and an
EMA target ViT sees one un-dropped view; both are matched to learnable prototypes
via a soft-nearest-neighbour classifier under an MSN cross-entropy + a me-max
regulariser. The ViT and the MSN loss are the official facebookresearch/msn code,
pinned as the submodule third_party/msn and imported (never copied); only the
multi-view augmentation is reimplemented (the upstream one trips the pinned Pillow)
and the trainer is single-process.

`encoder.pt` is the anchor ViT trunk (the projection head `fc.*` excluded).
`linear_eval` probes its CLS token. Licence: CC BY-NC 4.0 (research-use, documented).
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
from _checkout import needs_checkout         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHOD = ROOT / "methods" / "34_msn"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "msn"
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
    HAVE_DEPS, "34_msn needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("msn_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny ViT at 32px rand / 16px focal
# (patch16 -> a 2x2 = 4 patch grid), embed_dim 32, 2 blocks, 2 heads, tiny head,
# 1 rand + 2 focal views, 8 prototypes. The paper's deit_small / 224px / 800
# epochs / 1024 prototypes live in the shipped config.
IMG, PATCH, EMBED = 32, 16, 32
RV, FV = 1, 2   # rand_views, focal_views

MODEL = {"img_size": IMG, "patch_size": PATCH, "embed_dim": EMBED, "depth": 2,
         "num_heads": 2, "mlp_ratio": 2.0, "use_bn": True, "hidden_dim": 64,
         "output_dim": 16, "drop_path_rate": 0.0}
DATA = {"focal_size": 16, "rand_crop_scale": [0.3, 1.0],
        "focal_crop_scale": [0.05, 0.3], "color_jitter": 0.5, "rand_views": RV,
        "focal_views": FV, "patch_drop": 0.15, "num_workers": 0,
        "label_smoothing": 0.0}
CRITERION = {"num_proto": 8, "temperature": 0.1, "start_sharpen": 0.25,
             "final_sharpen": 0.25, "me_max": True, "memax_weight": 1.0,
             "use_ent": True, "ent_weight": 0.0, "use_sinkhorn": True}
TRAINING = {"epochs": 1, "batch_size": 2, "lr": 1.0e-3, "start_lr": 2.0e-4,
            "final_lr": 1.0e-6, "warmup": 0, "weight_decay": 0.04,
            "final_weight_decay": 0.4, "clip_grad": 3.0, "ema_start": 0.996,
            "ema_final": 1.0}
TRAIN = {**MODEL, **DATA, **CRITERION, **TRAINING}
EVAL_MODEL_ARGS = ("img_size", "patch_size", "embed_dim", "depth", "num_heads",
                   "mlp_ratio", "drop_path_rate")
EVAL_TRAIN = {**{k: MODEL[k] for k in EVAL_MODEL_ARGS},
              "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}

# The additive unified deit_base/16 Step-2 recipe (recipe: unified, in `train`).
# The native MSN trainer already implements the recipe -- it uses the lr directly
# (init_opt takes ref_lr as-is; the capture baked lr=6e-4 into the config), builds
# the arch from the config dims (deit_base = embed_dim 768, num_heads 12), and its
# cosine weight-decay is constant when weight_decay == final_weight_decay. The only
# addition is milestone checkpoints (save_at_epochs), the sole unified-only key. The
# smoke keeps the tiny dims for CPU speed (the arch is set by shared value keys, not
# by the recipe); deit_base is covered by config translation.
UNIFIED_TRAINING = {"epochs": 1, "batch_size": 2, "lr": 6.0e-4, "start_lr": 0.0,
                    "final_lr": 1.0e-6, "warmup": 0, "weight_decay": 0.05,
                    "final_weight_decay": 0.05, "clip_grad": 3.0,
                    "ema_start": 0.996, "ema_final": 1.0, "save_at_epochs": [1]}
UNIFIED_TRAIN = {"recipe": "unified", **MODEL, **DATA, **CRITERION,
                 **UNIFIED_TRAINING}
DEIT_BASE_MODEL = {**MODEL, "embed_dim": 768, "num_heads": 12}


def _submodule_present() -> bool:
    return (UPSTREAM / "src" / "deit.py").is_file()


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
                Image.fromarray((base + noise).astype("uint8")).save(d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="msn-"))
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


class TestThePinnedUpstream(unittest.TestCase):
    @needs_checkout
    def test_the_adapter_records_the_checked_out_commit(self):
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("the submodule is not checked out here")
        self.assertEqual(r.stdout.strip(), adapter.UPSTREAM["commit"])

    def test_provenance_agrees_and_records_non_commercial_licence(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(prov["upstream"]["commit"], adapter.UPSTREAM["commit"])
        self.assertIn("facebookresearch/msn", adapter.UPSTREAM["repo"])
        blob = json.dumps(prov).lower()
        self.assertIn("cc by-nc", blob)
        self.assertIn("non-commercial", blob)


class TestTheModel(unittest.TestCase):
    def models(self):
        return load("msn_models", METHOD / "models" / "__init__.py")

    @needs_deps
    def test_encoder_forward_returns_before_and_after_head(self):
        import torch
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        enc = self.models().build_msn_model(**MODEL)
        views = [torch.randn(2, 3, 16, 16) for _ in range(FV)]
        h, z = enc(views, return_before_head=True, patch_drop=0.15)
        # anchor rows = FV views x batch 2; z (after head) at output_dim
        self.assertEqual(tuple(z.shape), (FV * 2, MODEL["output_dim"]))

    @needs_deps
    def test_backbone_forward_features_is_the_cls_at_embed_dim(self):
        import torch
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        bb = self.models().build_msn_backbone(**{k: MODEL[k] for k in EVAL_MODEL_ARGS})
        h, _z = bb(torch.randn(3, 3, IMG, IMG), return_before_head=True)
        self.assertEqual(tuple(h.shape), (3, EMBED))


class TestTheData(Base):
    def data_mod(self):
        return load("msn_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_multiview_returns_rand_plus_focal_views(self):
        from PIL import Image
        import numpy as np
        aug = self.data_mod().MSNMultiViewTransform(
            rand_size=IMG, focal_size=16, rand_views=RV, focal_views=FV)
        views = aug(Image.fromarray(np.zeros((48, 48, 3), dtype="uint8")))
        self.assertEqual(len(views), RV + FV)
        self.assertEqual(tuple(views[0].shape), (3, IMG, IMG))
        self.assertEqual(tuple(views[-1].shape), (3, 16, 16))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_trunk_comes_out(self):
        got = adapter.extract_encoder({
            "cls_token": 1, "blocks.0.norm1.weight": 2, "patch_embed.proj.weight": 3,
            "fc.fc1.weight": 4, "fc.bn1.weight": 5})
        self.assertEqual(set(got),
                         {"cls_token", "blocks.0.norm1.weight",
                          "patch_embed.proj.weight"})

    def test_the_projection_head_is_left_out(self):
        got = adapter.extract_encoder({"norm.weight": 1, "fc.fc3.weight": 2})
        self.assertEqual(set(got), {"norm.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"fc.fc1.weight": 1})
        self.assertIn("empty", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["embed_dim"], EMBED)
        self.assertEqual(built["data"]["focal_views"], FV)
        self.assertEqual(built["criterion"]["num_proto"], 8)
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
            adapter.to_run_config(self.config(train={"nonsense": 1}), out=self.out)
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
        cfg = self.eval_config(train={"num_proto": 8})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("num_proto", str(e.exception))


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
        return load("msn_trainer", METHOD / "train_pretrain_msn.py")

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
        src = (METHOD / "train_pretrain_msn.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called)
        self.assertIn("make_deterministic", called)


class TestAStep1Smoke(Base):
    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data" / "train")
        c = self.config(**over)
        c["data_root"] = str(self.tmp / "data")
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(c), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return cfg, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_writes_an_encoder_and_a_pretext_loss(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        self.run_adapter()
        self.assertTrue((self.out / "encoder.pt").is_file())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m)

    @needs_deps
    def test_the_encoder_pt_it_wrote_loads_back(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        self.run_adapter()
        import torch
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved)
        # Bind `models` to this method's before load_encoder imports it (the
        # in-process suite shares the `models` package name across methods).
        load("this_methods_models", METHOD / "models" / "__init__.py")
        model = adapter.load_encoder(saved, self.eval_config())
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
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    @needs_deps
    def test_the_manifest_records_the_pinned_upstream(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


class TestALinearEvalSmoke(Base):
    def _step1(self):
        tiny_split(self.tmp / "data")
        s1data = self.tmp / "s1data"
        tiny_imagefolder(s1data / "train")
        s1cfg = {"stage": "pretrain", "seed": 0,
                 "data_root": str(s1data), "device": "cpu",
                 "train": dict(TRAIN)}
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
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        cfg, r = self.run_eval()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_reports_the_comparable_probe_numbers(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        self.run_eval()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy"):
            self.assertIn(name, m)

    @needs_deps
    def test_it_produces_no_encoder_and_says_so(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        self.run_eval()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)


class TestUnifiedConfigTranslation(Base):
    """The additive recipe: unified branch, selected in `train`. Its only extra
    key over the native step-1 set is save_at_epochs (milestones); the arch is
    changed by value (deit_base dims), not by a new key."""

    def unified_config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": dict(UNIFIED_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg

    def test_the_unified_recipe_is_accepted(self):
        built = adapter.to_run_config(self.unified_config(), out=self.out)
        self.assertEqual(built["training"]["save_at_epochs"], [1])
        self.assertEqual(built["training"]["weight_decay"], 0.05)
        self.assertEqual(built["training"]["final_weight_decay"], 0.05)
        self.assertNotIn("recipe", built)
        self.assertNotIn("recipe", built["training"])

    def test_the_native_recipe_is_still_accepted(self):
        native = self.config()
        self.assertNotIn("recipe", native["train"])
        built = adapter.to_run_config(native, out=self.out)
        self.assertNotIn("save_at_epochs", built["training"])

    def test_the_unified_recipe_accepts_deit_base_dims(self):
        built = adapter.to_run_config(
            self.unified_config(train=DEIT_BASE_MODEL), out=self.out)
        self.assertEqual(built["model"]["embed_dim"], 768)
        self.assertEqual(built["model"]["num_heads"], 12)

    def test_a_missing_unified_key_is_refused_by_name(self):
        for key in ("save_at_epochs", "weight_decay", "embed_dim"):
            with self.subTest(key=key):
                cfg = self.unified_config()
                cfg["train"] = {k: v for k, v in UNIFIED_TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_unified_key_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.unified_config(train={"mystery": 1}),
                                  out=self.out)
        self.assertIn("mystery", str(e.exception))

    def test_the_unified_only_key_on_the_native_path_is_refused(self):
        """save_at_epochs is the unified milestone knob; the native step-1 recipe
        does not know it."""
        cfg = self.config()
        cfg["train"] = {**cfg["train"], "save_at_epochs": [1]}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("save_at_epochs", str(e.exception))

    def test_a_bad_recipe_value_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.unified_config(train={"recipe": "turbo"}),
                                  out=self.out)
        self.assertIn("turbo", str(e.exception))


class TestAUnifiedStep2Smoke(Base):
    """A real unified-recipe run on a CPU (tiny dims for speed): milestone
    checkpoints and the full contract chain."""

    def run_adapter(self, **over):
        tiny_imagefolder(self.tmp / "data" / "train")
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": {**UNIFIED_TRAIN, **over.pop("train", {})}}
        for k, v in over.items():
            cfg[k] = v
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return p, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_each_milestone_encoder_is_written(self):
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
        _, r = self.run_adapter(train={"epochs": 2, "save_at_epochs": [1, 2]})
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((self.out / "encoder.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch1.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch2.pt").is_file())


BACKBONE_DIM = 384  # deit_small embed_dim (shipped linear_eval.yaml); the MSN
# anchor ViT CLS token, one vector per image -- the raw feature the probe reads.


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. It reuses this method's
    own encoder loader and eval pipeline, so the check is that it returns the
    384-d MSN backbone feature (the anchor ViT CLS token) -- raw, before the
    probe's normalise -- one row per val image, with honest meta.

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
        models = load("msn_models", METHOD / "models" / "__init__.py")
        train = cfg["train"]
        model = models.build_msn_backbone(
            **{k: train[k] for k in EVAL_MODEL_ARGS})
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("msn_feature_provider", METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_384d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("34_msn provider not yet present")
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
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
        self.assertEqual(feats.shape[1], BACKBONE_DIM,
                         "MSN anchor ViT CLS feature is 384-d")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], BACKBONE_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("34_msn provider not yet present")
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
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
        self.assertEqual(feats.shape, (6, BACKBONE_DIM))
        self.assertEqual(labels.shape[0], 6)
        self.assertEqual(meta["encoder_sha256"],
                         hashlib.sha256(encoder_pt.read_bytes()).hexdigest())

    @needs_deps
    def test_the_isolated_driver_run_extracts_this_method_end_to_end(self):
        """The whole driver, real subprocess, real provider -- it catches the
        class of regression where the isolated worker cannot see a repository-
        root module the provider needs (the worker puts ROOT on sys.path, as
        bin/launch.py sets PYTHONPATH=ROOT)."""
        if not (METHOD / "feature_provider.py").is_file():
            self.skipTest("34_msn provider not yet present")
        if not _submodule_present():
            self.skipTest("the msn submodule is not checked out here")
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
        self.assertEqual(feats.shape, (6, BACKBONE_DIM))


class TestTheOriginalIsReferencedNotCopied(unittest.TestCase):
    def test_no_distributed_or_tensorboard_machinery_is_used(self):
        import ast
        tree = ast.parse((METHOD / "train_pretrain_msn.py").read_text())
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
