#!/usr/bin/env python3
"""Specification for 31_dinov3 (DINOv3; Simeoni et al., 2025; arXiv:2508.10104).

The capture's step 1 loads the HF-gated official DINOv3 weights (the from-scratch
data, LVD-1689M, is not public), so it is excluded. What is ported is the
capture's step 2: the from-scratch **unified SSL comparison** on ImageNet, DINOv3's
core objective -- a student ViT (register tokens + axial RoPE) and an EMA teacher
over multi-crop views, trained by L_DINO (Sinkhorn-centred) + L_iBOT (masked
patches) + koleo_weight * L_KoLeo. The released Gram anchoring second stage is
excluded (the capture's `gram.mode: core_only`), as every port excludes a
secondary stage.

`encoder.pt` is the EMA teacher's ViT backbone (`backbone.*` from
`teacher_state_dict`, the prefix stripped); the DINO/iBOT heads and the student are
excluded. `linear_eval` probes the teacher backbone's CLS token. The ViT is the
lab's own (no timm), so this port is torch-only.
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
METHOD = ROOT / "methods" / "31_dinov3"
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
    HAVE_DEPS, "31_dinov3 needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("dinov3_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: a tiny ViT at 32px global / 16px local
# (patch16 -> a 2x2 = 4 patch grid), embed_dim 32, 2 blocks, 2 heads (RoPE needs
# embed_dim % (4*num_heads) == 0 -> 32 % 8 == 0), 2 register tokens, tiny heads,
# 2 global + 2 local crops. The paper's ViT-B/16 / 224px / 300 epochs live in the
# shipped config.
IMG = 32
PATCH = 16
EMBED = 32
NG, NL = 2, 2

MODEL = {"img_size": IMG, "patch_size": PATCH, "embed_dim": EMBED, "depth": 2,
         "num_heads": 2, "mlp_ratio": 2.0, "n_register_tokens": 2,
         "drop_path_rate": 0.0, "use_rope": True, "rope_base": 100.0,
         "dino_out_dim": 64, "ibot_out_dim": 64, "dino_head_hidden_dim": 32,
         "dino_head_bottleneck_dim": 16, "ibot_head_hidden_dim": 32,
         "ibot_head_bottleneck_dim": 16}
DATA = {"n_global_crops": NG, "n_local_crops": NL, "global_size": IMG,
        "local_size": 16, "global_scale": [0.5, 1.0], "local_scale": [0.2, 0.5],
        "num_workers": 0}
TRAINING = {"epochs": 1, "batch_size": 2, "lr": 1.0e-3, "min_lr": 1.0e-6,
            "warmup_epochs": 0, "weight_decay": 0.05,
            "teacher_momentum_start": 0.99, "teacher_momentum_end": 1.0,
            "grad_clip": 3.0, "ibot_mask_ratio_min": 0.1,
            "ibot_mask_ratio_max": 0.5, "ibot_mask_sample_probability": 0.5,
            "koleo_loss_weight": 0.1}
LOSS = {"student_temp": 0.1, "teacher_temp_start": 0.04, "teacher_temp_end": 0.07,
        "teacher_temp_warmup_epochs": 0, "sk_n_iters": 3}
TRAIN = {**MODEL, **DATA, **TRAINING, **LOSS}
EVAL_MODEL_ARGS = ("img_size", "patch_size", "embed_dim", "depth", "num_heads",
                   "mlp_ratio", "n_register_tokens", "drop_path_rate", "use_rope",
                   "rope_base")
EVAL_TRAIN = {**{k: MODEL[k] for k in EVAL_MODEL_ARGS},
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
                Image.fromarray((base + noise).astype("uint8")).save(d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinov3-"))
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
               "encoder": str(self.tmp / "encoder.pt"), "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg


class TestTheModel(unittest.TestCase):
    def trainer(self):
        return load("dinov3_trainer", METHOD / "train_step1_dinov3.py")

    @needs_deps
    def test_backbone_forward_returns_cls_and_patches(self):
        import torch
        t = self.trainer()
        vit = t.build_vit(**{k: MODEL[k] for k in EVAL_MODEL_ARGS})
        cls, patches = vit(torch.randn(2, 3, IMG, IMG), is_global=True)
        self.assertEqual(tuple(cls.shape), (2, EMBED))
        self.assertEqual(patches.shape[0], 2)
        self.assertEqual(patches.shape[-1], EMBED)

    @needs_deps
    def test_model_forward_returns_dino_ibot_cls_patches(self):
        import torch
        t = self.trainer()
        model = t.DINOv3Model(MODEL)
        crops = [torch.randn(2, 3, IMG, IMG) for _ in range(NG)] + \
                [torch.randn(2, 3, 16, 16) for _ in range(NL)]
        grid = (IMG // PATCH) ** 2
        masks = torch.zeros(2 * NG, grid, dtype=torch.bool)
        masks[:, 0] = True
        dino, ibot, cls_all, patches = model(crops, n_global=NG,
                                             masks_global=masks,
                                             ibot_selection_masks=masks)
        self.assertEqual(tuple(dino.shape), (2 * (NG + NL), MODEL["dino_out_dim"]))
        self.assertEqual(tuple(ibot.shape), (int(masks.sum()), MODEL["ibot_out_dim"]))

    @needs_deps
    def test_ema_moves_teacher_toward_student(self):
        import torch
        t = self.trainer()
        student = t.DINOv3Model(MODEL)
        teacher = t.DINOv3Model(MODEL)
        for p in teacher.parameters():
            p.data.zero_()
        for p in student.parameters():
            p.data.fill_(10.0)
        t.update_ema(teacher, student, momentum=0.9)
        moved = next(teacher.parameters()).data
        self.assertGreater(float(moved.mean()), 0.5)
        self.assertLess(float(moved.mean()), 5.0)


class TestTheLosses(unittest.TestCase):
    def losses(self):
        return load("dinov3_losses", METHOD / "losses" / "__init__.py")

    @needs_deps
    def test_dino_loss_is_a_finite_scalar(self):
        import torch
        L = self.losses()
        dino = L.DINOLoss(n_crops_global=NG, n_crops_local=NL)
        s = torch.randn(2 * (NG + NL), 64)
        te = torch.randn(2 * NG, 64)
        val = dino(s, te)
        self.assertEqual(val.dim(), 0)
        self.assertTrue(torch.isfinite(val))

    @needs_deps
    def test_koleo_is_zero_for_a_single_sample(self):
        import torch
        L = self.losses()
        self.assertEqual(float(L.KoLeoLoss()(torch.randn(1, 8))), 0.0)


class TestTheDataset(Base):
    def data_mod(self):
        return load("dinov3_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_multicrop_returns_global_plus_local_views(self):
        from PIL import Image
        import numpy as np
        d = self.data_mod()
        aug = d.MultiCropAugmentation(global_size=IMG, local_size=16, n_global=NG,
                                      n_local=NL)
        img = Image.fromarray(np.zeros((48, 48, 3), dtype="uint8"))
        views = aug(img)
        self.assertEqual(len(views), NG + NL)
        self.assertEqual(tuple(views[0].shape), (3, IMG, IMG))
        self.assertEqual(tuple(views[-1].shape), (3, 16, 16))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_backbone_comes_out_prefix_stripped(self):
        got = adapter.extract_encoder({
            "backbone.cls_token": 1, "backbone.blocks.0.norm1.weight": 2,
            "dino_head.last_layer.weight": 3, "ibot_head.mlp.0.weight": 4})
        self.assertEqual(set(got), {"cls_token", "blocks.0.norm1.weight"})

    def test_the_heads_are_left_out(self):
        got = adapter.extract_encoder({"backbone.norm.weight": 1,
                                       "dino_head.last_layer.weight": 2})
        self.assertEqual(set(got), {"norm.weight"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"dino_head.last_layer.weight": 1})
        self.assertIn("backbone", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["embed_dim"], EMBED)
        self.assertEqual(built["data"]["n_global_crops"], NG)
        self.assertEqual(built["loss"]["sk_n_iters"], 3)
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
            adapter.to_run_config(self.config(stage="gram"), out=self.out)

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
        cfg = self.eval_config(train={"n_global_crops": NG})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("n_global_crops", str(e.exception))


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
        return load("dinov3_trainer", METHOD / "train_step1_dinov3.py")

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
        src = (METHOD / "train_step1_dinov3.py").read_text()
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
        # step1 reads data_root as an ImageFolder root directly
        c = self.config(**over)
        c["data_root"] = str(self.tmp / "data" / "train")
        cfg.write_text(json.dumps(c), encoding="utf-8")
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
        s1cfg = {"stage": "step1", "seed": 0,
                 "data_root": str(s1data / "train"), "device": "cpu",
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
        tree = ast.parse((METHOD / "train_step1_dinov3.py").read_text())
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
