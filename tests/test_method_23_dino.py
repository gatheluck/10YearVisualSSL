#!/usr/bin/env python3
"""Specification for 23_dino (Caron et al., 2021; arXiv:2104.14294).

DINO: self-distillation with no labels. A **student** (ViT backbone + DINOHead)
sees all crops; a **teacher** (an EMA copy of the student) sees only the two
global crops. The loss is the cross-entropy between the teacher's centred +
sharpened output and the student's sharpened output, averaged over the crop pairs
where the views differ. An online **centre** (EMA) prevents collapse; the teacher
momentum follows a cosine schedule from its base to 1.0. DINO's ViT is
self-contained (its own vision_transformer.py, NOT timm), so the run is torch-only
and hermetic.

`encoder.pt` is the **teacher** ViT backbone (the representation DINO is known
for; the capture's own linear eval defaults to the teacher). The DINO head, the
centre and the whole student are training machinery and are excluded. `linear_eval`
probes the teacher backbone's CLS feature (embed_dim). The captured step 2 (ViT-B)
is excluded, as in every port.
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
METHOD = ROOT / "methods" / "23_dino"
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
    HAVE_DEPS, "23_dino needs torch, numpy, torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("dino_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run a step on a CPU: vit_small at a 32px global / 16px local
# input (patch16 -> a 2x2 / 1x1 token grid), a narrow head, 2 local crops. The
# paper's vit_small / 224px / out_dim 65536 / 100 epochs live in the shipped config.
IMG = 32
LOCAL = 16
N_LOCAL = 2
MODEL = {"arch": "vit_small", "img_size": IMG}
DINO = {"out_dim": 64, "hidden_dim": 32, "bottleneck_dim": 16,
        "use_bn_in_head": False, "norm_last_layer": True,
        "n_local_crops": N_LOCAL, "local_size": LOCAL,
        "global_crops_scale": [0.4, 1.0], "local_crops_scale": [0.05, 0.4],
        "student_temp": 0.1, "teacher_temp_init": 0.04,
        "teacher_temp_final": 0.04, "teacher_temp_warmup_epochs": 0,
        "momentum_teacher": 0.996}
STEP1_ONLY = {"epochs": 1, "batch_size": 2, "num_workers": 0,
              "drop_path_rate": 0.0, "lr": 1.0e-3, "min_lr": 1.0e-6,
              "warmup_epochs": 0, "weight_decay_start": 0.04,
              "weight_decay_end": 0.4, "clip_grad": 3.0, "freeze_last_layer": 0}
TRAIN = {**MODEL, **DINO, **STEP1_ONLY}
EVAL_TRAIN = {"arch": "vit_small", "img_size": IMG, "epochs": 2, "batch_size": 2,
              "num_workers": 0, "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

EMBED_DIM = 384  # vit_small CLS feature


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
        self.tmp = Path(tempfile.mkdtemp(prefix="dino-"))
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
        return load("dino_models", METHOD / "models" / "__init__.py")

    def _model(self, m):
        return m.build_dino(
            arch=MODEL["arch"], out_dim=DINO["out_dim"],
            n_local_crops=DINO["n_local_crops"],
            student_temp=DINO["student_temp"],
            teacher_temp_init=DINO["teacher_temp_init"],
            teacher_temp_final=DINO["teacher_temp_final"],
            teacher_temp_warmup_epochs=DINO["teacher_temp_warmup_epochs"],
            hidden_dim=DINO["hidden_dim"], bottleneck_dim=DINO["bottleneck_dim"],
            use_bn_in_head=DINO["use_bn_in_head"],
            norm_last_layer=DINO["norm_last_layer"], img_size=MODEL["img_size"])

    def _crops(self, torch, b=2):
        g = [torch.randn(b, 3, IMG, IMG) for _ in range(2)]
        loc = [torch.randn(b, 3, LOCAL, LOCAL) for _ in range(N_LOCAL)]
        return g + loc

    @needs_deps
    def test_forward_returns_a_finite_scalar_loss(self):
        import torch
        model = self._model(self.models())
        model.train()
        loss = model(self._crops(torch))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @needs_deps
    def test_the_teacher_is_frozen(self):
        model = self._model(self.models())
        self.assertTrue(all(not p.requires_grad
                            for p in model.teacher.parameters()))

    @needs_deps
    def test_update_teacher_moves_the_teacher_toward_the_student(self):
        # Set the teacher to 0 and the student to 10; the EMA update with m=0.9
        # must pull the teacher to 0.9*0 + 0.1*10 = 1.0 -- a real fraction of the
        # way toward the student. A decay-only (or teacher-reads-itself) update
        # leaves it at 0.
        import torch
        model = self._model(self.models())
        with torch.no_grad():
            for p in model.teacher.parameters():
                p.fill_(0.0)
            for p in model.student.parameters():
                p.fill_(10.0)
        model.update_teacher(0.9)
        after = next(iter(model.teacher.parameters()))
        self.assertGreater(after.mean().item(), 0.5,
                           "the EMA update did not move the teacher toward the student")
        self.assertLess(after.mean().item(), 5.0,
                        "the teacher jumped to the student instead of an EMA step")

    @needs_deps
    def test_the_centre_updates_toward_the_batch_mean(self):
        import torch
        model = self._model(self.models())
        loss_fn = model.loss_fn
        before = loss_fn.center.clone()
        teacher_out = torch.full((4, DINO["out_dim"]), 5.0)
        loss_fn._update_center(teacher_out)
        after = loss_fn.center
        self.assertFalse(torch.allclose(before, after), "the centre did not move")
        self.assertLess((after - 5.0).abs().sum().item(),
                        (before - 5.0).abs().sum().item(),
                        "the centre did not move toward the batch mean")

    @needs_deps
    def test_the_head_l2_normalises_the_bottleneck(self):
        # Replace the last layer with identity so the head output IS the
        # bottleneck: it must be unit-norm, because forward L2-normalises it. The
        # small-init MLP produces a sub-unit bottleneck, so dropping the
        # normalisation is caught here (norms != 1).
        import torch
        import torch.nn as nn
        head = self.models().DINOHead(
            in_dim=EMBED_DIM, out_dim=DINO["out_dim"],
            hidden_dim=DINO["hidden_dim"], bottleneck_dim=DINO["bottleneck_dim"],
            norm_last_layer=DINO["norm_last_layer"])
        head.last_layer = nn.Identity()
        head.eval()
        with torch.no_grad():
            out = head(torch.randn(5, EMBED_DIM) * 7.0)
        norms = out.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            "the bottleneck fed to the last layer is not L2-normalised")

    @needs_deps
    def test_get_backbone_returns_the_cls_feature(self):
        import torch
        model = self._model(self.models())
        feats = model.get_backbone()(torch.randn(2, 3, IMG, IMG))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))


class TestTheSchedules(unittest.TestCase):
    def trainer(self):
        return load("dino_trainer", METHOD / "train_step1_dino.py")

    @needs_deps
    def test_teacher_momentum_rises_toward_one(self):
        t = self.trainer()
        start = t.cosine_schedule(0, 100, 0.996, 1.0)
        end = t.cosine_schedule(100, 100, 0.996, 1.0)
        self.assertAlmostEqual(start, 0.996, places=5)
        self.assertAlmostEqual(end, 1.0, places=5)
        self.assertLess(start, end)

    @needs_deps
    def test_the_lr_warms_up_from_zero(self):
        t = self.trainer()
        # Linear warmup from 0 over warmup_steps; at step 0 the LR is 0, and it
        # rises to base_lr by the end of warmup.
        lr0 = t.lr_schedule_value(0, 1000, base_lr=1.0, min_lr=0.0,
                                  warmup_steps=100)
        lr_end = t.lr_schedule_value(99, 1000, base_lr=1.0, min_lr=0.0,
                                     warmup_steps=100)
        self.assertAlmostEqual(lr0, 0.0, places=6)
        self.assertGreater(lr_end, lr0)


class TestTheDataset(Base):
    def dataset_mod(self):
        return load("dino_data", METHOD / "data" / "__init__.py")

    @needs_deps
    def test_an_item_is_a_crop_list_and_a_label(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        ds = self.dataset_mod().DINODataset(
            str(self.tmp / "data"), n_local_crops=N_LOCAL,
            global_size=IMG, local_size=LOCAL)
        crops, label = ds[0]
        self.assertEqual(len(crops), 2 + N_LOCAL)
        self.assertEqual(tuple(crops[0].shape), (3, IMG, IMG))
        self.assertEqual(tuple(crops[-1].shape), (3, LOCAL, LOCAL))
        self.assertFalse(torch.equal(crops[0], crops[1]),
                         "the two global views are identical, not augmented")

    @needs_deps
    def test_the_collate_groups_crops_by_view(self):
        import torch
        tiny_imagefolder(self.tmp / "data")
        d = self.dataset_mod()
        ds = d.DINODataset(str(self.tmp / "data"), n_local_crops=N_LOCAL,
                           global_size=IMG, local_size=LOCAL)
        batch = [ds[0], ds[1]]
        crops, labels = d.multicrop_collate(batch)
        self.assertEqual(len(crops), 2 + N_LOCAL)
        self.assertEqual(tuple(crops[0].shape), (2, 3, IMG, IMG))
        self.assertEqual(tuple(labels.shape), (2,))


class TestExtractingTheEncoder(unittest.TestCase):
    def test_only_the_teacher_backbone_comes_out(self):
        got = adapter.extract_encoder({
            "teacher.backbone.cls_token": 1,
            "teacher.backbone.blocks.0.norm1.weight": 2,
            "teacher.head.mlp.0.weight": 3,
            "student.backbone.cls_token": 4,
            "loss_fn.center": 5})
        self.assertEqual(set(got), {"cls_token", "blocks.0.norm1.weight"})

    def test_the_head_center_and_student_are_left_out(self):
        got = adapter.extract_encoder({"teacher.backbone.pos_embed": 1,
                                       "teacher.head.last_layer.weight_v": 2,
                                       "student.backbone.pos_embed": 3,
                                       "loss_fn.center": 4})
        self.assertEqual(set(got), {"pos_embed"})

    def test_nothing_matching_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"student.backbone.cls_token": 1})
        self.assertIn("encoder", str(e.exception).lower())


class TestConfigTranslation(Base):
    def test_step1_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["model"]["arch"], "vit_small")
        self.assertEqual(built["dino"]["out_dim"], 64)
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
        cfg = self.eval_config(train={"out_dim": 64})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("out_dim", str(e.exception))


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
        return load("dino_trainer", METHOD / "train_step1_dino.py")

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
        src = (METHOD / "train_step1_dino.py").read_text()
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
        tree = ast.parse((METHOD / "train_step1_dino.py").read_text())
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
