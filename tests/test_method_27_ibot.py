#!/usr/bin/env python3
"""Specification for the sixth method: 27_ibot (Zhou et al., 2021).

Chosen by the same measurement that ordered the other official-style ports
(Capture repository, DESIGN 5.40/5.41): the six candidates carrying
`*_official*` files, of which iBOT was the heaviest and so came last -- the one
constraint the now-available GPU removes. It shares the template of the three
ported before it: `setup_dist()` returns early when `LOCAL_RANK` is unset, and
there is no automatic mixed precision.

What is new here, rather than repeated:

- **The encoder is the teacher, and which one is read from the recipe, not the
  code.** The model exposes `get_encoder()` returning the student, but every
  official probe uses `--checkpoint_key teacher` and the paper reports the
  teacher (the EMA). So `encoder.pt` holds the teacher ViT, selected by the
  `teacher.` prefix (which does not match `teacher_head.`)
- **Two metrics have no contract slot.** The trainer reports the CLS and patch
  components of its loss; both are real and belong to no family, so they map to
  `None`: kept under their own names, kept out of the comparable block
- **The config is nested**, as the original groups it (`model`, `data`, `ibot`,
  `training`), rather than flattened. Every key the trainer reads is declared,
  so a run's temperature schedule or masking ratio is never a silent default
"""

from __future__ import annotations

import copy
import hashlib
import json
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
METHOD = ROOT / "methods" / "27_ibot"
BIN = ROOT / "bin"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

# The trainer and evaluator import SummaryWriter at module level, so without
# tensorboard the smoke tests would fail rather than skip. The condition names
# what the method actually needs.
try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import tensorboard                                 # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_torch = unittest.skipUnless(
    HAVE_DEPS, "this method needs torch, torchvision and tensorboard")


def load(name: str, path: Path):
    """Delegates to the shared helper: several methods define `data` and
    `models`, and whichever test file imports first would otherwise win."""
    return load_from(METHOD, name, path)


adapter = load("ibot_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run on a CPU in seconds, and every key each stage reads.
# `out_dim`/head dims shrink only the projection head; the ViT-Small backbone
# (embed_dim=384, depth=12) is the real one, because a port that shrank the
# architecture would not be a port. Tiny crops keep the sequence length short.
MODEL = {"arch": "vit_small", "patch_size": 16, "embed_dim": 384,
         "drop_path_rate": 0.0}
DATA = {"global_size": 32, "local_size": 16, "n_global_crops": 2,
        "n_local_crops": 1, "global_crops_scale": [0.5, 1.0],
        "local_crops_scale": [0.2, 0.5], "num_workers": 0}
IBOT = {"out_dim": 64, "head_hidden_dim": 64, "head_bottleneck_dim": 32,
        "head_nlayers": 1, "shared_head": True, "norm_last_layer": False,
        "student_temp": 0.1, "teacher_temp": 0.04, "teacher_patch_temp": 0.04,
        "teacher_temp_warmup": 0.04, "teacher_patch_temp_warmup": 0.04,
        "teacher_temp_warmup_epochs": 0, "lambda_token": 1.0,
        "pred_ratio": [0.0, 0.3], "pred_ratio_var": [0.0, 0.0],
        "pred_shape": "block", "pred_start_epoch": 0,
        "teacher_momentum_start": 0.996, "teacher_momentum_end": 1.0,
        "center_momentum": 0.9, "center_momentum_patch": 0.9}
TRAINING = {"epochs": 1, "batch_size": 2, "lr": 5.0e-4, "min_lr": 1.0e-6,
            "weight_decay_start": 0.04, "weight_decay_end": 0.4,
            "warmup_epochs": 0, "grad_clip": 3.0, "freeze_last_layer": 0,
            "checkpoint_health": {"min_total_loss": 0.0,
                                  "min_component_loss": 0.0,
                                  "max_total_loss": 1000000.0},
            "fail_fast_after_epoch": 1000, "print_freq": 1, "save_freq": 1}

EVAL_MODEL = {"arch": "vit_small", "patch_size": 16, "n_last_blocks": 4,
              "avgpool_patchtokens": 0}
EVAL = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1.0e-3}

# ── The unified ViT-B/16 Step-2 recipe (recipe: unified) ─────────────────────
# The capture's Step 2 plugs the same iBOT objective into the unified ViT-B/16
# backbone (arch vit_base, embed_dim 768). It differs from the native step-1
# recipe by: mask_ratio_min/max instead of pred_ratio/pred_ratio_var, a fixed
# weight_decay instead of a start/end cosine, a direct lr (no batch/256 rescale),
# grad_clip 0.3, freeze_last_layer 3, and milestone checkpoints (save_at_epochs).
# As with the native smoke, only the head and crops are shrunk -- the ViT-Base
# backbone (embed_dim=768, depth=12) is the real one.
UNIFIED_MODEL = {"arch": "vit_base", "patch_size": 16, "embed_dim": 768,
                 "drop_path_rate": 0.0}
UNIFIED_IBOT = {"out_dim": 64, "head_hidden_dim": 64, "head_bottleneck_dim": 32,
                "head_nlayers": 1, "shared_head": True, "norm_last_layer": True,
                "student_temp": 0.1, "teacher_temp": 0.04,
                "teacher_patch_temp": 0.04, "teacher_temp_warmup": 0.04,
                "teacher_patch_temp_warmup": 0.04,
                "teacher_temp_warmup_epochs": 0, "lambda_token": 1.0,
                "mask_ratio_min": 0.1, "mask_ratio_max": 0.5,
                "pred_shape": "block",
                "teacher_momentum_start": 0.996, "teacher_momentum_end": 1.0,
                "center_momentum": 0.9, "center_momentum_patch": 0.9}
UNIFIED_TRAINING = {"epochs": 1, "batch_size": 2, "lr": 6.0e-4, "min_lr": 1.0e-6,
                    "weight_decay": 0.05, "warmup_epochs": 0, "grad_clip": 0.3,
                    "freeze_last_layer": 0, "save_at_epochs": [1],
                    "print_freq": 1}
UNIFIED_EVAL_MODEL = {"arch": "vit_base", "patch_size": 16, "n_last_blocks": 4,
                      "avgpool_patchtokens": 0}


def tiny_imagenet(root: Path, n: int = 4) -> Path:
    """A few synthetic images in the layout `ImageFolder` walks.

    `iBOTDataset` wraps `ImageFolder`, so this exercises the real multi-crop
    and masking code without ImageNet. `n` must be at least the batch size: the
    loader uses `drop_last=True`, so a smaller set yields no batches and the
    training loop would run zero times while everything still looked fine.
    """
    from PIL import Image
    import random
    rng = random.Random(0)
    d = root / "train" / "class0"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64))
        img.putdata([(rng.randrange(256), rng.randrange(256),
                      rng.randrange(256)) for _ in range(64 * 64)])
        img.save(d / f"{i}.jpg")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ibot-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "model": copy.deepcopy(MODEL), "data": copy.deepcopy(DATA),
               "ibot": copy.deepcopy(IBOT), "training": copy.deepcopy(TRAINING)}
        for k, v in over.items():
            if k in ("model", "data", "ibot", "training") and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        return cfg

    def run_adapter(self, cfg: dict | None = None):
        import os
        cfg_path = self.tmp / "resolved.json"
        cfg_path.write_text(
            json.dumps(cfg if cfg is not None else self.config()),
            encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg_path),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        return cfg_path, r


class TestTheConfigIsTranslated(Base):
    """The contract's config is nested exactly as the original groups it, and
    declares only what affects the result. This is where the two meet."""

    def test_a_config_naming_an_output_location_is_refused(self):
        for key in ("checkpoint", "output"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(**{key: {"x": 1}}),
                                          self.out)
                # Naming the key is not enough: the generic unknown-key rule
                # also refuses it. The assertion is on the reason only this
                # check gives, so a mutation removing it is caught.
                self.assertIn("--out", str(e.exception))

    def test_every_key_the_stage_reads_must_be_declared(self):
        cfg = self.config()
        del cfg["training"]["epochs"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("epochs", str(e.exception))

    def test_a_key_the_stage_never_reads_is_refused(self):
        cfg = self.config()
        cfg["ibot"]["mystery"] = 1
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("mystery", str(e.exception))

    def test_a_missing_top_level_key_is_refused(self):
        cfg = self.config()
        del cfg["seed"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("seed", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(stage="step2"), self.out)
        self.assertIn("step2", str(e.exception))

    def test_an_unknown_device_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(device="tpu"), self.out)
        self.assertIn("tpu", str(e.exception))

    def test_a_block_must_be_a_mapping(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(model="not a dict"), self.out)

    def test_the_health_block_is_checked_by_name(self):
        cfg = self.config()
        del cfg["training"]["checkpoint_health"]["max_total_loss"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("max_total_loss", str(e.exception))

    def test_an_unknown_arch_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(model={"arch": "vit_base"}),
                                  self.out)
        self.assertIn("vit_base", str(e.exception))

    def test_embed_dim_must_match_the_arch(self):
        """A config whose embed_dim disagrees with its architecture would
        misdescribe the run."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(model={"embed_dim": 768}),
                                  self.out)
        self.assertIn("embed_dim", str(e.exception))

    def test_a_malformed_scale_range_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.config(data={"global_crops_scale": [0.5]}), self.out)
        self.assertIn("global_crops_scale", str(e.exception))

    def test_the_output_goes_under_out(self):
        run = adapter.to_run_config(self.config(), self.out)
        got = Path(run["checkpoint"]["save_dir"])
        self.assertTrue(str(got).startswith(str(self.out)),
                        f"{got} escapes {self.out}")
        self.assertTrue(run["data"]["train_path"].startswith(
            str(self.tmp / "data")))

    def test_the_learning_rate_is_passed_through(self):
        """The batch-scaled LR rule is the original's; the adapter passes the
        base lr and batch size and does not recompute it here."""
        run = adapter.to_run_config(self.config(), self.out)
        self.assertEqual(run["training"]["lr"], TRAINING["lr"])
        self.assertEqual(run["training"]["batch_size"], TRAINING["batch_size"])


class TestWhichCheckpointIsTheFinalOne(Base):
    def make(self, *epochs: int) -> Path:
        work = self.tmp / "work"
        work.mkdir(parents=True, exist_ok=True)
        for e in epochs:
            (work / f"checkpoint_epoch_{e}.pth").write_bytes(b"x")
        return work

    def test_the_highest_epoch_wins_not_the_last_name(self):
        work = self.make(1, 9, 10)
        self.assertEqual(adapter.latest_checkpoint(work).name,
                         "checkpoint_epoch_10.pth")

    def test_no_checkpoint_at_all_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.latest_checkpoint(self.make())
        self.assertIn("checkpoint", str(e.exception))


@needs_torch
class TestTheDeviceIsResolved(Base):
    """The captured trainer called `.cuda(local_rank)` unconditionally, so it
    could not start without a GPU. Patched rather than skipped, so the decision
    is checked on every machine instead of only on one without a GPU."""

    def trainer(self):
        return load("ibot_trainer", METHOD / "train_pretrain.py")

    def test_cpu_is_honoured(self):
        self.assertEqual(self.trainer().resolve_device("cpu").type, "cpu")

    def test_asking_for_cuda_without_one_is_an_error(self):
        import torch
        t = self.trainer()
        real = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            with self.assertRaises(RuntimeError) as e:
                t.resolve_device("cuda")
            self.assertIn("cuda", str(e.exception).lower())
        finally:
            torch.cuda.is_available = real

    def test_auto_takes_the_cpu_when_there_is_no_gpu(self):
        import torch
        t = self.trainer()
        real = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            self.assertEqual(t.resolve_device("auto").type, "cpu")
        finally:
            torch.cuda.is_available = real

    def test_auto_takes_the_gpu_when_there_is_one(self):
        import torch
        t = self.trainer()
        real = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        try:
            self.assertEqual(t.resolve_device("auto").type, "cuda")
        finally:
            torch.cuda.is_available = real

    def test_an_unknown_device_is_refused(self):
        with self.assertRaises(ValueError):
            self.trainer().resolve_device("tpu")


class TestTheEncoderIsTheTeacher(Base):
    """Which backbone is the encoder is read from the original's recipe: the
    official probe evaluates the teacher, so that is what `encoder.pt` holds.
    The student, both heads and the centering buffers are training machinery.
    """

    def test_only_the_teacher_comes_across_prefix_stripped(self):
        state = {"teacher.blocks.0.weight": 1, "student.blocks.0.weight": 2,
                 "head.mlp.0.weight": 3, "teacher_head.mlp.0.weight": 4}
        got = adapter.extract_encoder(state)
        self.assertEqual(set(got), {"blocks.0.weight"})

    def test_the_teacher_head_is_not_mistaken_for_the_teacher(self):
        """`teacher_head.` starts with `teacher` but not with `teacher.`, so
        the head must not be swept into the encoder."""
        got = adapter.extract_encoder({"teacher.x": 1, "teacher_head.y": 2})
        self.assertEqual(set(got), {"x"})

    def test_an_empty_result_is_an_error_not_an_empty_file(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"something.else": 1})
        self.assertIn("teacher", str(e.exception))

    def test_a_ddp_prefix_is_stripped(self):
        got = adapter.extract_encoder({"module.teacher.blocks.0.weight": 1})
        self.assertEqual(set(got), {"blocks.0.weight"})


class TestTheMetricNames(Base):
    def test_every_mapped_name_is_in_the_contract_vocabulary(self):
        for raw, target in adapter.PRETRAIN_METRIC_NAMES.items():
            if target is None:
                continue
            with self.subTest(metric=raw):
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_the_loss_is_a_pretext_number(self):
        self.assertEqual(
            adapterlib.METRIC_VOCABULARY[
                adapter.PRETRAIN_METRIC_NAMES["final_loss"]],
            adapterlib.PER_METHOD)

    def test_the_loss_components_are_kept_but_given_no_slot(self):
        """The CLS and patch components are real numbers with no family; they
        stay under their own names and out of the comparable block."""
        for raw in ("final_cls_loss", "final_patch_loss"):
            with self.subTest(metric=raw):
                self.assertIn(raw, adapter.PRETRAIN_METRIC_NAMES)
                self.assertIsNone(adapter.PRETRAIN_METRIC_NAMES[raw])


class TestTheTrainingCallIsTheOriginals(Base):
    def test_the_original_run_is_called_once(self):
        calls = []

        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            calls.append((args, config))
            return {"epochs": 1, "final_loss": 4.0, "final_cls_loss": 2.0,
                    "final_patch_loss": 2.0}

        adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(len(calls), 1)

    def test_the_metrics_come_from_that_call(self):
        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            return {"epochs": 1, "final_loss": 4.0, "final_cls_loss": 2.0,
                    "final_patch_loss": 2.0}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(m["final_loss"], 4.0)
        self.assertNotIn("metrics_unavailable", m)

    def test_a_run_that_reports_nothing_is_flagged(self):
        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            return None

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)
        self.assertNotIn("final_loss", m)

    def test_a_non_numeric_loss_is_counted_not_written(self):
        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            return {"epochs": 1, "final_loss": "low"}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertNotIn("final_loss", m)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)


@needs_torch
class TestASmokeRun(Base):
    """A real training run, on a CPU, through the whole contract chain."""

    def setUp(self) -> None:
        super().setUp()
        tiny_imagenet(self.tmp / "data")

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        """The GPU path, on real hardware -- the case CPU-only testing could
        never reach. A tensor left on the wrong device raises inside training,
        so a run that finishes with a non-empty encoder is the GPU path working
        end to end."""
        _, r = self.run_adapter(self.config(device="cuda"))
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty after a CUDA run")

    def test_it_completes_and_satisfies_the_contract(self):
        cfg_path, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg_path), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_it_runs_on_the_cpu(self):
        """The reason the GPU only removed a compute limit, not a blocker: the
        captured trainer sent the model and every crop to CUDA unconditionally.
        The port resolves a device instead."""
        _, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["status"], "ok")

    def test_the_encoder_is_the_teacher_backbone(self):
        """`encoder.pt` must be *exactly* a teacher ViT state_dict.

        Checked against a freshly built teacher (`use_mask_token=False`): set
        equality catches both a head or student weight leaking in and a real
        backbone weight going missing. And the student's `mask_token` -- present
        only when `use_mask_token=True` -- must be absent, which is what
        distinguishes the teacher backbone from the student. A prefix denylist
        would not fire here, because `extract_encoder` has already stripped the
        `teacher.` prefix down to bare ViT names.
        """
        import torch
        self.run_adapter()
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(state, "the encoder is empty")
        models = load("this_methods_models", METHOD / "models" / "__init__.py")
        teacher_keys = set(models.vit_small(
            patch_size=MODEL["patch_size"], use_mask_token=False).state_dict())
        self.assertEqual(set(state), teacher_keys,
                         "encoder.pt is not exactly a teacher ViT backbone")
        self.assertNotIn(
            "mask_token", state,
            "mask_token present -- this is the student, not the teacher")

    def test_the_metrics_use_the_contract_names(self):
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_pretext_loss", doc["metrics"])
        self.assertIn("epochs_completed", doc["metrics"])
        for k, v in doc["metrics"].items():
            with self.subTest(metric=k):
                self.assertIsInstance(v, (int, float))
                self.assertNotIsInstance(v, bool)

    def test_the_loss_components_survive_under_their_own_names(self):
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        for raw in ("final_cls_loss", "final_patch_loss"):
            self.assertIn(raw, doc["metrics_raw"])
            self.assertNotIn(raw, doc["metrics"])

    def test_no_probe_name_appears(self):
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertFalse([k for k in doc["metrics"] if "linear_probe" in k])

    def test_the_originals_scratch_files_stay_inside_out(self):
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        listed = sorted(a["path"] for a in man["artifacts"])
        on_disk = sorted(str(p.relative_to(self.out))
                         for p in self.out.rglob("*") if p.is_file()
                         and p.name != "run_manifest.json")
        self.assertEqual(listed, on_disk)
        self.assertTrue([p for p in on_disk if p.startswith("work/")],
                        "the original wrote nothing of its own")

    def test_the_same_config_twice_gives_the_same_encoder(self):
        """The guarantee the whole project rests on, for this method. iBOT's
        multi-crop and block masking draw random numbers every step, so this
        fails unless every source is seeded -- the change the port made."""
        digests = []
        for i in range(2):
            self.out = self.tmp / f"out{i}"
            _, r = self.run_adapter()
            self.assertEqual(r.returncode, 0,
                             r.stdout[-3000:] + r.stderr[-3000:])
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1],
                         "two runs of one config produced different weights")

    @needs_torch
    def test_the_encoder_pt_it_wrote_loads_back(self):
        """The round trip, end to end. Weights are compared, not just the
        absence of an exception: loading into a fresh model and reading back
        default values would satisfy a check that only asked whether it
        raised."""
        import torch
        self.run_adapter()
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty")
        load("this_methods_models", METHOD / "models" / "__init__.py")
        loaded = adapter.load_encoder(saved, self.config()).state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


def tiny_classified(root: Path, classes: int = 2, per_class: int = 2) -> Path:
    """A labelled ImageFolder tree, with the classes separable.

    Separable because the evaluation saves its classifier only when accuracy
    improves on zero, and on pure noise which side of the boundary a few images
    fall on is decided by floating-point detail this project states is not
    reproducible across hardware. A test resting on that is not flaky by
    accident.
    """
    from PIL import Image
    import random
    rng = random.Random(0)
    for split in ("train", "val"):
        for c in range(classes):
            d = root / split / f"c{c}"
            d.mkdir(parents=True, exist_ok=True)
            base = int(30 + c * (200 / max(classes - 1, 1)))
            for i in range(per_class):
                img = Image.new("RGB", (64, 64))
                img.putdata([(min(255, max(0, base + rng.randrange(-8, 9))),) * 3
                             for _ in range(64 * 64)])
                img.save(d / f"{i}.jpg")
    return root


class TestTheLinearEvaluationStage(Base):
    """The second stage, producing the numbers the contract says may be
    compared across methods."""

    def config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "encoder": str(self.tmp / "encoder.pt"), "device": "cpu",
               "model": copy.deepcopy(EVAL_MODEL), "eval": copy.deepcopy(EVAL)}
        for k, v in over.items():
            if k in ("model", "eval") and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        return cfg

    def test_the_stage_is_known(self):
        self.assertIn("linear_eval", adapter.STAGES)

    def test_it_needs_an_encoder_to_evaluate(self):
        cfg = self.config()
        del cfg["encoder"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("encoder", str(e.exception))

    def test_a_step1_only_key_is_refused_here(self):
        cfg = self.config()
        cfg["eval"]["momentum"] = 0.9
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("momentum", str(e.exception))

    def test_no_pretext_name_is_produced_here(self):
        for raw, target in adapter.LINEAR_EVAL_METRIC_NAMES.items():
            with self.subTest(metric=raw):
                self.assertNotIn("pretext", str(target))

    def test_its_accuracies_are_comparable_ones(self):
        probes = [t for t in adapter.LINEAR_EVAL_METRIC_NAMES.values()
                  if t and "accuracy" in t]
        self.assertTrue(probes)
        for target in probes:
            with self.subTest(metric=target):
                self.assertIn("linear_probe", target)
                self.assertEqual(adapterlib.METRIC_VOCABULARY[target],
                                 adapterlib.COMPARABLE)

    def test_it_reports_all_four_the_original_produces(self):
        """iBOT's evaluation reports a best top-5 as well, so all four
        comparable slots are filled -- unlike the SimSiam port's three."""
        accuracies = sorted(
            k for k, t in adapter.LINEAR_EVAL_METRIC_NAMES.items()
            if t and "accuracy" in t)
        self.assertEqual(accuracies, ["best_top1_acc",
                                      "best_top5_acc_at_best_top1",
                                      "final_top1_acc", "final_top5_acc"])

    @needs_torch
    def test_an_encoder_missing_its_backbone_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.load_encoder({}, self.config())
        self.assertIn("teacher", str(e.exception).lower())

    @needs_torch
    def test_the_evaluation_refuses_a_gpu_it_does_not_have(self):
        """Replacing the requested device with `auto` changes nothing without a
        GPU, so nothing else catches it -- and the failure it hides is a run
        asked for CUDA that quietly used a CPU and reported success."""
        import torch
        evaluation = load("ibot_eval", METHOD / "evaluate_linear.py")
        args = adapter.eval_args(self.config(device="cuda"), self.out)
        real = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            with self.assertRaises(RuntimeError) as e:
                evaluation.run(args, encoder=object(), in_dim=8)
            self.assertIn("cuda", str(e.exception).lower())
        finally:
            torch.cuda.is_available = real

    def test_the_device_in_the_config_reaches_the_evaluation(self):
        args = adapter.eval_args(self.config(device="cuda"), self.out)
        self.assertEqual(args.device, "cuda")

    def test_the_teacher_is_the_probed_backbone(self):
        """The official recipe probes the teacher; the adapter fixes that
        rather than exposing it, so a run cannot quietly probe the student."""
        args = adapter.eval_args(self.config(), self.out)
        self.assertEqual(args.checkpoint_key, "teacher")


@needs_torch
class TestTheLinearEvaluationRuns(Base):
    def setUp(self) -> None:
        super().setUp()
        tiny_classified(self.tmp / "data")

    def make_encoder(self) -> None:
        """A real `encoder.pt`, produced by this method's own first stage."""
        tiny_imagenet(self.tmp / "data")
        first = TestASmokeRun("test_it_runs_on_the_cpu")
        first.tmp, first.out = self.tmp, self.tmp / "step1out"
        _, r = first.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        (self.tmp / "encoder.pt").write_bytes(
            (first.out / "encoder.pt").read_bytes())

    def config(self, **over) -> dict:
        return TestTheLinearEvaluationStage.config(self, **over)

    def test_it_completes_and_satisfies_the_contract(self):
        self.make_encoder()
        cfg_path, r = self.run_adapter(self.config())
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg_path), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_the_numbers_are_comparable_ones(self):
        self.make_encoder()
        self.run_adapter(self.config())
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_linear_probe_top1_accuracy", doc["metrics"])
        self.assertFalse([k for k in doc["metrics"] if "pretext" in k],
                         "a pretext name reached a downstream stage")
        self.assertIn("final_top1_acc", doc["metrics_raw"])

    def test_it_says_why_there_is_no_encoder_to_hand_on(self):
        self.make_encoder()
        self.run_adapter(self.config())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertFalse((self.out / "encoder.pt").exists())
        self.assertTrue(man["encoder_absent_reason"].strip())
        self.assertEqual(man["stage"], "linear_eval")


class UnifiedBase(Base):
    """The unified ViT-B/16 Step-2 pretrain config (recipe: unified). The recipe
    is a top-level key; absent means the native step-1 path."""

    def config(self, **over) -> dict:
        cfg = {"recipe": "unified", "stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "model": copy.deepcopy(UNIFIED_MODEL),
               "data": copy.deepcopy(DATA), "ibot": copy.deepcopy(UNIFIED_IBOT),
               "training": copy.deepcopy(UNIFIED_TRAINING)}
        for k, v in over.items():
            if k in ("model", "data", "ibot", "training") and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        return cfg


class TestTheUnifiedConfigIsTranslated(UnifiedBase):
    """The additive Step-2 branch: selected by recipe: unified, validated against
    its own key sets, disjoint from the native ones so nothing leaks."""

    def test_the_unified_recipe_is_accepted(self):
        built = adapter.to_run_config(self.config(), self.out)
        self.assertEqual(built["training"]["weight_decay"], 0.05)
        self.assertEqual(built["ibot"]["mask_ratio_min"], 0.1)
        self.assertEqual(built["ibot"]["mask_ratio_max"], 0.5)
        self.assertEqual(built["training"]["save_at_epochs"], [1])
        # `recipe` is consumed by the selector, not passed to the trainer.
        self.assertNotIn("recipe", built)

    def test_the_native_recipe_is_still_accepted(self):
        """Regression: a config with no recipe is the native step-1 path."""
        native = Base.config(self)
        self.assertNotIn("recipe", native)
        built = adapter.to_run_config(native, self.out)
        self.assertEqual(built["model"]["arch"], "vit_small")
        self.assertIn("weight_decay_start", built["training"])

    def test_the_unified_recipe_needs_vit_base(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.config(model={"arch": "vit_small", "embed_dim": 384}),
                self.out)
        self.assertIn("vit_small", str(e.exception))

    def test_the_unified_embed_dim_must_match_vit_base(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(model={"embed_dim": 384}),
                                  self.out)
        self.assertIn("embed_dim", str(e.exception))

    def test_a_missing_unified_key_is_refused_by_name(self):
        for block, key in (("training", "weight_decay"),
                           ("training", "save_at_epochs"),
                           ("ibot", "mask_ratio_min"),
                           ("ibot", "mask_ratio_max")):
            with self.subTest(key=key):
                cfg = self.config()
                del cfg[block][key]
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_unified_key_is_refused(self):
        cfg = self.config()
        cfg["training"]["mystery"] = 1
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("mystery", str(e.exception))

    def test_a_native_only_key_on_the_unified_path_is_refused(self):
        """Leakage one way: the native cosine-WD and masking knobs are unknown
        to the unified recipe."""
        for block, key, value in (("training", "weight_decay_start", 0.04),
                                  ("training", "weight_decay_end", 0.4),
                                  ("ibot", "pred_ratio", [0.0, 0.3]),
                                  ("ibot", "pred_ratio_var", [0.0, 0.2])):
            with self.subTest(key=key):
                cfg = self.config()
                cfg[block][key] = value
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, self.out)
                self.assertIn(key, str(e.exception))

    def test_a_unified_only_key_on_the_native_path_is_refused(self):
        """Leakage the other way: the unified fixed-WD and mask-ratio knobs are
        unknown to the native step-1 recipe."""
        for block, key, value in (("training", "weight_decay", 0.05),
                                  ("training", "save_at_epochs", [1]),
                                  ("ibot", "mask_ratio_min", 0.1),
                                  ("ibot", "mask_ratio_max", 0.5)):
            with self.subTest(key=key):
                cfg = Base.config(self)
                cfg[block][key] = value
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, self.out)
                self.assertIn(key, str(e.exception))

    def test_a_bad_recipe_value_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(recipe="turbo"), self.out)
        self.assertIn("turbo", str(e.exception))

    def test_a_malformed_scale_range_is_still_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.config(data={"global_crops_scale": 0.5}), self.out)
        self.assertIn("global_crops_scale", str(e.exception))


class TestUnifiedArchWidening(Base):
    @needs_torch
    def test_load_encoder_builds_a_vit_base_and_round_trips(self):
        import torch
        models = load("this_methods_models", METHOD / "models" / "__init__.py")
        teacher = models.vit_base(patch_size=UNIFIED_MODEL["patch_size"],
                                  use_mask_token=False)
        saved = teacher.state_dict()
        cfg = UnifiedBase.config(self)
        loaded = adapter.load_encoder(saved, cfg).state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")

    def test_the_evaluation_accepts_a_vit_base_encoder(self):
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "encoder": str(self.tmp / "encoder.pt"), "device": "cpu",
               "model": copy.deepcopy(UNIFIED_EVAL_MODEL),
               "eval": copy.deepcopy(EVAL)}
        args = adapter.eval_args(cfg, self.out)
        self.assertEqual(args.model_type, "vit_base")


@needs_torch
class TestTheUnifiedSmokeRuns(UnifiedBase):
    """A real unified ViT-B/16 run, on a CPU, through the whole contract chain."""

    def setUp(self) -> None:
        super().setUp()
        tiny_imagenet(self.tmp / "data")

    def test_it_completes_and_satisfies_the_contract(self):
        cfg_path, r = self.run_adapter(self.config())
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg_path), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_the_encoder_is_a_vit_base_teacher_backbone(self):
        import torch
        self.run_adapter(self.config())
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(state, "the encoder is empty")
        models = load("this_methods_models", METHOD / "models" / "__init__.py")
        teacher_keys = set(models.vit_base(
            patch_size=UNIFIED_MODEL["patch_size"],
            use_mask_token=False).state_dict())
        self.assertEqual(set(state), teacher_keys,
                         "encoder.pt is not exactly a ViT-Base teacher backbone")
        self.assertNotIn("mask_token", state,
                         "mask_token present -- this is the student, not teacher")

    def test_each_milestone_encoder_is_written(self):
        """save_at_epochs writes checkpoint_epoch_{N}.pth per milestone; the
        adapter hands over encoder_epoch{N}.pt for each so the 100/200/300 sweep
        can probe every frozen backbone."""
        _, r = self.run_adapter(self.config(training={"epochs": 2,
                                                      "save_at_epochs": [1, 2]}))
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((self.out / "encoder.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch1.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch2.pt").is_file())


@needs_torch
class TestTheUnifiedLinearEvaluationRuns(Base):
    def setUp(self) -> None:
        super().setUp()
        tiny_classified(self.tmp / "data")

    def make_encoder(self) -> None:
        tiny_imagenet(self.tmp / "data")
        first = TestTheUnifiedSmokeRuns("test_it_completes_and_satisfies_the_contract")
        first.tmp, first.out = self.tmp, self.tmp / "step2out"
        _, r = first.run_adapter(UnifiedBase.config(first))
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        (self.tmp / "encoder.pt").write_bytes(
            (first.out / "encoder.pt").read_bytes())

    def config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "encoder": str(self.tmp / "encoder.pt"), "device": "cpu",
               "model": copy.deepcopy(UNIFIED_EVAL_MODEL),
               "eval": copy.deepcopy(EVAL)}
        for k, v in over.items():
            if k in ("model", "eval") and isinstance(v, dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        return cfg

    def test_it_completes_and_satisfies_the_contract(self):
        self.make_encoder()
        cfg_path, r = self.run_adapter(self.config())
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg_path), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_the_numbers_are_comparable_ones(self):
        self.make_encoder()
        self.run_adapter(self.config())
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_linear_probe_top1_accuracy", doc["metrics"])
        self.assertFalse((self.out / "encoder.pt").exists())


class TestWhatCameFromTheCapture(unittest.TestCase):
    def test_the_captured_files_are_unchanged(self):
        expected = json.loads(
            (METHOD / "provenance.json").read_text())["captured_sha256"]
        self.assertTrue(expected, "nothing is pinned")
        for rel, want in sorted(expected.items()):
            with self.subTest(file=rel):
                got = hashlib.sha256((METHOD / rel).read_bytes()).hexdigest()
                self.assertEqual(got, want, f"{rel} differs from the capture")

    def test_the_model_and_the_loader_are_among_them(self):
        expected = json.loads(
            (METHOD / "provenance.json").read_text())["captured_sha256"]
        for rel in ("models/ibot.py", "models/vision_transformer.py",
                    "data/multicrop.py"):
            self.assertIn(rel, expected)

    def test_what_was_rewritten_is_recorded(self):
        doc = json.loads((METHOD / "provenance.json").read_text())
        self.assertIn("train_pretrain.py", doc["rewritten_during_the_port"])
        self.assertIn("evaluate_linear.py", doc["rewritten_during_the_port"])

    def test_the_unified_step2_files_are_recorded(self):
        """The additive Step-2 port authored three files; provenance names each
        so the port is fully documented."""
        doc = json.loads((METHOD / "provenance.json").read_text())
        for rel in ("train_pretrain_vit_ibot.py", "configs/pretrain_vit.yaml",
                    "configs/linear_eval_vit.yaml"):
            self.assertIn(rel, doc["rewritten_during_the_port"])


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. It reuses this method's
    own encoder loader (`adapter.load_encoder`) and eval extraction
    (`evaluate_linear.extract_features`), so the check is that it returns the
    teacher ViT's frozen feature -- raw, before the probe's normalise -- one
    row per val image, with honest meta.

    The shipped linear_eval config is vit_small with n_last_blocks=1,
    avgpool_patchtokens=0: the feature is the [CLS] token of the single final
    block, so feat_dim is embed_dim (384, measured) -- one canonical final
    layer, per BASIC5_FAIR_v1 rule d (no multi-layer concatenation). This port
    follows the same single-feature policy as 23_dino; iBOT's official last-4
    concatenation would be 1536 (384 x 4), so the width assertion pins that the
    port does NOT concatenate.

    The encoder.pt is built from the shipped config's architecture (the
    provider reads that config), via the same `extract_encoder` filter the
    adapter writes with -- a teacher ViT state_dict under the `teacher.` prefix,
    stripped back to bare ViT names; random weights do not affect the
    shape-and-plumbing this proves.
    """

    # embed_dim 384 (vit_small, measured) x n_last_blocks 1 (shipped config):
    # one canonical final-block CLS, not the last-4 concat (which would be 1536)
    FEATURE_DIM = 384 * 1

    def setUp(self) -> None:
        super().setUp()
        tiny_classified(self.tmp / "data")

    def _shipped_config(self) -> dict:
        import yaml
        return yaml.safe_load(
            (METHOD / "configs" / "linear_eval.yaml").read_text())

    def _make_encoder(self, cfg: dict) -> Path:
        """A teacher ViT `encoder.pt`, produced through the adapter's own
        `extract_encoder` filter: build the backbone the shipped config names,
        put it under the `teacher.` prefix as a real iBOT checkpoint does, then
        let `extract_encoder` lift it back out to a bare ViT state_dict."""
        import torch
        models = load("this_methods_models", METHOD / "models" / "__init__.py")
        model = cfg["model"]
        builder = {"vit_small": models.vit_small,
                   "vit_base": models.vit_base}[model["arch"]]
        teacher = builder(patch_size=int(model["patch_size"]),
                          use_mask_token=False)
        full = {f"teacher.{k}": v for k, v in teacher.state_dict().items()}
        state = adapter.extract_encoder(full)
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("ibot_feature_provider", METHOD / "feature_provider.py")

    @needs_torch
    def test_it_returns_raw_384d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("27_ibot provider not yet present")
        import numpy as np
        cfg = self._shipped_config()
        encoder_pt = self._make_encoder(cfg)

        prov = self._provider()
        feats, labels, meta = prov.extract_val_features(
            encoder_path=str(encoder_pt), data_root=str(self.tmp / "data"),
            split="val", device="cpu", batch_size=2, num_workers=0)

        feats = np.asarray(feats)
        self.assertEqual(feats.ndim, 2)
        self.assertEqual(feats.shape[0], 4, "4 val images expected")
        self.assertEqual(feats.shape[1], self.FEATURE_DIM,
                         "iBOT single canonical final-block CLS is 384-d "
                         "(embed_dim x 1); a last-4 concat would be 1536")
        self.assertEqual(np.asarray(labels).shape[0], 4)
        self.assertEqual(meta["feat_dim"], self.FEATURE_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_torch
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("27_ibot provider not yet present")
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        encoder_pt = self._make_encoder(self._shipped_config())

        record = {"method": METHOD.name, "status": "ready",
                  "provider": str(prov_path), "encoder": str(encoder_pt)}
        out = self.tmp / "features"
        updated = driver.extract_one(
            record, data_root=str(self.tmp / "data"), split="val", out=out,
            device="cpu", batch_size=2, num_workers=0)

        self.assertEqual(updated["status"], "ok", updated.get("reason", ""))
        method_out = out / METHOD.name
        feats = np.load(method_out / "features.npy")
        labels = np.load(method_out / "labels.npy")
        meta = json.loads((method_out / "meta.json").read_text())
        self.assertEqual(feats.shape, (4, self.FEATURE_DIM))
        self.assertEqual(labels.shape[0], 4)
        self.assertEqual(meta["encoder_sha256"],
                         hashlib.sha256(encoder_pt.read_bytes()).hexdigest())

    @needs_torch
    def test_the_isolated_driver_run_extracts_this_method_end_to_end(self):
        """The whole driver, real subprocess, real provider -- catches the
        class of regression where the isolated worker cannot see a
        repository-root module the provider needs (the worker puts ROOT on
        sys.path, as bin/launch.py sets PYTHONPATH=ROOT)."""
        if not (METHOD / "feature_provider.py").is_file():
            self.skipTest("27_ibot provider not yet present")
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        encoder_pt = self._make_encoder(self._shipped_config())
        out = self.tmp / "features"
        manifest = driver.run(
            METHOD.parent, data_root=str(self.tmp / "data"), split="val",
            out=out, encoders={METHOD.name: str(encoder_pt)},
            encoders_root=None, device="cpu", batch_size=2, num_workers=0,
            venvs_root=ROOT / ".venvs")

        rec = {r["method"]: r for r in manifest["records"]}[METHOD.name]
        self.assertEqual(rec["status"], "ok", rec.get("reason", ""))
        feats = np.load(out / METHOD.name / "features.npy")
        self.assertEqual(feats.shape, (4, self.FEATURE_DIM))


if __name__ == "__main__":
    unittest.main()
