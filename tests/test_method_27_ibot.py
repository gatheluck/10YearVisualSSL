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


if __name__ == "__main__":
    unittest.main()
