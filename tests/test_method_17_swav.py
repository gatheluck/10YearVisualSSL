#!/usr/bin/env python3
"""Specification for the fifth method: 17_swav (Caron et al., 2020).

Instead of comparing two views directly, SwAV assigns each view to a set of
learned prototypes and makes one view predict the other's assignment. The
assignment is computed by a Sinkhorn-Knopp normalisation that keeps the
prototypes evenly used, which is what stops the representation collapsing.

Two things here are new, and both are about the shape of a configuration
rather than about the training.

**Multi-crop settings are lists, and they have to agree.** A run is described
by four parallel lists -- the crop sizes, how many of each, and the scale
bounds -- and the loader asserts they are the same length. A config whose
lists disagree is not a run at all, so it is refused here rather than reaching
an assertion three frames down.

**Most of its settings are optional in the original.** The trainer reads a
dozen keys with `cfg.get(...)` and a default. Leaving them out would let a
resolved config claim a run whose Sinkhorn epsilon, warmup and prototype
freezing are whatever that version of the code happened to default to, which
is exactly what "the resolved config has to say what ran" is against. They are
all declared.
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
METHOD = ROOT / "methods" / "17_swav"
BIN = ROOT / "bin"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import tensorboard                                 # noqa: F401
    import PIL                                         # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_torch = unittest.skipUnless(
    HAVE_DEPS, "this method needs torch, torchvision, tensorboard and Pillow")


def load(name: str, path: Path):
    """Five methods now define `data` and `models`."""
    return load_from(METHOD, name, path)


adapter = load("swav_adapter", METHOD / "adapter" / "__init__.py")

# Two crop sizes, as the paper's multi-crop uses, but tiny.
TRAIN = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.6,
         "final_lr": 0.0006, "momentum": 0.9, "weight_decay": 1.0e-6,
         "warmup_epochs": 0, "start_warmup": 0.0, "eta": 0.001,
         "larc_clip": True, "freeze_prototypes_steps": 0,
         "size_crops": [32, 16], "nmb_crops": [2, 2],
         "min_scale_crops": [0.14, 0.05], "max_scale_crops": [1.0, 0.14],
         "color_jitter_strength": 1.0, "temperature": 0.1,
         "sinkhorn_eps": 0.05, "sinkhorn_iters": 3,
         "out_dim": 32, "hidden_mlp": 64, "nmb_prototypes": 8,
         "save_freq": 1, "print_freq": 1}


def tiny_imagenet(root: Path, n: int = 4) -> Path:
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
        self.tmp = Path(tempfile.mkdtemp(prefix="swav-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
            else:
                cfg[k] = v
        return cfg

    def run_adapter(self, cfg: dict | None = None):
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


class TestTheMultiCropListsMustAgree(Base):
    """**The new problem this port had to solve.**

    Four parallel lists describe the crops. The loader asserts they are the
    same length, which arrives as a bare `AssertionError` from inside the
    dataset with nothing to say which config was wrong. A set of lists that do
    not line up is not a run, so it is refused here, by name.
    """

    CROP_KEYS = ("size_crops", "nmb_crops", "min_scale_crops",
                 "max_scale_crops")

    def test_lists_of_equal_length_are_accepted(self):
        adapter.to_run_config(self.config(), self.out)

    def test_a_shorter_list_is_refused_and_named(self):
        for key in self.CROP_KEYS:
            with self.subTest(key=key):
                short = list(TRAIN[key])[:-1]
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(train={key: short}),
                                          self.out)
                msg = str(e.exception)
                self.assertIn(key, msg)
                self.assertIn("same length", msg)

    def test_an_empty_crop_list_is_refused(self):
        """Zero crops is a run that trains on nothing, and the loader would
        simply yield no batches."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.config(train={k: [] for k in self.CROP_KEYS}), self.out)
        self.assertIn("at least one", str(e.exception))

    def test_a_crop_setting_that_is_not_a_list_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"nmb_crops": 2}),
                                  self.out)
        self.assertIn("nmb_crops", str(e.exception))

    def test_the_lists_reach_the_run_config_unchanged(self):
        run = adapter.to_run_config(self.config(), self.out)
        for key in self.CROP_KEYS:
            with self.subTest(key=key):
                self.assertEqual(run["data"][key], TRAIN[key])


class TestEveryOptionalSettingIsDeclared(Base):
    """The original reads a dozen keys with a default behind them.

    A config that leaves them out would describe a run whose Sinkhorn
    epsilon, warmup and prototype freezing were whatever the code defaulted
    to that day. The contract says the resolved config has to say what ran.
    """

    OPTIONAL_IN_THE_ORIGINAL = ("sinkhorn_eps", "sinkhorn_iters", "eta",
                                "final_lr", "freeze_prototypes_steps",
                                "larc_clip", "start_warmup", "warmup_epochs",
                                "color_jitter_strength", "save_freq")

    def test_each_one_must_be_present(self):
        for key in self.OPTIONAL_IN_THE_ORIGINAL:
            with self.subTest(key=key):
                cfg = self.config()
                del cfg["train"][key]
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, self.out)
                self.assertIn(key, str(e.exception))

    def test_they_are_all_actually_read_by_the_original(self):
        """Against a declaration list that has grown a key nothing uses --
        which would be the same mistake in the other direction."""
        source = (METHOD / "train_step1_resnet.py").read_text(encoding="utf-8")
        for key in self.OPTIONAL_IN_THE_ORIGINAL:
            with self.subTest(key=key):
                self.assertIn(key, source)


class TestTheConfigIsTranslated(Base):
    def test_a_config_naming_an_output_location_is_refused(self):
        for key in ("checkpoint", "output"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(**{key: {"x": 1}}),
                                          self.out)
                self.assertIn("--out", str(e.exception))

    def test_a_key_the_stage_never_reads_is_refused(self):
        cfg = self.config()
        cfg["train"]["mystery"] = 1
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("mystery", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(stage="step2"), self.out)
        self.assertIn("step2", str(e.exception))

    def test_the_output_goes_under_out(self):
        run = adapter.to_run_config(self.config(), self.out)
        self.assertTrue(
            str(Path(run["checkpoint"]["save_dir"])).startswith(str(self.out)))

    def test_the_sinkhorn_settings_reach_the_loss_section(self):
        """The trainer looks for them under `loss`, not under `training`.

        Asserted with values that are **not** the defaults: a mutation that
        replaced the lookup with the literal default survived, because the
        fixture happened to use that same number.
        """
        run = adapter.to_run_config(
            self.config(train={"temperature": 0.07, "sinkhorn_eps": 0.03,
                               "sinkhorn_iters": 5}), self.out)
        self.assertEqual(run["loss"]["temperature"], 0.07)
        self.assertEqual(run["loss"]["sinkhorn_eps"], 0.03)
        self.assertEqual(run["loss"]["sinkhorn_iters"], 5)


class TestTheEncoderIsTheBackbone(Base):
    def test_only_the_encoder_comes_across(self):
        got = adapter.extract_encoder(
            {"encoder.0.weight": 1, "projection_head.0.weight": 2,
             "prototypes.weight": 3})
        self.assertEqual(set(got), {"encoder.0.weight"})

    def test_the_prototypes_do_not_come_across(self):
        """They are the training machinery: a set of learned cluster centres,
        not the representation."""
        got = adapter.extract_encoder(
            {"encoder.0.weight": 1, "prototypes.weight": 2})
        self.assertNotIn("prototypes.weight", got)

    def test_an_empty_result_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"prototypes.weight": 1})
        self.assertIn("encoder", str(e.exception))

    def test_a_ddp_prefix_is_stripped(self):
        got = adapter.extract_encoder({"module.encoder.0.weight": 1})
        self.assertEqual(set(got), {"encoder.0.weight"})


class TestTheMetricNames(Base):
    def test_every_mapped_name_is_in_the_contract_vocabulary(self):
        for raw, target in adapter.STEP1_METRIC_NAMES.items():
            if target is None:
                continue
            with self.subTest(metric=raw):
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_the_loss_is_a_pretext_number(self):
        self.assertEqual(
            adapterlib.METRIC_VOCABULARY[
                adapter.STEP1_METRIC_NAMES["final_loss"]],
            adapterlib.PER_METHOD)

    def test_no_probe_name_is_produced_by_this_stage(self):
        for target in adapter.STEP1_METRIC_NAMES.values():
            with self.subTest(metric=target):
                self.assertNotIn("linear_probe", str(target))


class TestTheTrainingCallIsTheOriginals(Base):
    def test_the_original_run_is_called_once(self):
        calls = []

        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            calls.append(args)
            return {"epochs": 1, "final_loss": 3.2}

        adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(len(calls), 1)

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
class TestSinkhornWithoutADistributedGroup(Base):
    """The assignment step reduces across processes when there are any.

    Run on one process there is nothing to reduce, and the captured code
    guards every collective with `dist.is_initialized()`. That guard is the
    reason this method runs here at all, so it is pinned: a version that
    reduced unconditionally would fail on a single CPU, and the failure would
    look like a porting error rather than a missing process group.
    """

    def sinkhorn(self):
        return load("swav_sinkhorn", METHOD / "distributed_sinkhorn.py")

    def test_it_produces_an_assignment_on_one_process(self):
        import torch
        q = self.sinkhorn().distributed_sinkhorn(
            torch.randn(4, 8), niters=3, eps=0.05)
        self.assertEqual(tuple(q.shape), (4, 8))

    def test_the_assignment_is_a_distribution(self):
        """Each row sums to one: it is a soft assignment over prototypes, and
        a normalisation that stopped normalising would go unnoticed."""
        import torch
        q = self.sinkhorn().distributed_sinkhorn(
            torch.randn(4, 8), niters=3, eps=0.05)
        self.assertTrue(torch.allclose(q.sum(dim=1), torch.ones(4),
                                       atol=1e-4), q.sum(dim=1))

    def test_it_refuses_settings_that_would_not_normalise(self):
        """The captured code checks this itself, and the check is worth
        pinning: a mutation that set the iteration count to zero was caught
        only by the file's hash, which says the edit happened but nothing
        about what it would have done."""
        for kwargs in ({"niters": 0}, {"eps": 0.0}):
            with self.subTest(**kwargs):
                import torch
                with self.assertRaises(ValueError):
                    self.sinkhorn().distributed_sinkhorn(
                        torch.randn(4, 8), **kwargs)

    def test_it_spreads_mass_across_the_prototypes(self):
        """The point of Sinkhorn here is that the prototypes stay evenly
        used. A collapsed assignment would put everything on one."""
        import torch
        torch.manual_seed(0)
        q = self.sinkhorn().distributed_sinkhorn(
            torch.randn(16, 8), niters=3, eps=0.05)
        used = (q.sum(dim=0) > 1e-3).sum().item()
        self.assertGreater(used, 1, "every sample went to one prototype")


@needs_torch
class TestWhichCheckpointAndWhichDevice(Base):
    """Both added because a mutation survived: the smoke run writes one
    checkpoint, so name order and epoch order agree until there are ten, and
    nothing asked for a GPU."""

    def make(self, *epochs: int) -> Path:
        work = self.tmp / "work"
        work.mkdir(parents=True, exist_ok=True)
        for e in epochs:
            (work / f"checkpoint_epoch_{e}.pth").write_bytes(b"x")
        return work

    def test_the_highest_epoch_wins_not_the_last_name(self):
        self.assertEqual(adapter.latest_checkpoint(self.make(1, 9, 10)).name,
                         "checkpoint_epoch_10.pth")

    def test_no_checkpoint_at_all_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.latest_checkpoint(self.make())
        self.assertIn("checkpoint", str(e.exception))

    def test_asking_for_cuda_without_one_is_an_error(self):
        import torch
        t = load("swav_trainer", METHOD / "train_step1_resnet.py")
        real = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            with self.assertRaises(RuntimeError) as e:
                t.resolve_device("cuda")
            self.assertIn("cuda", str(e.exception).lower())
        finally:
            torch.cuda.is_available = real

    def test_auto_takes_the_gpu_when_there_is_one(self):
        import torch
        t = load("swav_trainer", METHOD / "train_step1_resnet.py")
        real = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        try:
            self.assertEqual(t.resolve_device("auto").type, "cuda")
        finally:
            torch.cuda.is_available = real


@needs_torch
class TestASmokeRun(Base):
    def setUp(self) -> None:
        super().setUp()
        tiny_imagenet(self.tmp / "data")

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        """The GPU path, on real hardware -- the case CPU-only testing could
        never reach. A device-placement mistake (a tensor left on the CPU while
        the model is on the GPU) raises inside training, so a run that finishes
        with a non-empty encoder is the GPU path working end to end."""
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
        _, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["status"], "ok")

    def test_the_metrics_use_the_contract_names(self):
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_pretext_loss", doc["metrics"])
        self.assertIn("final_loss", doc["metrics_raw"])

    def test_the_encoder_pt_it_wrote_loads_back(self):
        import torch
        self.run_adapter()
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty")
        load("this_methods_models", METHOD / "models" / "__init__.py")
        returned = adapter.load_encoder(saved, self.config())

        # **Compared by value, not by key.** `get_encoder()` hands back the
        # encoder wrapped in a Sequential with a flatten, so its keys are
        # renumbered -- `0.0.weight` where the file says `encoder.0.weight`.
        # Matching on names would report that nothing loaded, which is what
        # the first version of this test did. What has to hold is that the
        # weights arrived.
        got = list(returned.state_dict().values())
        self.assertTrue(got, "the returned module has no parameters")
        for key, want in saved.items():
            with self.subTest(weight=key):
                self.assertTrue(
                    any(t.shape == want.shape and torch.equal(t, want)
                        for t in got),
                    f"{key} is not in the module that was handed back")

    def test_the_same_config_twice_gives_the_same_encoder(self):
        digests = []
        for i in range(2):
            self.out = self.tmp / f"out{i}"
            _, r = self.run_adapter()
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    def test_the_originals_scratch_files_stay_inside_out(self):
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        listed = sorted(a["path"] for a in man["artifacts"])
        on_disk = sorted(str(p.relative_to(self.out))
                         for p in self.out.rglob("*") if p.is_file()
                         and p.name != "run_manifest.json")
        self.assertEqual(listed, on_disk)


EVAL_TRAIN = {"epochs": 1, "batch_size": 2, "lr": 0.3, "weight_decay": 1.0e-6,
              "num_workers": 0, "img_size": 32, "out_dim": 32,
              "hidden_mlp": 64, "nmb_prototypes": 8}


def tiny_classified(root: Path, classes: int = 2, per_class: int = 2) -> Path:
    """A labelled ImageFolder tree, with the classes separable.

    Separable because the evaluation saves its classifier only when accuracy
    improves on zero, and on pure noise which side of the boundary a few images
    fall on is decided by floating-point detail this project states is not
    reproducible across hardware.
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
               "encoder": str(self.tmp / "encoder.pt"),
               "device": "cpu", "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
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
        """A multi-crop key is a step-1 setting; a stage that never reads it
        must refuse it, or it claims an effect it never had."""
        cfg = self.config()
        cfg["train"]["size_crops"] = [32]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("size_crops", str(e.exception))

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

    def test_it_reports_the_three_the_original_produces(self):
        accuracies = sorted(
            k for k, t in adapter.LINEAR_EVAL_METRIC_NAMES.items()
            if t and "accuracy" in t)
        self.assertEqual(
            accuracies,
            ["best_top1_acc", "final_top1_acc", "final_top5_acc"])

    @needs_torch
    def test_an_encoder_missing_its_backbone_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.load_encoder({}, self.config())
        self.assertIn("encoder", str(e.exception).lower())

    @needs_torch
    def test_asking_for_the_vit_is_refused_by_name(self):
        """Step 2 was not brought across, so its model is absent."""
        evaluation = load("swav_eval", METHOD / "evaluate_linear.py")
        with self.assertRaises(NotImplementedError) as e:
            evaluation.load_encoder(str(self.tmp / "any.pth"), "vit")
        self.assertIn("step 2", str(e.exception))

    @needs_torch
    def test_the_evaluation_refuses_a_gpu_it_does_not_have(self):
        import torch
        evaluation = load("swav_eval", METHOD / "evaluate_linear.py")
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

    def test_the_model_and_the_sinkhorn_are_among_them(self):
        expected = json.loads(
            (METHOD / "provenance.json").read_text())["captured_sha256"]
        for rel in ("models/resnet_swav.py", "distributed_sinkhorn.py"):
            self.assertIn(rel, expected)


if __name__ == "__main__":
    unittest.main()
