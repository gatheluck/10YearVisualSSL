#!/usr/bin/env python3
"""Specification for the third method: 20_simsiam (Chen & He, 2020).

Chosen by measuring the six remaining candidates that carry official-style
files, not by taste. Its trainer is the smallest (288 lines), it uses no
automatic mixed precision at all, its distributed setup already returns early
when `LOCAL_RANK` is unset, and its dataset subclasses
`torchvision.datasets.ImageFolder` -- so the synthetic tree the first port
uses works here unchanged. The measurements are in the Capture repository,
DESIGN 5.41.

What is new here, rather than repeated from the two earlier ports:

- **This is the first port with a metric that has no contract slot.** The
  trainer reports `z_std`, the standard deviation of the L2-normalised
  embeddings, as SimSiam's collapse monitor. It is meaningful and it belongs
  to no family in the vocabulary, so its translation table maps it to `None`:
  kept under the original's name, kept out of the comparable block. That path
  existed and nothing had used it
- **The encoder is one of three modules**, and which one is not a matter of
  taste: the original's own linear evaluation calls `get_encoder()`, which
  returns `self.backbone`. The projector and predictor are training
  machinery. Read from the source rather than decided here
- **The checkpoint directory lives inside the config**, as
  `checkpoint.save_dir`, and in the capture it is an absolute path on the
  cluster -- the same shape of problem the second port met, so the same
  refusal applies
"""

from __future__ import annotations

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
METHOD = ROOT / "methods" / "20_simsiam"
BIN = ROOT / "bin"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

# The trainer imports SummaryWriter at module level, so without tensorboard
# the smoke tests would *fail* rather than skip -- a suite reporting a defect
# where the environment is merely incomplete. The condition names what the
# method actually needs.
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
    """Delegates to the shared helper: three methods now define `data` and
    `models`, and whichever test file imports first would otherwise win."""
    return load_from(METHOD, name, path)


adapter = load("simsiam_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run on a CPU in seconds, and every key the stage reads.
# `dim` and `pred_dim` shrink only the projector head; the ResNet-50 backbone
# is the real one, because a port that swapped the architecture would not be
# a port.
TRAIN = {"epochs": 1, "batch_size": 2, "num_workers": 0, "base_lr": 0.05,
         "momentum": 0.9, "weight_decay": 1.0e-4, "img_size": 32,
         "dim": 64, "pred_dim": 16, "save_freq": 1, "print_freq": 1}


def tiny_imagenet(root: Path, n: int = 4) -> Path:
    """A few synthetic images in the layout `ImageFolder` walks.

    `SimSiamDataset` subclasses `ImageFolder`, so this exercises the real
    sampling and augmentation code without ImageNet. `n` must be at least the
    batch size: the loader is built with `drop_last=True`, so a smaller set
    yields no batches at all and the training loop would run zero times while
    everything still looked fine.
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
        self.tmp = Path(tempfile.mkdtemp(prefix="simsiam-"))
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
    """The contract's config is flat and declares only what affects the
    result. The original reads a nested mapping. This is where they meet."""

    def test_a_config_naming_an_output_location_is_refused(self):
        """The captured config carries an absolute path on the cluster as
        `checkpoint.save_dir`. The contract fixes the output at `--out`, and
        a config naming a directory would claim a location that was not
        used. Overriding it quietly is what is refused."""
        for key in ("checkpoint", "output"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(**{key: {"x": 1}}),
                                          self.out)
                # **Naming the key is not enough.** Deleting this check
                # entirely still refuses the config -- the generic
                # unknown-key rule catches it and its message also contains
                # the word. A surviving mutation showed that, so the
                # assertion is on the reason, which only this check gives.
                self.assertIn("--out", str(e.exception))

    def test_every_key_the_stage_reads_must_be_declared(self):
        cfg = self.config()
        del cfg["train"]["epochs"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("epochs", str(e.exception))

    def test_a_key_the_stage_never_reads_is_refused(self):
        """A key that is ignored is a setting that never took effect, sitting
        in a config that claims it did."""
        cfg = self.config()
        cfg["train"]["mystery"] = 1
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

    def test_train_must_be_a_mapping(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(train=None), self.out)

    def test_the_output_goes_under_out(self):
        run = adapter.to_run_config(self.config(), self.out)
        got = Path(run["checkpoint"]["save_dir"])
        self.assertTrue(str(got).startswith(str(self.out)),
                        f"{got} escapes {self.out}")

    def test_the_learning_rate_rule_is_the_originals(self):
        """`init_lr = base_lr * batch_size / 256` is the linear scaling rule
        the original applies, and it belongs to the original -- the adapter
        passes the inputs and does not recompute it here."""
        run = adapter.to_run_config(self.config(), self.out)
        self.assertEqual(run["training"]["base_lr"], TRAIN["base_lr"])
        self.assertEqual(run["training"]["batch_size"], TRAIN["batch_size"])


class TestWhichCheckpointIsTheFinalOne(Base):
    """**Added because a mutation survived.** The smoke run writes one
    checkpoint, so sorting by name and taking the highest epoch agree, and
    every test passed with either. They disagree the moment there are ten."""

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

    def test_a_single_checkpoint_is_found(self):
        work = self.make(1)
        self.assertEqual(adapter.latest_checkpoint(work).name,
                         "checkpoint_epoch_1.pth")

    def test_no_checkpoint_at_all_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.latest_checkpoint(self.make())
        self.assertIn("checkpoint", str(e.exception))


@needs_torch
class TestTheDeviceIsResolved(Base):
    """**Added because a mutation survived.** Nothing asked for a GPU, so
    removing the availability check changed no test -- and the failure it
    guards against is a run that was asked for CUDA, silently got a CPU, and
    reported success. Patched rather than skipped, so it is checked on every
    machine instead of only on one without a GPU."""

    def trainer(self):
        return load("simsiam_trainer", METHOD / "train_step1_resnet.py")

    def test_cpu_is_honoured(self):
        t = self.trainer()
        self.assertEqual(t.resolve_device("cpu").type, "cpu")

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
        """The other half: the fallback must not become the only path."""
        import torch
        t = self.trainer()
        real = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        try:
            self.assertEqual(t.resolve_device("auto").type, "cuda")
        finally:
            torch.cuda.is_available = real

    def test_an_unknown_device_is_refused(self):
        t = self.trainer()
        with self.assertRaises(ValueError):
            t.resolve_device("tpu")


class TestTheEncoderIsTheBackbone(Base):
    """Which of the three modules is the encoder is read from the original.

    `SimSiamResNet.get_encoder()` returns `self.backbone`, and the original's
    own `evaluate_linear_official.py` builds its frozen feature extractor
    from exactly that. The projector and the predictor are training
    machinery: shipping them would change what `encoder.pt` means from one
    method to the next.
    """

    def test_only_the_backbone_comes_across(self):
        state = {"backbone.0.weight": 1, "projector.0.weight": 2,
                 "predictor.0.weight": 3}
        got = adapter.extract_encoder(state)
        self.assertEqual(set(got), {"backbone.0.weight"})

    def test_an_empty_result_is_an_error_not_an_empty_file(self):
        """If the layout changes, `encoder.pt` would be written empty and
        every later stage would read nothing."""
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"something.else": 1})
        self.assertIn("backbone", str(e.exception))

    def test_a_ddp_prefix_is_stripped(self):
        """A checkpoint written under DistributedDataParallel carries
        `module.`; the original's own loader strips it before loading."""
        got = adapter.extract_encoder({"module.backbone.0.weight": 1})
        self.assertEqual(set(got), {"backbone.0.weight"})


class TestTheMetricNames(Base):
    def test_every_mapped_name_is_in_the_contract_vocabulary(self):
        for raw, target in adapter.STEP1_METRIC_NAMES.items():
            if target is None:
                continue
            with self.subTest(metric=raw):
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_the_loss_is_a_pretext_number(self):
        """SimSiam's negative cosine similarity is its own objective. It
        shares no scale with another method's loss, and the vocabulary has to
        say so or a table will average them."""
        self.assertEqual(
            adapterlib.METRIC_VOCABULARY[
                adapter.STEP1_METRIC_NAMES["final_loss"]],
            adapterlib.PER_METHOD)

    def test_the_collapse_monitor_is_kept_but_not_given_a_slot(self):
        """**The first use of the `None` path.** `z_std` is the standard
        deviation of the normalised embeddings -- SimSiam's collapse monitor.
        It is a real number worth keeping and it belongs to no family, so it
        stays under the original's name and out of the comparable block.
        Inventing a contract name for it would offer it for comparison with
        methods that have no such quantity."""
        self.assertIn("final_z_std", adapter.STEP1_METRIC_NAMES)
        self.assertIsNone(adapter.STEP1_METRIC_NAMES["final_z_std"])


class TestTheTrainingCallIsTheOriginals(Base):
    """The original's loop is called, never reimplemented."""

    def test_the_original_run_is_called_once(self):
        calls = []

        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            calls.append((args, config))
            return {"epochs": 1, "final_loss": -0.5, "final_z_std": 0.1}

        adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(len(calls), 1)

    def test_the_metrics_come_from_that_call(self):
        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            return {"epochs": 1, "final_loss": -0.5, "final_z_std": 0.1}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(m["final_loss"], -0.5)
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
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg_path), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    def test_it_runs_on_the_cpu(self):
        """**The reason this method was chosen.** The captured trainer sends
        its batches and its model to CUDA unconditionally; on a machine
        without a GPU that raises. The port resolves a device instead."""
        _, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["status"], "ok")

    def test_the_encoder_is_written_and_loads_as_the_backbone(self):
        import torch
        self.run_adapter()
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=False)
        self.assertTrue(state, "the encoder is empty")
        self.assertTrue(all(k.startswith("backbone.") for k in state),
                        sorted(state)[:5])

    def test_the_metrics_use_the_contract_names(self):
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_pretext_loss", doc["metrics"])
        self.assertIn("epochs_completed", doc["metrics"])
        for k, v in doc["metrics"].items():
            with self.subTest(metric=k):
                self.assertIsInstance(v, (int, float))
                self.assertNotIsInstance(v, bool)

    def test_the_collapse_monitor_survives_under_its_own_name(self):
        """It has no contract slot, so it must be in the original block and
        nowhere else. Losing it would be a silent loss; promoting it would
        offer it for a comparison it cannot support."""
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertIn("final_z_std", doc["metrics_raw"])
        self.assertNotIn("final_z_std", doc["metrics"])

    def test_no_probe_name_appears(self):
        """This stage trains the method's own objective. A downstream
        accuracy here would be a number nothing measured."""
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        self.assertFalse([k for k in doc["metrics"] if "linear_probe" in k])

    def test_the_originals_scratch_files_stay_inside_out(self):
        """The trainer writes checkpoints, a copy of its config and
        TensorBoard events. All of it belongs under `--out`, and all of it
        has to be listed."""
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
        """The guarantee the whole project rests on, for this method.

        SimSiam's augmentation draws two random views per image, so this
        fails outright unless every source of randomness the run touches is
        seeded -- which is the change the port had to make.
        """
        digests = []
        for i in range(2):
            self.out = self.tmp / f"out{i}"
            _, r = self.run_adapter()
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1],
                         "two runs of one config produced different weights")


    @needs_torch
    def test_the_encoder_pt_it_wrote_loads_back(self):
        """**The round trip, end to end.** Writing the file and never reading
        one back is how an `encoder.pt` that loads nothing goes unnoticed:
        `strict=False` matches no keys and tells nobody, and an evaluation on
        default initialisation reports a number that looks like a result.

        Weights are compared, not just the absence of an exception -- loading
        into a freshly built model and getting default values back would
        satisfy a check that only asked whether it raised.
        """
        import torch
        self.run_adapter()
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty")
        # Three methods define a package called `models`, and only one can
        # be in sys.modules at a time. The adapter imports its own lazily, so
        # the shared helper has to put the right one there first -- the same
        # isolation the rest of the suite uses, in the one place that owns it.
        load("this_methods_models", METHOD / "models" / "__init__.py")
        loaded = adapter.load_encoder(saved, self.config()).state_dict()
        pairs = 0
        for key, want in saved.items():
            short = key.split(".", 1)[1] if key.split(".", 1)[0].isalpha() \
                and key.split(".", 1)[0] not in loaded and "." in key else key
            got = loaded.get(key, loaded.get(short))
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")

EVAL_TRAIN = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "optimizer": "sgd", "weight_decay": 0.0, "img_size": 32,
              "dim": 64, "pred_dim": 16, "print_freq": 1}


def tiny_classified(root: Path, classes: int = 2, per_class: int = 2) -> Path:
    """A labelled ImageFolder tree, with the classes separable.

    Separable for the reason the first port's fixture is: the original saves
    its classifier only when the accuracy improves on zero, and on pure noise
    which side of the boundary four images fall on is decided by
    floating-point detail -- which this project states is not reproducible
    across hardware. A test resting on that is not flaky by accident.
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
    """The second stage, and the first time two different methods produce
    numbers the contract says may be compared."""

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

    def test_step1_settings_are_refused_here(self):
        """A key the stage never reads is a setting claiming an effect it
        never had. The two stages do not read the same keys."""
        cfg = self.config()
        cfg["train"]["momentum"] = 0.9
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("momentum", str(e.exception))

    def test_no_pretext_name_is_produced_here(self):
        """This stage measures downstream classification against real labels.
        A pretext name here would be the false equivalence the vocabulary
        exists to prevent, in the direction that matters."""
        for raw, target in adapter.LINEAR_EVAL_METRIC_NAMES.items():
            with self.subTest(metric=raw):
                self.assertNotIn("pretext", str(target))

    def test_its_accuracies_are_comparable_ones(self):
        """The counters are neither family and belong to every stage; the
        accuracies are the numbers this project compares."""
        probes = [t for t in adapter.LINEAR_EVAL_METRIC_NAMES.values()
                  if t and "accuracy" in t]
        self.assertTrue(probes)
        for target in probes:
            with self.subTest(metric=target):
                self.assertIn("linear_probe", target)
                self.assertEqual(adapterlib.METRIC_VOCABULARY[target],
                                 adapterlib.COMPARABLE)

    def test_it_reports_the_three_the_original_produces(self):
        """**Three, not four.** The first port's evaluation reports a best
        top-5 as well; this one does not, and inventing it would be a number
        nothing measured."""
        accuracies = sorted(
            k for k, t in adapter.LINEAR_EVAL_METRIC_NAMES.items()
            if t and "accuracy" in t)
        self.assertEqual(
            accuracies,
            ["best_top1_acc", "final_top1_acc", "final_top5_acc"])

    @needs_torch
    def test_an_encoder_missing_its_backbone_is_refused(self):
        """A truncated file would otherwise load nothing and the evaluation
        would score default initialisation, which produces a number that
        looks like a result."""
        # Empty rather than a wrong-typed value: a bad type fails the copy
        # first, which is a different complaint and would let the check the
        # test is about go unexercised.
        with self.assertRaises(RuntimeError) as e:
            adapter.load_encoder({}, self.config())
        self.assertIn("backbone", str(e.exception))

    @needs_torch
    def test_asking_for_the_vit_is_refused_by_name(self):
        """Step 2 was not brought across, so its model is absent. Before this
        it failed as an ImportError three frames away, which says nothing
        about why."""
        import torch
        evaluation = load("simsiam_eval",
                          METHOD / "evaluate_linear_official.py")
        # A real file: the captured loader reads the checkpoint before it
        # looks at the model type, so a missing path would fail for an
        # unrelated reason.
        ckpt = self.tmp / "any.pth"
        torch.save({"state_dict": {}, "config": {}}, ckpt)
        with self.assertRaises(NotImplementedError) as e:
            evaluation.load_encoder(str(ckpt), "vit")
        self.assertIn("step 2", str(e.exception))

    @needs_torch
    def test_the_evaluation_refuses_a_gpu_it_does_not_have(self):
        """**Added because a mutation survived.** Replacing the requested
        device with `auto` changes nothing on a machine without a GPU, so
        nothing caught it -- and the failure it hides is a run asked for CUDA
        that quietly used a CPU and reported success. Patched rather than
        skipped, so it is checked everywhere instead of only on a GPU box."""
        import torch
        evaluation = load("simsiam_eval",
                          METHOD / "evaluate_linear_official.py")
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
        """Otherwise a run asked for a GPU could quietly use a CPU and report
        success, and the two are not the same run."""
        args = adapter.eval_args(self.config(device="cuda"), self.out)
        self.assertEqual(args.device, "cuda")
        args = adapter.eval_args(self.config(device="cpu"), self.out)
        self.assertEqual(args.device, "cpu")


@needs_torch
class TestTheLinearEvaluationRuns(Base):
    def setUp(self) -> None:
        super().setUp()
        tiny_classified(self.tmp / "data")

    def make_encoder(self) -> None:
        """A real `encoder.pt`, produced by this method's own first stage."""
        import torch
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
        """CONTRACT section 3: this stage produces a classifier, not an
        encoder. Not producing one quietly is what is forbidden."""
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
        for rel in ("models/simsiam_resnet.py", "data/simsiam_dataset.py"):
            self.assertIn(rel, expected)

    def test_what_was_rewritten_is_recorded(self):
        """A file changed during the port and not listed here reads as
        untouched, which is the claim the hashes exist to make."""
        doc = json.loads((METHOD / "provenance.json").read_text())
        self.assertIn("train_step1_resnet.py", doc["rewritten_during_the_port"])


if __name__ == "__main__":
    unittest.main()
