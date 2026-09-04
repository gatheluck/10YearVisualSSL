#!/usr/bin/env python3
"""Specification for the fourth method: 21_barlow_twins (Zbontar et al., 2021).

Two distorted views of an image go through the same encoder, and the loss
drives the cross-correlation matrix of the two embeddings towards the
identity: the diagonal to one, everything off it to zero. No negative pairs
and no stop-gradient -- redundancy reduction is what keeps the representation
from collapsing.

Chosen as the runner-up of the six measured candidates (DESIGN 5.41). It shares
the template of the port before it, which is the point: the third and fourth
ports differ in exactly two ways, and both of them are things the earlier port
did not have to solve.

**Mixed precision.** The captured trainer builds `torch.amp.autocast` and
`GradScaler` with `device_type="cuda"` written in, and offers three settings:
fp32, bf16 and fp16. On a CPU, fp32 and bf16 are available and **fp16 is not**.
Quietly falling back would report a run at a precision it did not use, so the
combination is refused by name.

**Python's own random module.** Here the solarisation and the blur call
`random.random()` directly rather than going through a torchvision transform,
so seeding torch alone does not determine the augmentation in this process.

A first version of this port went further and claimed that loader workers
needed seeding too, and changed the captured loader to take a
`worker_init_fn`. **That claim was wrong.** Torch's worker loop seeds
`random` itself -- `random.seed` is in `_worker_loop`, and two runs with no
`worker_init_fn` draw identical values. Both were measured, after the fact,
because a mutation survived that should not have. The change was reverted;
the test below still pins reproducibility with workers, which is a real
property, but it no longer claims to be testing a mechanism this port added.
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
METHOD = ROOT / "methods" / "21_barlow_twins"
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

try:
    import timm                                         # noqa: F401
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    """Four methods now define `data` and `models`, and only one can be in
    `sys.modules` at a time."""
    return load_from(METHOD, name, path)


adapter = load("barlow_adapter", METHOD / "adapter" / "__init__.py")

TRAIN = {"epochs": 1, "batch_size": 2, "num_workers": 0,
         "lr_weights": 0.2, "lr_biases": 0.0048,
         "weight_decay": 1.0e-6, "img_size": 32, "projector": "64-64",
         "lambd": 0.0051, "warmup_epochs": 0, "precision": "fp32",
         "save_freq": 1, "print_freq": 1}


def tiny_imagenet(root: Path, n: int = 4) -> Path:
    """Enough images for one batch; the loader drops a short last batch."""
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
        self.tmp = Path(tempfile.mkdtemp(prefix="barlow-"))
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


class TestPrecisionIsNotSilentlyDowngraded(Base):
    """**The new problem this port had to solve.**

    Three precisions are offered and only two exist on a CPU. A run asked for
    fp16 that quietly used fp32 would be a different run reported as the same
    one, which is the failure the device rule already refuses -- so this
    refuses it the same way, by name.
    """

    def test_fp32_is_accepted_anywhere(self):
        adapter.to_run_config(self.config(train={"precision": "fp32"}),
                              self.out)

    def test_bf16_is_accepted_on_a_cpu(self):
        """bf16 autocast exists on CPU, so there is nothing to refuse."""
        adapter.to_run_config(self.config(train={"precision": "bf16"}),
                              self.out)

    def test_fp16_on_a_cpu_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.config(device="cpu", train={"precision": "amp_fp16"}),
                self.out)
        msg = str(e.exception)
        self.assertIn("amp_fp16", msg)
        self.assertIn("cpu", msg.lower())

    def test_fp16_is_allowed_when_a_gpu_is_asked_for(self):
        """The refusal is about the pair, not about fp16."""
        adapter.to_run_config(
            self.config(device="cuda", train={"precision": "amp_fp16"}),
            self.out)

    def test_an_unknown_precision_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"precision": "fp8"}),
                                  self.out)
        self.assertIn("fp8", str(e.exception))

    @needs_torch
    def test_the_autocast_follows_the_device(self):
        """Written into the captured trainer as `device_type="cuda"`, which
        cannot run anywhere else."""
        trainer = load("barlow_trainer", METHOD / "train_pretrain_resnet.py")
        with trainer.autocast_context("bf16", "cpu"):
            pass                       # would raise if the device were wrong

    @needs_torch
    def test_fp32_needs_no_autocast_at_all(self):
        import contextlib
        trainer = load("barlow_trainer", METHOD / "train_pretrain_resnet.py")
        self.assertIsInstance(trainer.autocast_context("fp32", "cpu"),
                              contextlib.nullcontext)


class TestTheConfigIsTranslated(Base):
    def test_a_config_naming_an_output_location_is_refused(self):
        for key in ("checkpoint", "output"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(**{key: {"x": 1}}),
                                          self.out)
                self.assertIn("--out", str(e.exception))

    def test_every_key_the_stage_reads_must_be_declared(self):
        cfg = self.config()
        del cfg["train"]["lambd"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("lambd", str(e.exception))

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
        got = Path(run["checkpoint"]["save_dir"])
        self.assertTrue(str(got).startswith(str(self.out)))


class TestTheEncoderIsTheBackbone(Base):
    def test_only_the_backbone_comes_across(self):
        got = adapter.extract_encoder(
            {"backbone.conv1.weight": 1, "projector.0.weight": 2})
        self.assertEqual(set(got), {"backbone.conv1.weight"})

    def test_an_empty_result_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"projector.0.weight": 1})
        self.assertIn("backbone", str(e.exception))

    def test_a_ddp_prefix_is_stripped(self):
        got = adapter.extract_encoder({"module.backbone.conv1.weight": 1})
        self.assertEqual(set(got), {"backbone.conv1.weight"})


class TestTheMetricNames(Base):
    def test_every_mapped_name_is_in_the_contract_vocabulary(self):
        for raw, target in adapter.PRETRAIN_METRIC_NAMES.items():
            if target is None:
                continue
            with self.subTest(metric=raw):
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_the_loss_is_a_pretext_number(self):
        """Barlow Twins' redundancy-reduction objective shares no scale with
        another method's loss."""
        self.assertEqual(
            adapterlib.METRIC_VOCABULARY[
                adapter.PRETRAIN_METRIC_NAMES["final_loss"]],
            adapterlib.PER_METHOD)

    def test_no_probe_name_is_produced_by_this_stage(self):
        for target in adapter.PRETRAIN_METRIC_NAMES.values():
            with self.subTest(metric=target):
                self.assertNotIn("linear_probe", str(target))


class TestTheTrainingCallIsTheOriginals(Base):
    def test_the_original_run_is_called_once(self):
        calls = []

        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            calls.append(args)
            return {"epochs": 1, "final_loss": 12.5}

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

    def test_bf16_runs_on_the_cpu_too(self):
        """Not merely accepted by the config check -- actually executed."""
        _, r = self.run_adapter(self.config(train={"precision": "bf16"}))
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])

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
        loaded = adapter.load_encoder(saved, self.config()).state_dict()
        pairs = 0
        for key, want in saved.items():
            got = loaded.get(key, loaded.get(key.split(".", 1)[-1]))
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")

    def test_the_same_config_twice_gives_the_same_encoder(self):
        digests = []
        for i in range(2):
            self.out = self.tmp / f"out{i}"
            _, r = self.run_adapter()
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    def test_it_is_still_reproducible_with_loader_workers(self):
        """Reproducibility does not depend on how many workers load the data.

        It holds because torch seeds each worker -- including Python's
        `random`, which this method's augmentation uses. That is torch's
        doing, not this port's: an earlier version of this file claimed
        otherwise. Pinned anyway, because it is a real property and the day
        it stops holding is a day somebody should look.
        """
        digests = []
        for i in range(2):
            self.out = self.tmp / f"w{i}"
            _, r = self.run_adapter(self.config(train={"num_workers": 2}))
            self.assertEqual(r.returncode, 0, r.stderr[-2500:])
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1],
                         "two runs with loader workers disagreed: the "
                         "workers' random module is not seeded")

    def test_the_originals_scratch_files_stay_inside_out(self):
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        listed = sorted(a["path"] for a in man["artifacts"])
        on_disk = sorted(str(p.relative_to(self.out))
                         for p in self.out.rglob("*") if p.is_file()
                         and p.name != "run_manifest.json")
        self.assertEqual(listed, on_disk)


@needs_torch
class TestWhichCheckpointAndWhichDevice(Base):
    """**Both added because a mutation survived**, and neither survivor was
    equivalent: the smoke run writes one checkpoint, so choosing by name and
    by epoch agree until there are ten, and nothing asked for a GPU."""

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

    def trainer(self):
        return load("barlow_trainer", METHOD / "train_pretrain_resnet.py")

    def test_asking_for_cuda_without_one_is_an_error(self):
        """A run asked for a GPU that quietly used a CPU would report success
        for a run that did not happen. Patched rather than skipped, so it is
        checked on every machine."""
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

    def test_the_training_loop_autocasts_on_the_resolved_device(self):
        """**The subtlest of the three.** `autocast(device_type="cuda")` on a
        machine without CUDA does not raise -- it simply does nothing -- so a
        run at bf16 would silently be a run at fp32, and every test still
        passed. This records what the loop actually asks for.
        """
        t = self.trainer()
        seen = []
        real = t.autocast_context

        def spy(precision, device_type="cuda"):
            seen.append(device_type)
            return real(precision, device_type)

        t.autocast_context = spy
        try:
            tiny_imagenet(self.tmp / "data")
            cfg = self.config(train={"precision": "bf16"})
            args = adapter.to_args(cfg, self.out)
            t.run(args, adapter.to_run_config(cfg, self.out))
        finally:
            t.autocast_context = real
        self.assertTrue(seen, "the loop never entered an autocast")
        self.assertEqual(set(seen), {"cpu"},
                         f"the loop autocast on {set(seen)}, not the "
                         "device it resolved")


EVAL_TRAIN = {"epochs": 1, "batch_size": 2, "lr": 0.3, "num_workers": 0,
              "img_size": 32, "projector": "64-64"}


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
        """The precision knob is a step-1 setting; a stage that never reads it
        must refuse it, or it claims an effect it never had."""
        cfg = self.config()
        cfg["train"]["precision"] = "fp32"
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("precision", str(e.exception))

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
        """Three, not four: this original reports a best top-1 and a final
        top-1 and top-5, but no best top-5."""
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
        self.assertIn("backbone", str(e.exception))

    @needs_torch
    def test_asking_for_the_vit_is_refused_by_name(self):
        """Step 2 was not brought across, so its model is absent; before this
        the top-level import of build_barlow_vit failed at import time."""
        evaluation = load("barlow_eval", METHOD / "evaluate_linear.py")
        with self.assertRaises(NotImplementedError) as e:
            evaluation.load_encoder(str(self.tmp / "any.pth"), "vit")
        self.assertIn("step 2", str(e.exception))

    @needs_torch
    def test_the_evaluation_refuses_a_gpu_it_does_not_have(self):
        import torch
        evaluation = load("barlow_eval", METHOD / "evaluate_linear.py")
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

    def test_the_model_is_among_them(self):
        expected = json.loads(
            (METHOD / "provenance.json").read_text())["captured_sha256"]
        self.assertIn("models/barlow_resnet.py", expected)

    def test_the_loader_came_across_untouched(self):
        """**A first version of this port changed it, and should not have.**

        It was given a `worker_init_fn` on the belief that torch does not seed
        Python's `random` in a DataLoader worker. It does -- measured by
        reading `_worker_loop` and by drawing from two runs without one. The
        change was reverted and the file is pinned, which is the claim the
        hashes exist to make.
        """
        doc = json.loads((METHOD / "provenance.json").read_text())
        self.assertIn("data/barlow_dataset.py", doc["captured_sha256"])
        self.assertNotIn("data/barlow_dataset.py",
                         doc["rewritten_during_the_port"])
        self.assertIn("a_change_that_was_reverted", doc)


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native
# ResNet-50/LARS Barlow Twins pretrain. Cross-correlation (redundancy-reduction)
# loss on two views, the CLS token through a 3-layer projector; tiny dims for a
# CPU smoke (batch_size must be >1 for the projector/output BatchNorm1d).
VIT_MODEL_ARGS = {"projector": "8-8", "lambd": 0.0051, "image_size": 32,
                  "patch_size": 16, "embed_dim": 16, "depth": 1, "num_heads": 2,
                  "mlp_ratio": 4.0, "drop_rate": 0.0, "attn_drop_rate": 0.0}
VIT_TRAIN_TINY = {"arch": "vit", "projector": "8-8", "lambd": 0.0051,
                  "img_size": 32, "patch_size": 16, "embed_dim": 16, "depth": 1,
                  "num_heads": 2, "mlp_ratio": 4.0, "drop_rate": 0.0,
                  "attn_drop_rate": 0.0, "epochs": 2, "batch_size": 2,
                  "num_workers": 0, "lr": 6.0e-4, "weight_decay": 0.05,
                  "warmup_epochs": 0, "min_lr": 0.0, "save_at_epochs": [1, 2]}


class TestVitConfigTranslation(Base):
    def vit_config(self, train=None, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": dict(train if train is not None else VIT_TRAIN_TINY)}
        for k, v in over.items():
            cfg[k] = v
        return cfg

    def test_the_vit_step2_config_is_accepted(self):
        built = adapter.to_run_config(self.vit_config(), self.out)
        self.assertEqual(built["arch"], "vit")
        self.assertEqual(built["model"]["embed_dim"], 16)
        self.assertEqual(built["barlow"]["lambd"], 0.0051)
        self.assertEqual(built["training"]["save_at_epochs"], [1, 2])

    def test_native_path_has_no_top_level_arch(self):
        built = adapter.to_run_config(self.config(), self.out)
        self.assertNotIn("arch", built)

    def test_a_bad_arch_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"arch": "resnext"}),
                                  self.out)
        self.assertIn("arch", str(e.exception))

    def test_a_missing_vit_setting_is_refused_by_name(self):
        for key in VIT_TRAIN_TINY:
            if key == "arch":
                continue
            with self.subTest(key=key):
                t = {k: v for k, v in VIT_TRAIN_TINY.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.vit_config(train=t), self.out)
                self.assertIn(key, str(e.exception))

    def test_native_lars_knob_does_not_leak_into_the_vit_path(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.vit_config(train={**VIT_TRAIN_TINY, "lr_weights": 0.2}),
                self.out)
        self.assertIn("lr_weights", str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_barlow", METHOD / "models" / "vit_barlow.py")
        return vm.build_barlow_vit(**VIT_MODEL_ARGS)

    def _batch(self, torch, b=2):
        return torch.randn(b, 3, VIT_MODEL_ARGS["image_size"],
                           VIT_MODEL_ARGS["image_size"])

    @needs_timm
    def test_the_encoder_returns_the_cls_feature(self):
        import torch
        feats = self._model().get_encoder()(self._batch(torch))
        self.assertEqual(tuple(feats.shape), (2, VIT_MODEL_ARGS["embed_dim"]))

    @needs_timm
    def test_forward_is_a_finite_scalar_loss(self):
        import torch
        model = self._model()
        model.train()
        loss = model(self._batch(torch), self._batch(torch))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    @needs_timm
    def test_encoder_pt_holds_only_the_backbone(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith("backbone.") for k in got))
        self.assertFalse([k for k in got if k.startswith("projector")])
        self.assertFalse([k for k in got if k.startswith("bn")])

    @needs_timm
    def test_load_encoder_round_trips_the_backbone(self):
        import torch
        saved = adapter.extract_encoder(self._model().state_dict())
        cfg = {"train": {"arch": "vit", "projector": "8-8", "img_size": 32,
                         "patch_size": 16, "embed_dim": 16, "depth": 1,
                         "num_heads": 2, "mlp_ratio": 4.0, "drop_rate": 0.0,
                         "attn_drop_rate": 0.0}}
        encoder = adapter.load_encoder(saved, cfg)
        loaded = encoder.state_dict()
        pairs = 0
        for k, want in saved.items():
            got = loaded.get(k[len("backbone."):])  # get_encoder() is the backbone
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{k} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the backbone")


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
                "train": {"arch": "vit", "projector": "8-8", "img_size": 32,
                          "patch_size": 16, "embed_dim": 16, "depth": 1,
                          "num_heads": 2, "mlp_ratio": 4.0, "drop_rate": 0.0,
                          "attn_drop_rate": 0.0, "epochs": 1, "batch_size": 2,
                          "lr": 0.3, "num_workers": 0}}

    @needs_torch  # the smoke runs the eval subprocess, which writes tensorboard logs
    @needs_timm
    def test_pretrain_milestones_then_probe_passes_contract(self):
        tiny_imagenet(self.tmp / "data")
        pre = self.tmp / "pre_out"
        _, r = self._adapter(
            {"stage": "pretrain", "seed": 0, "data_root": str(self.tmp / "data"),
             "device": "cpu", "train": dict(VIT_TRAIN_TINY)}, pre)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((pre / "encoder.pt").is_file())
        for n in (1, 2):
            self.assertTrue((pre / f"encoder_epoch{n}.pt").is_file(),
                            f"milestone encoder_epoch{n}.pt not written")

        tiny_classified(self.tmp / "eval")
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


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. It reuses this method's
    own encoder loader and eval pipeline, so the check is that it returns the
    2048-d ResNet-50 backbone feature -- raw, before any probe normalise -- one
    row per val image, with honest meta.

    The encoder.pt is built from this method's own model, filtered through the
    same `extract_encoder` the adapter writes with, so only the backbone is
    saved; the provider then rebuilds it through the *shipped* linear_eval
    config's architecture (`adapter.load_encoder`, which for the resnet path
    returns the backbone itself). Random weights do not affect the
    shape-and-plumbing this proves. Modules load through `load` (`load_from`),
    which purges any other method's `adapter`/`models` first -- the whole suite
    runs many methods in one interpreter.

    Two barlow-specific facts, both measured: `load_encoder` returns the
    backbone directly (no second `get_encoder()`), and the feature width is
    `adapter.BACKBONE_DIM` (2048), the ResNet-50 pooled feature the evaluation
    itself probes.
    """

    def _shipped_config(self) -> dict:
        import yaml
        return yaml.safe_load(
            (METHOD / "configs" / "linear_eval.yaml").read_text())

    def _make_encoder(self) -> Path:
        """A real `encoder.pt`: this method's model, backbone-only, as the
        adapter writes it. A tiny projector keeps it light -- only the backbone
        is saved, and the provider rebuilds the shipped projector itself."""
        import torch
        models = load("this_methods_models", METHOD / "models" / "__init__.py")
        model = models.build_barlow_resnet(projector="64-64")
        state = adapter.extract_encoder(model.state_dict())
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(state, encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("barlow_feature_provider", METHOD / "feature_provider.py")

    @needs_torch
    def test_it_returns_raw_2048d_features_one_per_val_image(self):
        import numpy as np
        tiny_classified(self.tmp / "data")            # train + val, 2 x 2 each
        encoder_pt = self._make_encoder()

        prov = self._provider()
        feats, labels, meta = prov.extract_val_features(
            encoder_path=str(encoder_pt), data_root=str(self.tmp / "data"),
            split="val", device="cpu", batch_size=2, num_workers=0)

        feats = np.asarray(feats)
        self.assertEqual(feats.ndim, 2)
        self.assertEqual(feats.shape[0], 4, "4 val images expected")
        self.assertEqual(feats.shape[1], adapter.BACKBONE_DIM,
                         "ResNet-50 feature is 2048-d")
        self.assertEqual(np.asarray(labels).shape[0], 4)
        self.assertEqual(meta["feat_dim"], adapter.BACKBONE_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_torch
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads it,
        with the encoder's sha256 recorded in meta."""
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        tiny_classified(self.tmp / "data")
        encoder_pt = self._make_encoder()

        record = {"method": METHOD.name, "status": "ready",
                  "provider": str(METHOD / "feature_provider.py"),
                  "encoder": str(encoder_pt)}
        out = self.tmp / "features"
        updated = driver.extract_one(
            record, data_root=str(self.tmp / "data"), split="val", out=out,
            device="cpu", batch_size=2, num_workers=0)

        self.assertEqual(updated["status"], "ok", updated.get("reason", ""))
        method_out = out / METHOD.name
        feats = np.load(method_out / "features.npy")
        labels = np.load(method_out / "labels.npy")
        meta = json.loads((method_out / "meta.json").read_text())
        self.assertEqual(feats.shape, (4, adapter.BACKBONE_DIM))
        self.assertEqual(labels.shape[0], 4)
        self.assertEqual(meta["encoder_sha256"],
                         hashlib.sha256(encoder_pt.read_bytes()).hexdigest())

    @needs_torch
    def test_the_isolated_driver_run_extracts_this_method_end_to_end(self):
        """The whole driver, real subprocess, real provider -- a method whose
        adapter imports the shared `adapterlib`, so it catches the class of
        regression where the isolated worker cannot see a repository-root module
        the provider needs (the worker puts ROOT on sys.path)."""
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        tiny_classified(self.tmp / "data")
        encoder_pt = self._make_encoder()
        out = self.tmp / "features"
        manifest = driver.run(
            METHOD.parent, data_root=str(self.tmp / "data"), split="val",
            out=out, encoders={METHOD.name: str(encoder_pt)}, encoders_root=None,
            device="cpu", batch_size=2, num_workers=0, venvs_root=ROOT / ".venvs")

        rec = {r["method"]: r for r in manifest["records"]}[METHOD.name]
        self.assertEqual(rec["status"], "ok", rec.get("reason", ""))
        feats = np.load(out / METHOD.name / "features.npy")
        self.assertEqual(feats.shape, (4, adapter.BACKBONE_DIM))


if __name__ == "__main__":
    unittest.main()
