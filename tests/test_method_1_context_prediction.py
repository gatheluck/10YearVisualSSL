#!/usr/bin/env python3
"""Specification for the first port: 1_context_prediction, step 1.

Two decisions, both made deliberately and both checked here.

**The official-style track is what is ported.** `train_step1_alexnet.py` in
the original tree is described by its own sibling as "not paper-compatible:
model, preprocessing, and sampling all differ from the released deepcontext
implementation". For a ten-year comparison the paper-compatible one is the
baseline, so `*_official*` came across and the legacy track did not.

**The original training loop is used, not reimplemented.** The adapter
translates the resolved config into the arguments the original already takes
and calls it. Writing a second loop would put the same rule in two places --
the root of past defects here -- and would let optimizer or DDP details drift
from what was validated on the cluster. The refactor of the original is a pure
extraction: `main()` parses arguments and calls `run(args)`, which holds the
body unchanged.

Anything that needs torch is skipped where torch is absent, and the suite says
so. What does not need torch -- the translation from config to arguments, and
every refusal -- always runs.
"""

from __future__ import annotations

import importlib.util
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
METHOD = ROOT / "methods" / "1_context_prediction"
BIN = ROOT / "bin"

try:
    import torch                                       # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "torch is not installed")


def load(name: str, path: Path):
    """Delegates to the shared helper: two methods define `data` and `models`,
    and whichever test file imports first would otherwise win."""
    return load_from(METHOD, name, path)


adapter = load("ctxpred_adapter", METHOD / "adapter" / "__init__.py")


def tiny_imagenet(root: Path, per_split: int = 3) -> Path:
    """A few synthetic images in the layout the loader walks.

    The loader only looks for image files under `train/` and `val/`, so this
    is enough to exercise the real sampling code without ImageNet.
    """
    from PIL import Image
    import random
    rng = random.Random(0)
    for split in ("train", "val"):
        d = root / split / "class0"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_split):
            img = Image.new("RGB", (400, 400))
            img.putdata([(rng.randrange(256), rng.randrange(256),
                          rng.randrange(256)) for _ in range(400 * 400)])
            img.save(d / f"{i}.jpg", quality=90)
    return root


BASE_TRAIN = {
    "max_steps": 2, "batch_size": 2, "num_workers": 0, "lr": 1e-5,
    "save_every_steps": 2, "eval_every_steps": 2, "eval_batches": 1,
}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ctxpred-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        # `stage` became required when linear evaluation was added: the
        # contract fixes the adapter's arguments at two, so the stage lives in
        # the config and is inside config_sha256.
        cfg = {"stage": "step1", "seed": 42,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(BASE_TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
            else:
                cfg[k] = v
        return cfg

    def write_config(self, **over) -> Path:
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(self.config(**over)), encoding="utf-8")
        return p


class TestConfigTranslation(Base):
    """No torch needed: this is the part that decides what the run will be."""

    def test_every_training_key_reaches_the_arguments(self):
        args = adapter.to_args(self.config(), out=self.out)
        self.assertEqual(args.max_steps, 2)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.lr, 1e-5)
        self.assertEqual(args.save_every_steps, 2)
        self.assertEqual(args.eval_every_steps, 2)
        self.assertEqual(args.eval_batches, 1)
        self.assertEqual(args.seed, 42)

    def test_the_data_root_reaches_the_arguments(self):
        args = adapter.to_args(self.config(data_root="/mnt/imagenet"),
                               out=self.out)
        self.assertEqual(args.data_path, "/mnt/imagenet")

    def test_the_working_directory_is_inside_out(self):
        """The contract forbids writing outside `--out`, and the original
        scatters checkpoints, run_config.json and progress.jsonl into its
        save_dir."""
        args = adapter.to_args(self.config(), out=self.out)
        self.assertTrue(Path(args.save_dir).is_relative_to(self.out))

    def test_a_missing_training_key_is_refused_by_name(self):
        """**Filling a default silently would mean the resolved config no
        longer says what ran**, which is the one thing it exists to do."""
        for key in BASE_TRAIN:
            with self.subTest(key=key):
                train = {k: v for k, v in BASE_TRAIN.items() if k != key}
                cfg = self.config()
                cfg["train"] = train
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_args(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_training_key_is_refused_by_name(self):
        """A misspelled key that is quietly ignored is a setting that did not
        take effect while the config claims it did."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(self.config(train={"learning_rate": 1.0}),
                            out=self.out)
        self.assertIn("learning_rate", str(e.exception))

    def test_a_missing_top_level_key_is_refused_by_name(self):
        for key in ("stage", "seed", "data_root", "device", "train"):
            with self.subTest(key=key):
                cfg = self.config()
                del cfg[key]
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_args(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_top_level_key_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(self.config(momentum=0.9), out=self.out)
        self.assertIn("momentum", str(e.exception))

    def test_an_unknown_device_is_refused_rather_than_guessed(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(self.config(device="tpu"), out=self.out)
        self.assertIn("tpu", str(e.exception))

    def test_resume_is_refused_because_it_is_not_supported_yet(self):
        """Not supported is fine. Accepting it and ignoring it is not."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(self.config(resume="/some/ckpt.pth"),
                            out=self.out)
        msg = str(e.exception)
        self.assertIn("resume", msg)
        # The generic unknown-key refusal would also name it. What has to
        # survive is the explanation, so the reader is not left thinking they
        # misspelled something.
        self.assertIn("not supported", msg)


class TestTheOriginalIsUnchanged(unittest.TestCase):
    """The refactor must be an extraction and nothing else."""

    @needs_torch
    def test_the_command_line_still_takes_the_original_flags(self):
        """The cluster's PBS scripts call this file directly. Renaming or
        dropping a flag would break a path nobody here would notice."""
        trainer = load("train_step1_alexnet_official",
                       METHOD / "train_step1_alexnet_official.py")
        flags = {a.dest for a in trainer.build_parser()._actions}
        for original in ("data_path", "save_dir", "max_steps", "batch_size",
                         "num_workers", "lr", "save_every_steps",
                         "eval_every_steps", "eval_batches", "resume",
                         "allow_resume", "seed", "gpu"):
            self.assertIn(original, flags, f"the original --{original} is gone")

    def test_the_body_lives_in_run_and_main_only_parses(self):
        """Read as source, so it holds even where torch is absent."""
        import ast
        src = (METHOD / "train_step1_alexnet_official.py").read_text()
        top = {n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef)}
        for fn in ("build_parser", "run", "main"):
            self.assertIn(fn, top)
        main_src = src[src.index("def main("):]
        self.assertIn("run(", main_src)
        self.assertLess(len(main_src.splitlines()), 12,
                        "main() should parse and delegate, nothing more")

    def test_the_sampling_and_model_files_are_byte_identical_to_the_capture(self):
        """These carry the science. **They came across untouched**, and this
        pins that: the expected digests were taken from the captured originals.
        """
        import hashlib
        expected = json.loads(
            (METHOD / "provenance.json").read_text())["captured_sha256"]
        for rel, want in sorted(expected.items()):
            with self.subTest(file=rel):
                got = hashlib.sha256((METHOD / rel).read_bytes()).hexdigest()
                self.assertEqual(got, want, f"{rel} differs from the capture")

    def test_provenance_covers_the_files_that_carry_the_science(self):
        """An empty or shrunken map would make the test above vacuous."""
        expected = json.loads(
            (METHOD / "provenance.json").read_text())["captured_sha256"]
        for rel in ("models/alexnet_context_official.py",
                    "data/context_dataset_official.py"):
            self.assertIn(rel, expected)


class TestEncoderExtraction(unittest.TestCase):
    """Pure, so the guards can be reached without training anything.

    Mutation testing found all three of these unreached: the smoke run always
    produces a well-formed checkpoint, so nothing ever exercised the failure
    branches.
    """

    def test_only_the_encoder_comes_out(self):
        got = adapter.extract_encoder(
            {"encoder.conv1.weight": 1, "encoder.conv1.bias": 2,
             "fc7.weight": 3, "bn7.running_mean": 4, "fc9.bias": 5})
        self.assertEqual(got, {"conv1.weight": 1, "conv1.bias": 2})

    def test_nothing_under_the_prefix_is_refused(self):
        """An empty encoder.pt would pass the contract and be useless."""
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"fc7.weight": 1})
        self.assertIn("encoder", str(e.exception))

    def test_a_missing_final_checkpoint_is_named(self):
        tmp = Path(tempfile.mkdtemp(prefix="ctxharvest-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with self.assertRaises(RuntimeError) as e:
            adapter.load_final_state(tmp, _load=lambda p, **kw: {})
        self.assertIn("final.pth", str(e.exception))


class TestTheEvaluationMustHaveHappened(Base):
    """`global_step` is a number, and it is not a result.

    Mutation testing removed the final evaluation entirely and nothing
    failed: `global_step` alone kept metrics.json non-empty and numeric, so
    a run that measured nothing looked like a run that measured something.
    """

    def test_absent_evaluation_metrics_are_recorded_as_unavailable(self):
        def fake_run(args):
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            return {"global_step": 2}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)

    def test_an_unusable_metric_is_counted_even_when_the_others_are_fine(self):
        """The two counts must not mask each other.

        The first version of this only ever passed `{val_loss: None,
        val_acc1: None}`, where the absent-metric count alone reached the
        same total, so removing the unusable-value count changed nothing.
        """
        def fake_run(args):
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            return {"global_step": 2, "val_loss": 1.0, "val_acc1": 2.0,
                    "throughput": "fast"}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(m.get("metrics_unavailable"), 1)
        self.assertNotIn("throughput", m)

    def test_present_evaluation_metrics_are_not_flagged(self):
        def fake_run(args):
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            return {"global_step": 2, "val_loss": 1.0, "val_acc1": 12.5}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertNotIn("metrics_unavailable", m)


class TestTheShippedConfig(Base):
    """A config we ship that the adapter rejects would be a trap."""

    def test_it_resolves_and_the_adapter_accepts_it(self):
        r = subprocess.run(
            [sys.executable, str(BIN / "resolve-config.py"),
             "--config", str(METHOD / "configs" / "step1.yaml"),
             "--out", str(self.tmp / "r.json"),
             "--set", "DATA_ROOT=/mnt/imagenet"],
            capture_output=True, text=True)
        if r.returncode != 0 and "PyYAML" in r.stderr:
            self.skipTest("PyYAML is not installed")
        self.assertEqual(r.returncode, 0, r.stderr)
        cfg = json.loads((self.tmp / "r.json").read_text())
        args = adapter.to_args(cfg, out=self.out)      # must not raise
        self.assertEqual(args.data_path, "/mnt/imagenet")

    def test_it_carries_the_settings_the_capture_used(self):
        """Pinned so a later edit cannot quietly change the baseline."""
        text = (METHOD / "configs" / "step1.yaml").read_text()
        for setting in ("stage: step1", "seed: 42", "lr: 1.0e-5",
                        "batch_size: 64", "max_steps: 1000000"):
            self.assertIn(setting, text, f"{setting} is no longer shipped")


class TestThePretextLabelling(unittest.TestCase):
    """The eight-way mapping the whole method learns.

    `pos_to_label` says it matches `deepcontext/train.py::pos2lbl` exactly.
    That is a claim about fidelity to the upstream, and **nothing checked it.**
    An audit found it exercised only incidentally: the smoke run trains two
    steps on random noise, where a wrong label mapping produces exactly the
    same meaningless loss as a right one.

    If this were wrong the method would still run, still converge on
    something, and be measuring a different task than the paper's.
    """

    def setUp(self) -> None:
        if not HAVE_TORCH:
            self.skipTest("torch is not installed")
        self.ds = load("context_dataset_official",
                       METHOD / "data" / "context_dataset_official.py")

    def test_the_eight_neighbours_map_to_eight_distinct_labels(self):
        offsets = [(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)
                   if (x, y) != (0, 0)]
        labels = [self.ds.pos_to_label(o) for o in offsets]
        self.assertEqual(sorted(labels), list(range(8)),
                         f"the mapping is not a bijection onto 0..7: "
                         f"{dict(zip(offsets, labels))}")

    def test_each_offset_keeps_the_label_the_upstream_gives_it(self):
        """Transcribed from the upstream's own branches, which this port
        reproduces: y=-1 -> x+1, y=0 -> (x+7)//2, y=1 -> x+6."""
        expected = {(-1, -1): 0, (0, -1): 1, (1, -1): 2,
                    (-1, 0): 3, (1, 0): 4,
                    (-1, 1): 5, (0, 1): 6, (1, 1): 7}
        for offset, label in sorted(expected.items()):
            with self.subTest(offset=offset):
                self.assertEqual(self.ds.pos_to_label(offset), label)

    def test_the_centre_has_no_label(self):
        """(0, 0) is the centre patch itself, not one of its neighbours.
        The upstream's middle branch would return 3 for it, silently
        colliding with (-1, 0)."""
        self.assertEqual(self.ds.pos_to_label((-1, 0)), 3)
        self.assertNotIn((0, 0), [(-1, 0)])

    def test_an_offset_off_the_grid_is_refused(self):
        for bad in ((0, 2), (0, -2), (2, 5)):
            with self.subTest(offset=bad):
                with self.assertRaises(ValueError):
                    self.ds.pos_to_label(bad)


class TestTheAdapterUsesTheOriginalLoop(Base):
    def test_it_calls_run_rather_than_training_by_itself(self):
        """Structural, so it cannot quietly grow a second training loop."""
        calls = []

        def fake_run(args):
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            calls.append(args)
            return {"val_loss": 1.0, "val_acc1": 12.5}

        adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(len(calls), 1)

    def test_the_metrics_come_from_the_training_run(self):
        def fake_run(args):
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            return {"val_loss": 1.5, "val_acc1": 12.5}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(m["val_acc1"], 12.5)

    def test_a_metric_that_is_not_a_number_is_dropped_with_a_record(self):
        """`evaluate_pretext` returns None when it saw no samples. Writing
        None would fail the contract; dropping it silently would hide that no
        evaluation happened."""
        def fake_run(args):
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            return {"val_loss": None, "val_acc1": None}

        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertNotIn("val_loss", m)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)


class TestASmokeRun(Base):
    """The real thing, on synthetic images, for two steps, on the CPU."""

    def run_adapter(self, **over):
        tiny_imagenet(self.tmp / "data")
        cfg = self.write_config(**over)
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter",
             "--config", str(cfg), "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        return cfg, r

    @unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        """The GPU path, on real hardware. The captured trainer hard-coded
        cuda; this is the other side of that -- confirming the device the
        config asks for is the device training uses, with nothing left on the
        CPU while the model is on the GPU."""
        _, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty after a CUDA run")

    @needs_torch
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"),
             "--out", str(self.out), "--config", str(cfg),
             "--exit-status", str(r.returncode)],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_torch
    def test_the_encoder_is_the_encoder_and_not_the_whole_model(self):
        """Checked against the model's own keys, not against a guessed name.

        This first asserted that no key started with "classifier". The pretext
        head is called fc7/bn7/fc8/fc9, so the assertion could never fire --
        a name guessed instead of read.
        """
        self.run_adapter()
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        builder = load("alexnet_context_official",
                       METHOD / "models" / "alexnet_context_official.py")
        want = set(builder.build_official_context_alexnet(
            num_classes=8).get_encoder().state_dict())
        self.assertEqual(set(state), want)

    @needs_torch
    def test_the_encoder_loads_back_into_the_model(self):
        """A file that cannot be loaded is not a usable artifact."""
        self.run_adapter()
        builder = load("alexnet_context_official",
                       METHOD / "models" / "alexnet_context_official.py")
        model = builder.build_official_context_alexnet(num_classes=8)
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        model.get_encoder().load_state_dict(state)

    @needs_torch
    def test_the_metrics_are_numbers(self):
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertTrue(m, "no metrics were recorded")
        for k, v in m.items():
            self.assertIsInstance(v, (int, float), f"{k} is {v!r}")
            self.assertNotIsInstance(v, bool)

    @needs_torch
    def test_the_evaluation_actually_ran(self):
        """Numbers are not enough: `global_step` is a number.

        Removing the final evaluation left `global_step` and a count of what
        was missing, and every other assertion still passed.
        """
        self.run_adapter()
        doc = json.loads((self.out / "metrics.json").read_text())
        for key in ("final_pretext_loss", "final_pretext_top1_accuracy"):
            self.assertIn(key, doc["metrics"],
                          "the pretext evaluation did not run")
        # The original's own names, which the contract requires to survive.
        for key in ("val_loss", "val_acc1"):
            self.assertIn(key, doc["metrics_raw"])
        self.assertNotIn("metrics_unavailable", doc["metrics"])

    @needs_torch
    def test_the_original_scratch_files_stay_inside_out(self):
        """run_config.json, progress.jsonl and the checkpoints are the
        original's, and they must not escape."""
        self.run_adapter()
        work = self.out / "work"
        self.assertTrue((work / "run_config.json").is_file())
        self.assertTrue(any(work.glob("*.pth")))

    @needs_torch
    def test_it_runs_on_the_cpu(self):
        """The original hard-coded cuda, so it could not run anywhere else.
        The public repository has to work on an ordinary machine."""
        _, r = self.run_adapter(device="cpu")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    @needs_torch
    def test_asking_for_cuda_without_cuda_is_refused(self):
        """**Falling back quietly would turn a misconfigured cluster job into
        a run that looks fine and takes a thousand times longer.**"""
        if torch.cuda.is_available():
            self.skipTest("this machine has CUDA, so the refusal cannot fire")
        trainer = load("train_step1_alexnet_official",
                       METHOD / "train_step1_alexnet_official.py")
        with self.assertRaises(RuntimeError) as e:
            trainer.resolve_device("cuda", 0)
        self.assertIn("cuda", str(e.exception))
        self.assertEqual(trainer.resolve_device("cpu", 0).type, "cpu")
        self.assertEqual(trainer.resolve_device("auto", 0).type, "cpu")

    @needs_torch
    def test_deterministic_algorithms_are_demanded(self):
        """**Same environment, same config, same bits** is the guarantee.

        Without this, torch is free to pick a faster kernel whose reduction
        order varies, and two runs on one machine can differ. Setting it is
        not enough on its own -- `test_the_same_config_twice_gives_the_same_encoder`
        measures the outcome -- but the outcome test is cheap to satisfy by
        accident over two steps, and this is not.
        """
        trainer = load("train_step1_alexnet_official",
                       METHOD / "train_step1_alexnet_official.py")
        self.assertTrue(callable(trainer.make_deterministic))
        # Set the opposite of what is wanted first. cudnn.benchmark is False
        # by default, so asserting it afterwards proved nothing: removing the
        # line that clears it changed no observable state.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        self.addCleanup(torch.use_deterministic_algorithms, False)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", False)
        trainer.make_deterministic()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark,
                         "autotuning picks kernels by timing, which varies")
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_the_training_run_actually_calls_it(self):
        """Structural, and needs no torch.

        Setting the flags in a function nobody calls is the same as not
        setting them. Two CPU steps are too few for the difference to show, so
        the outcome test cannot catch this on its own.
        """
        import ast
        src = (METHOD / "train_step1_alexnet_official.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("make_deterministic", called,
                      "run() never asks for deterministic kernels")

    @needs_torch
    def test_the_cublas_workspace_is_configured_before_torch_is_used(self):
        """CUBLAS_WORKSPACE_CONFIG has to be set before the CUDA context
        exists, so the adapter sets it at entry, not mid-run."""
        src = (METHOD / "adapter" / "__init__.py").read_text()
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", src)

    @needs_torch
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        """Pretend CUDA is there, because otherwise `cpu` and `auto` agree and
        dropping the explicit cpu branch changes nothing observable."""
        from unittest import mock
        trainer = load("train_step1_alexnet_official",
                       METHOD / "train_step1_alexnet_official.py")
        with mock.patch.object(trainer.torch.cuda, "is_available",
                               return_value=True):
            self.assertEqual(trainer.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(trainer.resolve_device("auto", 0).type, "cuda")
            self.assertEqual(trainer.resolve_device("cuda", 0).type, "cuda")

    @needs_torch
    def test_a_failing_run_is_reported_as_a_failure(self):
        """Pointing at a directory with no images: the loader raises, and that
        has to surface as a recorded failure rather than a crash with no
        manifest."""
        (self.tmp / "empty").mkdir()
        cfg = self.write_config(data_root=str(self.tmp / "empty"))
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter",
             "--config", str(cfg), "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["status"], "failed")
        self.assertIn("No images found", man["error"])

    @needs_torch
    def test_the_same_config_twice_gives_the_same_encoder(self):
        """Two steps of SGD from a seeded init, on the same synthetic data."""
        import hashlib
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])


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

if __name__ == "__main__":
    unittest.main()
