#!/usr/bin/env python3
"""Specification for the second stage of 1_context_prediction: linear evaluation.

**This is the first stage that eats another stage's output**, and that is why
it was chosen. Step 1 produced `encoder.pt` because the contract says so; if
nothing can consume it, the contract is decorative. Here it is consumed.

Two consequences fall out, both checked below.

**The stage comes from the config.** The contract fixes the adapter's
arguments at exactly two -- `--config` and `--out` -- and says that anything
else affecting the result belongs in the config (CONTRACT section 2). So the
resolved config names the stage, and the adapter dispatches on it. A
`--stage` flag would have been an input the `config_sha256` does not cover.

**This stage produces no encoder, and says so.** It evaluates one and produces
a classifier. CONTRACT section 3 allows that only with a recorded reason,
which is the mechanism written for exactly this case and never exercised
until now.

The original is used as it stands, with the same pure extraction as step 1.
It already ran on CPU, so unlike step 1 no device work was needed.
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
    import torchvision                                 # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH,
                                  "torch and torchvision are not installed")


def load(name: str, path: Path):
    """Delegates to the shared helper: two methods define `data` and `models`,
    and whichever test file imports first would otherwise win."""
    return load_from(METHOD, name, path)


adapter = load("ctxpred_adapter", METHOD / "adapter" / "__init__.py")

EVAL_TRAIN = {"epochs": 1, "batch_size": 2, "feature_batch_size": 2,
              "num_workers": 0, "lr": 0.1, "img_size": 64}


def tiny_classified(root: Path, classes: int = 2, per_class: int = 2) -> Path:
    """An ImageFolder tree: the loader needs labelled directories.

    **The classes are separable, and that is not cosmetic.** This was pure
    noise with arbitrary labels, so the accuracy the stage reached was decided
    by which side of the decision boundary four random images happened to fall
    on. The original saves its classifier only when the accuracy improves on
    `0.0`, so on a runner where all four went the wrong way nothing was
    written, and the test that looks for the classifier failed -- at the same
    commit that passed on the runner beside it.

    Chance was not the real problem. Where the four land depends on
    floating-point detail, and this project states plainly that agreement
    across different hardware is not achievable (README, "Reproducibility").
    A test whose outcome rides on that is not flaky by accident; it is asking
    for something the project says it will not get.

    So each class gets a distinct brightness. The signal is trivial, which is
    the point: the margin has to be wide enough that no arithmetic difference
    can flip it, on a fixture that still runs in seconds.
    """
    from PIL import Image
    import random
    rng = random.Random(0)
    for split in ("train", "val"):
        for c in range(classes):
            d = root / split / f"c{c}"
            d.mkdir(parents=True, exist_ok=True)
            # Spread the classes across the range, and keep the jitter far
            # smaller than the gap between them.
            base = int(30 + c * (200 / max(classes - 1, 1)))
            for i in range(per_class):
                img = Image.new("RGB", (300, 300))
                img.putdata([(min(255, max(0, base + rng.randrange(-8, 9))),) * 3
                             for _ in range(300 * 300)])
                img.save(d / f"{i}.jpg")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lineval-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 42,
               "data_root": str(self.tmp / "data"),
               "encoder": str(self.tmp / "encoder.pt"),
               "device": "cpu", "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
            else:
                cfg[k] = v
        return cfg


class TestTheStageComesFromTheConfig(Base):
    """The contract fixes the adapter's arguments at two. Anything else that
    changes the result has to be inside the hash."""

    def test_the_stage_is_a_required_key(self):
        cfg = self.config()
        del cfg["stage"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(cfg, out=self.out)
        msg = str(e.exception)
        self.assertIn("stage", msg)
        # The unknown-stage refusal also names it. What must survive is the
        # explanation that it is deliberately not defaulted.
        self.assertIn("not defaulted", msg)

    def test_an_unknown_stage_is_refused_and_lists_the_known_ones(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(self.config(stage="step3"), out=self.out)
        msg = str(e.exception)
        self.assertIn("step3", msg)
        self.assertIn("linear_eval", msg)
        self.assertIn("step1", msg)

    def test_each_stage_wants_its_own_keys(self):
        """`encoder` belongs to linear evaluation and `max_steps` to step 1.
        Accepting either anywhere would let a config claim a setting that the
        stage never reads."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(self.config(stage="step1"), out=self.out)
        self.assertIn("encoder", str(e.exception))

    def test_the_adapter_reports_the_stage_it_ran(self):
        self.assertEqual(adapter.stage_of(self.config()), "linear_eval")


class TestTheMetricsAreFilteredHere_too(Base):
    """Same rule as step 1, and it needed its own test.

    A number that is not a number breaks the contract, and a run that
    evaluated nothing must not look like one that did.
    """

    def test_only_numbers_survive(self):
        got = adapter._eval_metrics(
            {"best_top1_acc": 1.0, "best_top5_acc": 2.0,
             "final_top1_acc": 3.0, "final_top5_acc": 4.0,
             "model_type": "official_style_alexnet_context"})
        self.assertEqual(set(got), {"best_top1_acc", "best_top5_acc",
                                    "final_top1_acc", "final_top5_acc"})

    def test_a_missing_accuracy_is_counted_as_unavailable(self):
        got = adapter._eval_metrics({"best_top1_acc": 1.0})
        self.assertEqual(got.get("metrics_unavailable"), 3)

    def test_a_non_numeric_accuracy_is_counted_not_written(self):
        got = adapter._eval_metrics(
            {"best_top1_acc": "high", "best_top5_acc": 2.0,
             "final_top1_acc": 3.0, "final_top5_acc": 4.0})
        self.assertNotIn("best_top1_acc", got)
        self.assertEqual(got.get("metrics_unavailable"), 1)

    def test_a_boolean_is_not_a_number(self):
        got = adapter._eval_metrics(
            {"best_top1_acc": True, "best_top5_acc": 2.0,
             "final_top1_acc": 3.0, "final_top5_acc": 4.0})
        self.assertNotIn("best_top1_acc", got)


class TestConfigTranslation(Base):
    def test_every_setting_reaches_the_arguments(self):
        args = adapter.to_args(self.config(), out=self.out)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.feature_batch_size, 2)
        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.img_size, 64)
        self.assertEqual(args.seed, 42)

    def test_the_encoder_path_reaches_the_arguments(self):
        args = adapter.to_args(self.config(encoder="/runs/a/encoder.pt"),
                               out=self.out)
        self.assertEqual(args.encoder, "/runs/a/encoder.pt")

    def test_the_working_directory_is_inside_out(self):
        args = adapter.to_args(self.config(), out=self.out)
        self.assertTrue(Path(args.save_dir).is_relative_to(self.out))

    def test_a_missing_setting_is_refused_by_name(self):
        for key in EVAL_TRAIN:
            with self.subTest(key=key):
                cfg = self.config()
                cfg["train"] = {k: v for k, v in EVAL_TRAIN.items()
                                if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_args(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_setting_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_args(self.config(train={"learning_rate": 1.0}),
                            out=self.out)
        self.assertIn("learning_rate", str(e.exception))


class TestLoadingTheEncoderTheContractProduced(unittest.TestCase):
    """The original loads a training checkpoint; the contract hands over an
    encoder. Both must work, and the tool must say which it read."""

    def setUp(self) -> None:
        self.ev = load("evaluate_linear_official",
                       METHOD / "evaluate_linear_official.py") \
            if HAVE_TORCH else None

    @needs_torch
    def test_an_encoder_only_state_dict_is_accepted(self):
        import torch
        builder = load("alexnet_context_official",
                       METHOD / "models" / "alexnet_context_official.py")
        want = builder.build_official_context_alexnet(8).get_encoder()
        kind, state = self.ev.read_encoder_state(want.state_dict())
        self.assertEqual(kind, "encoder")
        self.assertEqual(set(state), set(want.state_dict()))

    @needs_torch
    def test_a_full_training_checkpoint_is_still_accepted(self):
        """The cluster's own checkpoints are whole models. Refusing them would
        break a path that works today."""
        builder = load("alexnet_context_official",
                       METHOD / "models" / "alexnet_context_official.py")
        model = builder.build_official_context_alexnet(8)
        kind, state = self.ev.read_encoder_state(
            {"state_dict": model.state_dict()})
        self.assertEqual(kind, "checkpoint")
        self.assertEqual(set(state), set(model.get_encoder().state_dict()))

    @needs_torch
    def test_something_that_is_neither_is_refused(self):
        """Guessing would load a wrong or empty encoder and evaluate noise."""
        with self.assertRaises(RuntimeError) as e:
            self.ev.read_encoder_state({"not": "a state dict"})
        self.assertIn("encoder", str(e.exception).lower())

    @needs_torch
    def test_an_empty_state_dict_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.ev.read_encoder_state({})

    @needs_torch
    def test_a_checkpoint_for_a_different_model_is_refused(self):
        """**Accepting it would evaluate an untrained encoder** and report a
        number that looks like a result. The keys are what tell them apart, so
        keys that do not match must be refused, not partially loaded."""
        import torch
        bad = {"state_dict": {"encoder.something_else": torch.zeros(1),
                              "fc7.weight": torch.zeros(1)}}
        with self.assertRaises(RuntimeError) as e:
            self.ev.read_encoder_state(bad)
        self.assertIn("different model", str(e.exception))

    @needs_torch
    def test_giving_both_inputs_is_refused(self):
        """Two sources and no stated precedence: whichever won would be a
        silent choice about which encoder was evaluated."""
        from argparse import Namespace
        args = Namespace(encoder="/a.pt", checkpoint="/b.pth", save_dir="/tmp",
                         seed=0, device="cpu", gpu=0)
        with self.assertRaises(RuntimeError) as e:
            self.ev.run(args)
        self.assertIn("exactly one", str(e.exception))

    @needs_torch
    def test_giving_neither_input_is_refused(self):
        from argparse import Namespace
        args = Namespace(encoder=None, checkpoint=None, save_dir="/tmp",
                         seed=0, device="cpu", gpu=0)
        with self.assertRaises(RuntimeError):
            self.ev.run(args)


class TestTheOriginalIsUnchanged(unittest.TestCase):
    @needs_torch
    def test_the_command_line_still_takes_the_original_flags(self):
        ev = load("evaluate_linear_official",
                  METHOD / "evaluate_linear_official.py")
        flags = {a.dest for a in ev.build_parser()._actions}
        for original in ("checkpoint", "data_path", "save_dir", "batch_size",
                         "feature_batch_size", "epochs", "lr", "img_size",
                         "num_workers", "gpu"):
            self.assertIn(original, flags, f"--{original} is gone")

    def test_the_body_lives_in_run_and_main_only_parses(self):
        import ast
        src = (METHOD / "evaluate_linear_official.py").read_text()
        top = {n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef)}
        for fn in ("build_parser", "run", "main", "read_encoder_state"):
            self.assertIn(fn, top)
        main_src = src[src.index("def main("):]
        self.assertLess(len(main_src.splitlines()), 12,
                        "main() should parse and delegate, nothing more")

    def test_it_seeds_every_generator_the_stage_uses(self):
        """Feature extraction shuffles and the classifier is initialised, so
        this stage has its own randomness. Same omission as step 1 had."""
        import ast
        src = (METHOD / "evaluate_linear_official.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        seeded = {ast.unparse(n.func) for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call)}
        for call in ("torch.manual_seed", "random.seed", "np.random.seed"):
            self.assertIn(call, seeded, f"{call} is never called")

    def test_it_asks_for_deterministic_kernels_too(self):
        """The same reproducibility guarantee applies to every stage."""
        import ast
        src = (METHOD / "evaluate_linear_official.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("make_deterministic", called)


class TestASmokeRun(Base):
    def run_adapter(self, cfg: dict):
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return p, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    def make_encoder(self) -> Path:
        import torch
        builder = load("alexnet_context_official",
                       METHOD / "models" / "alexnet_context_official.py")
        enc = builder.build_official_context_alexnet(8).get_encoder()
        p = self.tmp / "encoder.pt"
        torch.save(enc.state_dict(), p)
        return p

    @needs_torch
    def test_it_completes_and_satisfies_the_contract(self):
        tiny_classified(self.tmp / "data")
        self.make_encoder()
        cfg, r = self.run_adapter(self.config())
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_torch
    def test_the_absent_encoder_is_explained_not_silent(self):
        """**The case CONTRACT section 3 was written for**, reached for the
        first time. This stage produces a classifier, not an encoder."""
        tiny_classified(self.tmp / "data")
        self.make_encoder()
        self.run_adapter(self.config())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertFalse((self.out / "encoder.pt").exists())
        self.assertTrue(man["encoder_absent_reason"].strip())
        self.assertEqual(man["stage"], "linear_eval")

    @needs_torch
    def test_the_accuracies_are_recorded_as_numbers(self):
        tiny_classified(self.tmp / "data")
        self.make_encoder()
        self.run_adapter(self.config())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for key in ("best_top1_acc", "final_top1_acc"):
            self.assertIn(key, m)
            self.assertIsInstance(m[key], (int, float))
            self.assertNotIsInstance(m[key], bool)

    @needs_torch
    def test_the_classifier_it_trained_is_kept_and_listed(self):
        tiny_classified(self.tmp / "data")
        self.make_encoder()
        self.run_adapter(self.config())
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        if m["best_top1_acc"] <= 0.0:
            # Reported, never silent (DESIGN 2.4). The original saves its
            # classifier only when the accuracy improves on 0.0, so with
            # nothing learned there is no classifier to look for and the
            # check below would be measuring the fixture, not the port.
            self.skipTest(
                "the stage reached 0 accuracy on this fixture, so the "
                "original wrote no classifier: upstream saves only on an "
                "improvement over 0.0. Nothing about the port is in question")
        man = json.loads((self.out / "run_manifest.json").read_text())
        paths = [a["path"] for a in man["artifacts"]]
        self.assertTrue(any(p.endswith(".pth") for p in paths),
                        f"no classifier among {paths}")

    @needs_torch
    def test_everything_written_is_listed_and_everything_listed_exists(self):
        """The property the check above only samples, stated in full.

        "Some artifact ends in .pth" says nothing about the rest, and it is
        conditional on the stage having learned something. This is neither:
        the manifest and the directory must agree exactly, both ways, for
        every file. It is what makes a run auditable by someone who was not
        there.
        """
        tiny_classified(self.tmp / "data")
        self.make_encoder()
        self.run_adapter(self.config())
        man = json.loads((self.out / "run_manifest.json").read_text())
        listed = sorted(a["path"] for a in man["artifacts"])
        on_disk = sorted(str(p.relative_to(self.out))
                         for p in self.out.rglob("*") if p.is_file()
                         and p.name != "run_manifest.json")
        self.assertTrue(on_disk, "the stage wrote nothing at all")
        self.assertEqual(listed, on_disk,
                         "the manifest and the output directory disagree")

    @needs_torch
    def test_a_missing_encoder_file_is_reported_as_a_failure(self):
        tiny_classified(self.tmp / "data")
        cfg, r = self.run_adapter(self.config())
        self.assertNotEqual(r.returncode, 0)

    @needs_torch
    def test_the_same_config_twice_gives_the_same_classifier(self):
        """The guarantee applies here too, and this stage has its own RNG:
        feature extraction shuffles and the classifier is initialised."""
        import hashlib
        tiny_classified(self.tmp / "data")
        self.make_encoder()
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter(self.config())
            man = json.loads((self.out / "run_manifest.json").read_text())
            digests.append({a["path"]: a["sha256"] for a in man["artifacts"]})
        self.assertEqual(digests[0], digests[1])


class TestStepOneStillWorks(unittest.TestCase):
    """Adding a stage must not break the one that exists."""

    def test_the_shipped_step1_config_declares_its_stage(self):
        text = (METHOD / "configs" / "step1.yaml").read_text()
        self.assertIn("stage: step1", text)

    def test_a_linear_eval_config_is_shipped_too(self):
        self.assertTrue((METHOD / "configs" / "linear_eval.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
