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
        cfg = {"stage": "step1", "seed": 0,
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
        trainer = load("barlow_trainer", METHOD / "train_step1_resnet.py")
        with trainer.autocast_context("bf16", "cpu"):
            pass                       # would raise if the device were wrong

    @needs_torch
    def test_fp32_needs_no_autocast_at_all(self):
        import contextlib
        trainer = load("barlow_trainer", METHOD / "train_step1_resnet.py")
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
        for raw, target in adapter.STEP1_METRIC_NAMES.items():
            if target is None:
                continue
            with self.subTest(metric=raw):
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_the_loss_is_a_pretext_number(self):
        """Barlow Twins' redundancy-reduction objective shares no scale with
        another method's loss."""
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
        return load("barlow_trainer", METHOD / "train_step1_resnet.py")

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


if __name__ == "__main__":
    unittest.main()
