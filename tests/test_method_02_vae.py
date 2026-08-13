#!/usr/bin/env python3
"""Specification for the second method: 02_vae (Kingma & Welling, 2013).

Chosen by measuring all 37 methods, not by taste. **It is the only one that
uses MNIST**; the other 36 are ImageNet-only. At 28x28 with batch 100 and Adam
at 1e-3 it therefore trains to completion on a CPU, so this is the first port
whose tests run a real training run rather than two steps on noise.

It also breaks new ground rather than repeating the first port:

- **The output path lives inside the config**, as `output.checkpoint_dir`, and
  in the capture it is an absolute path on the cluster. The contract says the
  adapter writes only under `--out`, so that has to be redirected -- and the
  shipped config must not carry a machine's path at all
- **The data is not an `ImageFolder`.** It is `torchvision.datasets.MNIST`,
  which the original loads with `download=False`. Tests fabricate the IDX
  files rather than reach the network
- **It is a generative model**, so `encoder.pt` is one half of it. The
  encoder-extraction idea from the first port meets a different architecture
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _method_import import load_from        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHOD = ROOT / "methods" / "02_vae"
BIN = ROOT / "bin"

# tensorboard belongs in here, not only torch: the captured trainer imports
# SummaryWriter at module level, so without it the smoke tests *fail* rather
# than skip -- which is a suite reporting a defect where the environment is
# simply incomplete. The condition has to name what the method actually needs.
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
    """Delegates to the shared helper: two methods define `data` and `models`,
    and whichever test file imports first would otherwise win."""
    return load_from(METHOD, name, path)


adapter = load("vae_adapter", METHOD / "adapter" / "__init__.py")

TRAIN = {"epochs": 1, "batch_size": 4, "num_workers": 0, "lr": 1.0e-3,
         "beta": 1.0, "latent_dim": 4, "hidden_dim": 8, "img_size": 28,
         "save_freq": 1, "print_freq": 1}


def tiny_mnist(root: Path, n: int = 8) -> Path:
    """A valid MNIST directory, fabricated rather than downloaded.

    `torchvision.datasets.MNIST` is created with `download=False` by the
    original, so a test that reached the network would be testing the network.
    The IDX format is a short header and then raw bytes, so a real one small
    enough to train on is cheap to write.
    """
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    images = bytes(range(256)) * ((n * 28 * 28) // 256 + 1)
    images = images[:n * 28 * 28]
    labels = bytes(i % 10 for i in range(n))
    for name, header, body in (
            ("train-images-idx3-ubyte", struct.pack(">IIII", 2051, n, 28, 28),
             images),
            ("train-labels-idx1-ubyte", struct.pack(">II", 2049, n), labels),
            ("t10k-images-idx3-ubyte", struct.pack(">IIII", 2051, n, 28, 28),
             images),
            ("t10k-labels-idx1-ubyte", struct.pack(">II", 2049, n), labels)):
        (raw / name).write_bytes(header + body)
        with gzip.open(raw / f"{name}.gz", "wb") as f:
            f.write(header + body)
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vae-"))
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


class TestTwoMethodsDoNotCollide(unittest.TestCase):
    """Both methods define `data` and `models`, and only one can be in
    `sys.modules` at a time.

    Found when this method arrived: the first method's trainer did
    `from data.context_dataset_official import ...` and got *this* method's
    `data`. Each test file passed alone; only the whole suite showed it.
    """

    @needs_torch
    def test_each_method_gets_its_own_shared_packages(self):
        one = ROOT / "methods" / "01_context_prediction"
        a = load_from(METHOD, "vae_models_probe", METHOD / "models" / "__init__.py")
        b = load_from(one, "ctx_models_probe", one / "models" / "__init__.py")
        self.assertTrue(a.__file__.startswith(str(METHOD)))
        self.assertTrue(b.__file__.startswith(str(one)))
        self.assertNotEqual(a.__file__, b.__file__)

    @needs_torch
    def test_a_cached_shared_package_is_replaced_not_reused(self):
        """**The original failure, reproduced deliberately.**

        Putting the method's directory first on `sys.path` is not enough:
        once `sys.modules["data"]` holds one method's package, an import of
        `data` returns it whatever the path says. So the cache is purged, and
        this is the test that reaches that code -- loading by a unique name,
        as the other tests do, bypasses the cache and proves nothing.
        """
        one = ROOT / "methods" / "01_context_prediction"
        # Cache *this* method's `data` under the name the other one imports.
        load_from(METHOD, "data", METHOD / "data" / "__init__.py")
        self.assertTrue(sys.modules["data"].__file__.startswith(str(METHOD)))
        # The other method's trainer does `from data.context_dataset_official
        # import ...`, which fails outright if the wrong package is cached.
        trainer = load_from(
            one, "collision_probe", one / "train_pretrain_alexnet_official.py")
        self.assertTrue(callable(trainer.run))
        self.assertTrue(sys.modules["data"].__file__.startswith(str(one)))

    @needs_torch
    def test_the_import_of_one_does_not_poison_the_other(self):
        """The failure exactly: load this method's `data`, then ask the other
        method's module to import its own."""
        one = ROOT / "methods" / "01_context_prediction"
        load_from(METHOD, "vae_data_probe", METHOD / "data" / "__init__.py")
        mod = load_from(one, "ctx_data_probe", one / "data" / "__init__.py")
        self.assertTrue(mod.__file__.startswith(str(one)))
        self.assertTrue(hasattr(mod, "OfficialContextPredictionDataset"))


class TestTheOutputPathIsNotTheClusters(unittest.TestCase):
    """**The new wrinkle in this port.**

    The captured config carries
    `/groups/.../methods/02_vae/checkpoints/step1_mnist_original` as
    `output.checkpoint_dir`. Two things are wrong with shipping that: the
    contract says the adapter writes only under `--out`, and a machine's
    absolute path in a published config is not reproducible anywhere else.
    """

    def settings(self) -> str:
        """The config with its comments removed.

        The first version searched the whole file and matched the comment
        *explaining* that the output path is not there. What matters is
        whether anything is set, not what the prose says.
        """
        text = (METHOD / "configs" / "pretrain_mnist.yaml").read_text()
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    def test_the_shipped_config_names_no_machine_path(self):
        body = self.settings()
        self.assertNotIn("/groups/", body)
        self.assertNotIn("checkpoint_dir", body,
                         "the output path belongs to --out, not the config")
        self.assertNotIn("output", body)

    def test_removing_the_comments_left_the_settings(self):
        """Against an empty body the check above passes vacuously."""
        self.assertIn("stage: pretrain", self.settings())
        self.assertIn("latent_dim", self.settings())

    def test_the_adapter_puts_the_checkpoint_dir_under_out(self):
        cfg = {"stage": "pretrain", "seed": 0, "data_root": "/d",
               "device": "cpu", "train": dict(TRAIN)}
        built = adapter.to_run_config(cfg, out=Path("/tmp/somewhere/out"))
        self.assertTrue(
            Path(built["output"]["checkpoint_dir"]).is_relative_to(
                Path("/tmp/somewhere/out")))

    def test_a_config_that_tries_to_set_it_is_refused(self):
        """Accepting it would let a config write outside `--out`, which is the
        one thing the contract says an adapter must never do."""
        cfg = {"stage": "pretrain", "seed": 0, "data_root": "/d",
               "device": "cpu", "train": dict(TRAIN),
               "output": {"checkpoint_dir": "/anywhere"}}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=Path("/tmp/out"))
        msg = str(e.exception)
        self.assertIn("output", msg)
        # The generic unknown-key refusal also names it. What has to survive
        # is the reason: the location is fixed by the contract, not chosen.
        self.assertIn("--out", msg)


class TestExtractingTheEncoder(unittest.TestCase):
    """Pure, so the guards can be reached without training anything.

    Mutation testing found the empty-encoder guard unreachable: a real run
    always produces a model with the expected prefixes, so nothing exercised
    the branch that catches a changed layout.
    """

    def test_only_the_encoder_side_comes_out(self):
        got = adapter.extract_encoder({
            "encoder.0.weight": 1, "fc_mu.weight": 2, "fc_logvar.bias": 3,
            "decoder.0.weight": 4, "decoder_input.weight": 5})
        self.assertEqual(set(got),
                         {"encoder.0.weight", "fc_mu.weight",
                          "fc_logvar.bias"})

    def test_the_latent_projections_count_as_the_encoder(self):
        """`fc_mu` and `fc_logvar` map the features to the latent
        distribution. Leaving them out would hand over an encoder that cannot
        produce a code."""
        got = adapter.extract_encoder({"encoder.0.weight": 1,
                                       "fc_mu.weight": 2})
        self.assertIn("fc_mu.weight", got)

    def test_nothing_matching_is_refused(self):
        """An empty encoder.pt would satisfy the contract and be useless."""
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"decoder.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())

    def test_an_empty_checkpoint_is_refused(self):
        with self.assertRaises(RuntimeError):
            adapter.extract_encoder({})


class TestTheLossMustHaveBeenMeasured(Base):
    """A number is not a result.

    Mutation testing removed the return value entirely and nothing failed:
    the metrics fell back to a bare count of what was missing, which is
    numeric and non-empty, so every other assertion still passed.
    """

    def test_a_run_that_reports_nothing_is_flagged(self):
        def fake_run(args, config=None):
            return None
        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)
        self.assertNotIn("final_loss", m)

    def test_a_run_that_reports_a_loss_is_not_flagged(self):
        def fake_run(args, config=None):
            return {"epochs": 1, "final_loss": 12.5}
        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertEqual(m["final_loss"], 12.5)
        self.assertNotIn("metrics_unavailable", m)

    def test_a_non_numeric_loss_is_counted_not_written(self):
        def fake_run(args, config=None):
            return {"epochs": 1, "final_loss": "low"}
        m = adapter.run_training(self.config(), self.out, _run=fake_run)
        self.assertNotIn("final_loss", m)
        self.assertGreaterEqual(m.get("metrics_unavailable", 0), 1)


class TestConfigTranslation(Base):
    def test_every_setting_reaches_the_run_config(self):
        built = adapter.to_run_config(self.config(), out=self.out)
        self.assertEqual(built["training"]["epochs"], 1)
        self.assertEqual(built["training"]["lr"], 1.0e-3)
        self.assertEqual(built["training"]["beta"], 1.0)
        self.assertEqual(built["data"]["batch_size"], 4)
        self.assertEqual(built["data"]["img_size"], 28)
        self.assertEqual(built["model"]["latent_dim"], 4)

    def test_a_missing_setting_is_refused_by_name(self):
        for key in TRAIN:
            with self.subTest(key=key):
                cfg = self.config()
                cfg["train"] = {k: v for k, v in TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_setting_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"momentum": 0.9}),
                                  out=self.out)
        self.assertIn("momentum", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(stage="step9"), out=self.out)


class TestTheDataIsMnist(Base):
    """The loader chooses by inspecting the path, which is worth pinning."""

    @needs_torch
    def test_a_fabricated_mnist_directory_is_recognised(self):
        tiny_mnist(self.tmp / "data")
        ds = load("vae_dataset", METHOD / "data" / "vae_dataset.py")
        loader, _ = ds.get_vae_dataloader(
            data_path=str(self.tmp / "data"), batch_size=2, num_workers=0,
            img_size=28, augmentation_type="none", distributed=False)
        batch = next(iter(loader))
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        self.assertEqual(images.shape[-2:], (28, 28))

    @needs_torch
    def test_nothing_is_downloaded(self):
        """The original passes download=False. A test that reached the
        network would be testing the network."""
        src = (METHOD / "data" / "vae_dataset.py").read_text()
        self.assertIn("download=False", src)


class TestDeviceIsHonoured(Base):
    """The device is decided from the config, not sniffed from the hardware.

    02_vae was ported on the CPU track: the adapter validated `config["device"]`
    against auto/cuda/cpu, but `to_args()` never put the value on the trainer's
    arguments and `run()` chose the device from `torch.cuda.is_available()`
    alone. On a CPU-only machine the answer was always `cpu`, requested or not,
    so nothing showed. On a GPU it is a real defect -- `device: cpu` runs on the
    GPU, and `device: cuda` without a GPU runs on the CPU instead of refusing.
    Found the first time the port ran on real GPU hardware. See docs/GPU.md 4.
    """

    def trainer(self):
        return load("vae_trainer", METHOD / "train_pretrain_cnn.py")

    @needs_torch
    def test_asking_for_cuda_without_one_is_refused(self):
        """Falling back to the CPU without a word turns a misconfigured GPU job
        into a run that looks fine and misreports which hardware produced it."""
        from unittest import mock
        t = self.trainer()
        with mock.patch.object(t.torch.cuda, "is_available",
                               return_value=False):
            with self.assertRaises(RuntimeError) as e:
                t.resolve_device("cuda", 0)
            self.assertIn("cuda", str(e.exception).lower())
            self.assertEqual(t.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(t.resolve_device("auto", 0).type, "cpu")

    @needs_torch
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        """Pretend CUDA is there, because otherwise cpu and auto agree and
        dropping the explicit cpu branch changes nothing observable."""
        from unittest import mock
        t = self.trainer()
        with mock.patch.object(t.torch.cuda, "is_available",
                               return_value=True):
            self.assertEqual(t.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(t.resolve_device("auto", 0).type, "cuda")
            self.assertEqual(t.resolve_device("cuda", 0).type, "cuda")

    @needs_torch
    def test_the_adapter_forwards_the_device_to_the_trainer(self):
        """Validating the device and then not forwarding it is the defect this
        catches: `to_args()` must put it on the arguments the trainer reads."""
        args = adapter.to_args(self.config(device="cuda"), self.out)
        self.assertEqual(args.device, "cuda")

    def test_run_resolves_the_device_rather_than_sniffing_it(self):
        """Structural, and needs no torch. Choosing the device straight from
        `torch.cuda.is_available()` is exactly the bug; `run()` must go through
        `resolve_device`, so the requested value is what decides."""
        import ast
        src = (METHOD / "train_pretrain_cnn.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("resolve_device", called,
                      "run() selects the device without resolve_device")


class TestASmokeRun(Base):
    def run_adapter(self, **over):
        tiny_mnist(self.tmp / "data")
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.config(**over)), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return cfg, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    @needs_torch
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        """The GPU path, on real hardware -- the case CPU-only testing could
        never reach. A device-placement mistake (a tensor left on the CPU while
        the model is on the GPU) raises inside training, so a run that finishes
        and writes a non-empty encoder is the GPU path working end to end."""
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower(),
                      "the run did not report running on a CUDA device")
        saved = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(saved, "encoder.pt is empty after a CUDA run")

    @needs_torch
    def test_the_encoder_is_the_encoder_half_of_the_model(self):
        """A VAE is an encoder and a decoder. The contract wants the encoder,
        and shipping the decoder with it would change what encoder.pt means
        from one method to the next."""
        self.run_adapter()
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(state)
        self.assertFalse([k for k in state if k.startswith("decoder")],
                         "the decoder is in encoder.pt")

    @needs_torch
    def test_nothing_is_written_outside_out(self):
        """The captured config wrote to the cluster. This is the test that
        the redirection actually holds at run time, not just in translation."""
        before = {p for p in self.tmp.rglob("*")}
        self.run_adapter()
        stray = [p for p in self.tmp.rglob("*")
                 if p not in before and not p.is_relative_to(self.out)
                 and not p.is_relative_to(self.tmp / "data")
                 and p != self.tmp / "resolved.json"]
        self.assertEqual(stray, [], f"written outside --out: {stray}")

    @needs_torch
    def test_the_loss_is_recorded_as_a_number(self):
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertTrue(m, "no metrics were recorded")
        for k, v in m.items():
            self.assertIsInstance(v, (int, float), f"{k} is {v!r}")
            self.assertNotIsInstance(v, bool)

    @needs_torch
    def test_the_training_actually_measured_something(self):
        """Numbers are not enough. A bare count of missing metrics is numeric
        and non-empty, and would pass every other check here."""
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_pretext_loss", m, "the run reported no loss")
        self.assertNotIn("metrics_unavailable", m)

    @needs_torch
    def test_the_same_config_twice_gives_the_same_encoder(self):
        import hashlib
        digests = []
        for name in ("a", "b"):
            self.out = self.tmp / name
            self.run_adapter()
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    @needs_torch
    def test_a_missing_dataset_is_reported_as_a_failure(self):
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.config(
            data_root=str(self.tmp / "absent"))), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


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

# --- linear_eval: the stage that closes 02_vae's pretrain+probe pipeline ---
# The capture probes the VAE encoder's latent mean `mu` (models.VAE_CNN.get_features)
# with a single linear layer, SGD + cosine, CrossEntropyLoss, inputs kept in [0,1]
# (no ImageNet norm) and mu fed in unnormalised -- faithful to the VAE, and
# deliberately different from the shared ARSSL probe. Dataset-agnostic: MNIST for
# the shipped MNIST pretrain, ImageFolder if supplied.

EVAL_TRAIN = {"latent_dim": 4, "hidden_dim": 8, "img_size": 28,
              "epochs": 1, "batch_size": 4, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}


class TestLinearEvalConfig(Base):
    def eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "encoder": str(self.tmp / "encoder.pt"), "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v is not None:
                cfg["train"] = {**cfg["train"], **v}
            else:
                cfg[k] = v
        return cfg

    def test_linear_eval_is_accepted(self):
        adapter.validate_eval(self.eval_config())          # must not raise

    def test_the_encoder_must_be_named(self):
        """An unresolved ${ENCODER} leaves it empty; a probe with no encoder
        would fit on default weights and report a number that looks like a
        result."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.validate_eval(self.eval_config(encoder=""))
        self.assertIn("encoder", str(e.exception))

    def test_pretrain_only_settings_are_not_part_of_the_probe(self):
        """`beta` is the VAE pretraining objective's knob; the probe never reads
        it, so a config that sets it here claims an effect it did not have."""
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.validate_eval(self.eval_config(train={"beta": 1.0}))
        self.assertIn("beta", str(e.exception))


class TestTheMetricsAreInTheVocabulary(unittest.TestCase):
    def test_pretrain_maps_a_pretext_loss(self):
        self.assertEqual(adapter.PRETRAIN_METRIC_NAMES["final_loss"],
                         "final_pretext_loss")
        self.assertIn("final_pretext_loss",
                      adapter.adapterlib.METRIC_VOCABULARY)

    def test_eval_maps_the_comparable_probe_numbers(self):
        targets = set(adapter.LINEAR_EVAL_METRIC_NAMES.values())
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, targets)
            self.assertIn(name, adapter.adapterlib.METRIC_VOCABULARY)


class TestTheEvalProducesNoEncoder(Base):
    def _reason(self, stage):
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps({"stage": stage}), encoding="utf-8")
        return adapter._absent_reason(p)

    def test_linear_eval_declares_no_encoder(self):
        self.assertIsNotNone(self._reason("linear_eval"))

    def test_pretrain_gives_no_reason(self):
        self.assertIsNone(self._reason("pretrain"))


class TestTheModelExposesMu(Base):
    @needs_torch
    def test_get_features_returns_one_mu_per_image(self):
        import torch
        vc = load("vae_cnn_mu", METHOD / "models" / "vae_cnn.py")
        m = vc.VAE_CNN(latent_dim=4, image_size=28)
        feats = m.get_features(torch.zeros(2, 3, 28, 28))
        self.assertEqual(tuple(feats.shape), (2, 4),
                         "the probe reads mu; it must be one latent per image")


class TestALinearEvalSmoke(Base):
    """The pipeline end to end: pretrain -> encoder.pt -> frozen linear probe.

    This is what was missing. It exercises the same MNIST the shipped pretrain
    uses (dataset-agnostic loader, MNIST branch), so the probe runs on the
    representation the port actually trains.
    """

    def _adapter(self, cfg_dict, out):
        cfg = self.tmp / (out.name + ".json")
        cfg.write_text(json.dumps(cfg_dict), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(out)], cwd=METHOD, env=env,
            capture_output=True, text=True)
        return cfg, r

    def eval_config(self, encoder) -> dict:
        return {"stage": "linear_eval", "seed": 0,
                "data_root": str(self.tmp / "data"), "device": "cpu",
                "encoder": str(encoder), "train": dict(EVAL_TRAIN)}

    @needs_torch
    def test_pretrain_then_probe_satisfies_the_contract(self):
        tiny_mnist(self.tmp / "data")
        pre = self.tmp / "pre"
        _, r = self._adapter(self.config(), pre)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        encoder = pre / "encoder.pt"
        self.assertTrue(encoder.is_file(), "pretrain produced no encoder.pt")

        eout = self.tmp / "eval"
        cfg, r = self._adapter(self.eval_config(encoder), eout)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(eout), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

        m = json.loads((eout / "metrics.json").read_text())["metrics"]
        self.assertIn("final_linear_probe_top1_accuracy", m)
        self.assertIn("final_linear_probe_top5_accuracy", m)
        self.assertFalse((eout / "encoder.pt").exists(),
                         "linear_eval must not write an encoder")
        man = json.loads((eout / "run_manifest.json").read_text())
        self.assertIn("encoder_absent_reason", man)


class TestTheOriginalIsUnchanged(unittest.TestCase):
    def test_the_body_lives_in_run_and_main_only_parses(self):
        import ast
        src = (METHOD / "train_pretrain_cnn.py").read_text()
        top = {n.name for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef)}
        for fn in ("run", "main"):
            self.assertIn(fn, top)
        main_src = src[src.index("def main("):]
        self.assertLess(len(main_src.splitlines()), 12,
                        "main() should parse and delegate, nothing more")

    def test_it_asks_for_deterministic_kernels(self):
        import ast
        src = (METHOD / "train_pretrain_cnn.py").read_text()
        run_fn = next(n for n in ast.parse(src).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run")
        called = {n.func.id for n in ast.walk(run_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("make_deterministic", called)

    def test_the_science_files_are_byte_identical_to_the_capture(self):
        import hashlib
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
        for rel in ("models/vae_cnn.py", "data/vae_dataset.py"):
            self.assertIn(rel, expected)


if __name__ == "__main__":
    unittest.main()
