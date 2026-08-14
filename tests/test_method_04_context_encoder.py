#!/usr/bin/env python3
"""Specification for 04_context_encoder (Pathak et al., 2016), step 1 + linear eval.

Context Encoder learns features by inpainting a centre hole with a conv
encoder-decoder, optionally with a centre-hole adversarial discriminator. The
representation is the encoder plus its 4096-d bottleneck.

What is new here, rather than repeated from the earlier ports:

- **It is a GAN.** Step 1 trains a generator (the encoder-decoder) and, when
  adversarial training is on, a discriminator with its own Adam optimiser. The
  reconstruction and adversarial losses are components with no contract slot, so
  they map to `None`, like iBOT's cls/patch losses.
- **Nothing came across verbatim.** The capture interleaves step 1 (AlexNet) and
  step 2 (ViT) in single files, so every ported file was rewritten to extract a
  clean step 1; `captured_sha256` is empty and each file's captured digest is
  recorded under `rewritten_during_the_port`.
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
METHOD = ROOT / "methods" / "04_context_encoder"
BIN = ROOT / "bin"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import PIL                                         # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_torch = unittest.skipUnless(
    HAVE_DEPS, "this method needs torch, torchvision and Pillow")

try:
    import timm                                          # noqa: F401
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

needs_timm = unittest.skipUnless(
    HAVE_DEPS and HAVE_TIMM, "the ViT Step-2 path needs timm (arch: vit)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("ce_adapter", METHOD / "adapter" / "__init__.py")

# Small enough to run on a CPU in seconds. The decoder is fixed to a 128x128
# output, so mask_size stays 128; img_size only needs to exceed it. The AlexNet
# backbone is the real one (an adaptive pool makes the bottleneck size-agnostic).
TRAIN = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.01,
         "momentum": 0.9, "weight_decay": 1.0e-4, "warmup_epochs": 0,
         "img_size": 160, "mask_size": 128, "loss_type": "l2",
         "use_adversarial": True, "adversarial_weight": 0.001,
         "save_freq": 1, "print_freq": 1}

EVAL_TRAIN = {"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0, "img_size": 160}


def tiny_imagenet(root: Path, n: int = 4) -> Path:
    """A few synthetic 256x256 images in the ImageFolder layout the loader
    walks (Resize(256) + crop). `n` must be at least the batch size."""
    from PIL import Image
    import random
    rng = random.Random(0)
    d = root / "train" / "class0"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (256, 256))
        img.putdata([(rng.randrange(256), rng.randrange(256),
                      rng.randrange(256)) for _ in range(256 * 256)])
        img.save(d / f"{i}.jpg")
    return root


def tiny_classified(root: Path, classes: int = 2, per_class: int = 2) -> Path:
    """A labelled ImageFolder tree, separable so the best-accuracy save is not
    decided by cross-hardware float noise."""
    from PIL import Image
    import random
    rng = random.Random(0)
    for split in ("train", "val"):
        for c in range(classes):
            d = root / split / f"c{c}"
            d.mkdir(parents=True, exist_ok=True)
            base = int(30 + c * (200 / max(classes - 1, 1)))
            for i in range(per_class):
                img = Image.new("RGB", (256, 256))
                img.putdata([(min(255, max(0, base + rng.randrange(-8, 9))),) * 3
                             for _ in range(256 * 256)])
                img.save(d / f"{i}.jpg")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ce-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def config(self, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": copy.deepcopy(TRAIN)}
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
    def test_a_config_naming_an_output_location_is_refused(self):
        for key in ("checkpoint", "output"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(**{key: {"x": 1}}),
                                          self.out)
                self.assertIn("--out", str(e.exception))

    def test_every_key_the_stage_reads_must_be_declared(self):
        cfg = self.config()
        del cfg["train"]["loss_type"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("loss_type", str(e.exception))

    def test_a_key_the_stage_never_reads_is_refused(self):
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

    def test_an_unknown_loss_type_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.config(train={"loss_type": "huber"}),
                                  self.out)
        self.assertIn("huber", str(e.exception))

    def test_train_must_be_a_mapping(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.config(train=None), self.out)

    def test_the_output_goes_under_out(self):
        run = adapter.to_run_config(self.config(), self.out)
        self.assertTrue(str(Path(run["checkpoint"]["save_dir"]))
                        .startswith(str(self.out)))

    def test_the_data_root_is_the_parent_not_the_train_dir(self):
        """InpaintingDataset joins root + split itself, so train_path must be
        the parent; appending /train would double it."""
        run = adapter.to_run_config(self.config(), self.out)
        self.assertEqual(run["data"]["train_path"], str(self.tmp / "data"))


class TestTheEncoderIsTheEncoderAndBottleneck(Base):
    def test_only_encoder_and_bottleneck_come_across(self):
        state = {"encoder.0.weight": 1, "fc.0.weight": 2,
                 "decoder.0.weight": 3, "decoder_fc.weight": 4}
        got = adapter.extract_encoder(state)
        self.assertEqual(set(got), {"encoder.0.weight", "fc.0.weight"})

    def test_an_empty_result_is_an_error(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.extract_encoder({"decoder.0.weight": 1})
        self.assertIn("encoder", str(e.exception).lower())

    def test_a_ddp_prefix_is_stripped(self):
        got = adapter.extract_encoder({"module.encoder.0.weight": 1,
                                       "module.fc.0.weight": 2})
        self.assertEqual(set(got), {"encoder.0.weight", "fc.0.weight"})


class TestTheMetricNames(Base):
    def test_every_mapped_name_is_in_the_contract_vocabulary(self):
        for names in (adapter.PRETRAIN_METRIC_NAMES,
                      adapter.LINEAR_EVAL_METRIC_NAMES):
            for raw, target in names.items():
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
        for raw in ("final_recon_loss", "final_adv_loss"):
            with self.subTest(metric=raw):
                self.assertIn(raw, adapter.PRETRAIN_METRIC_NAMES)
                self.assertIsNone(adapter.PRETRAIN_METRIC_NAMES[raw])

    def test_no_probe_name_in_step1(self):
        for target in adapter.PRETRAIN_METRIC_NAMES.values():
            self.assertNotIn("linear_probe", str(target))


class TestTheTrainingCallIsTheOriginals(Base):
    def test_the_original_run_is_called_once(self):
        calls = []

        def fake_run(args, config=None):
            Path(config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                         exist_ok=True)
            calls.append(args)
            return {"epochs": 1, "final_loss": 2.0, "final_recon_loss": 1.9,
                    "final_adv_loss": 6.0}

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


class TestWhichCheckpointIsTheFinalOne(Base):
    def make(self, *epochs):
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


@needs_torch
class TestTheDeviceIsResolved(Base):
    def trainer(self):
        return load("ce_trainer", METHOD / "train_pretrain.py")

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


@needs_torch
class TestASmokeRun(Base):
    def setUp(self) -> None:
        super().setUp()
        tiny_imagenet(self.tmp / "data")

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_a_real_run_on_cuda_produces_a_loadable_encoder(self):
        """The GPU path, on real hardware. A tensor left on the wrong device
        raises inside training, so a run that finishes with a non-empty encoder
        is the GPU path working end to end."""
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
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["status"], "ok")

    def test_the_encoder_is_the_encoder_and_bottleneck(self):
        import torch
        self.run_adapter()
        state = torch.load(self.out / "encoder.pt", map_location="cpu",
                           weights_only=True)
        self.assertTrue(state, "the encoder is empty")
        self.assertTrue(all(k.startswith(("encoder.", "fc.")) for k in state),
                        sorted(state)[:5])
        self.assertFalse([k for k in state
                          if k.startswith(("decoder.", "decoder_fc."))])

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
        for raw in ("final_recon_loss", "final_adv_loss"):
            self.assertIn(raw, doc["metrics_raw"])
            self.assertNotIn(raw, doc["metrics"])

    def test_the_originals_scratch_files_stay_inside_out(self):
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        listed = sorted(a["path"] for a in man["artifacts"])
        on_disk = sorted(str(p.relative_to(self.out))
                         for p in self.out.rglob("*") if p.is_file()
                         and p.name != "run_manifest.json")
        self.assertEqual(listed, on_disk)
        self.assertTrue([p for p in on_disk if p.startswith("work/")])

    def test_the_same_config_twice_gives_the_same_encoder(self):
        digests = []
        for i in range(2):
            self.out = self.tmp / f"out{i}"
            _, r = self.run_adapter()
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            digests.append(hashlib.sha256(
                (self.out / "encoder.pt").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1],
                         "two runs of one config produced different weights")

    @needs_torch
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
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


class TestTheLinearEvaluationStage(Base):
    def config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "encoder": str(self.tmp / "encoder.pt"),
               "device": "cpu", "train": copy.deepcopy(EVAL_TRAIN)}
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
        cfg = self.config()
        cfg["train"]["loss_type"] = "l2"
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, self.out)
        self.assertIn("loss_type", str(e.exception))

    def test_no_pretext_name_is_produced_here(self):
        for target in adapter.LINEAR_EVAL_METRIC_NAMES.values():
            self.assertNotIn("pretext", str(target))

    def test_it_reports_all_four_comparable_accuracies(self):
        accuracies = sorted(
            k for k, t in adapter.LINEAR_EVAL_METRIC_NAMES.items()
            if t and "accuracy" in t)
        self.assertEqual(accuracies, ["best_top1_acc",
                                      "best_top5_acc_at_best_top1",
                                      "final_top1_acc", "final_top5_acc"])
        for target in adapter.LINEAR_EVAL_METRIC_NAMES.values():
            if target and "accuracy" in target:
                self.assertEqual(adapterlib.METRIC_VOCABULARY[target],
                                 adapterlib.COMPARABLE)

    @needs_torch
    def test_an_encoder_missing_its_backbone_is_refused(self):
        with self.assertRaises(RuntimeError) as e:
            adapter.load_encoder({}, self.config())
        self.assertIn("encoder", str(e.exception).lower())

    @needs_torch
    def test_asking_for_the_vit_is_refused_by_name(self):
        evaluation = load("ce_eval", METHOD / "evaluate_linear.py")
        with self.assertRaises(NotImplementedError) as e:
            evaluation.load_encoder(str(self.tmp / "any.pth"), "vit")
        self.assertIn("step 2", str(e.exception))

    @needs_torch
    def test_the_evaluation_refuses_a_gpu_it_does_not_have(self):
        import torch
        evaluation = load("ce_eval", METHOD / "evaluate_linear.py")
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
        self.assertFalse([k for k in doc["metrics"] if "pretext" in k])
        self.assertIn("final_top1_acc", doc["metrics_raw"])

    def test_it_says_why_there_is_no_encoder_to_hand_on(self):
        self.make_encoder()
        self.run_adapter(self.config())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertFalse((self.out / "encoder.pt").exists())
        self.assertTrue(man["encoder_absent_reason"].strip())
        self.assertEqual(man["stage"], "linear_eval")


class TestWhatCameFromTheCapture(unittest.TestCase):
    def test_the_provenance_records_the_capture(self):
        doc = json.loads((METHOD / "provenance.json").read_text())
        self.assertIn("CapturePrivate", doc["captured_from"])

    def test_nothing_is_pinned_because_everything_was_rewritten(self):
        """This method's captured files interleave step1 and step2, so none came
        across verbatim. Instead every rewritten file records its captured
        digest, so the port is still traceable to the capture."""
        doc = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(doc["captured_sha256"], {})
        rewritten = doc["rewritten_during_the_port"]
        for rel in ("models/context_encoder.py", "datasets.py",
                    "train_pretrain.py", "evaluate_linear.py"):
            with self.subTest(file=rel):
                self.assertIn(rel, rewritten)
                self.assertIn("sha256", rewritten[rel])


# --- Step 2: unified ViT-B/16 (arch: vit), additive alongside the native
# AlexNet path. The centre-hole inpainting task on a ViT-B/16 encoder + a
# transformer decoder, always adversarial (two AdamW optimisers). encoder.pt
# keeps only the ViT trunk (encoder.*); the probe reads the mean patch-token
# feature. Tiny dims so a CPU smoke is cheap (image 32, hole 16, patch 16).
VIT_MODEL_ARGS = {"image_size": 32, "patch_size": 16, "in_channels": 3,
                  "embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                  "decoder_dim": 16, "decoder_depth": 1, "decoder_heads": 2,
                  "hole_size": 16}
VIT_TRAIN_TINY = {"arch": "vit", **VIT_MODEL_ARGS, "epochs": 2, "batch_size": 2,
                  "num_workers": 0, "lr": 6.0e-4, "weight_decay": 0.05,
                  "warmup_epochs": 0, "min_lr": 0.0, "clip_grad": 1.0,
                  "adversarial_weight": 0.001, "save_at_epochs": [1, 2]}
VIT_EVAL_TINY = {"arch": "vit", **VIT_MODEL_ARGS, "epochs": 1, "batch_size": 2,
                 "num_workers": 0, "lr": 0.1, "momentum": 0.9,
                 "weight_decay": 0.0}
FEATURE_DIM_VIT = VIT_MODEL_ARGS["embed_dim"]
N_PATCHES = (VIT_MODEL_ARGS["image_size"] // VIT_MODEL_ARGS["patch_size"]) ** 2
PATCH_PIXELS = VIT_MODEL_ARGS["patch_size"] ** 2 * VIT_MODEL_ARGS["in_channels"]


class TestVitConfigTranslation(Base):
    def vit_config(self, train=None, **over) -> dict:
        cfg = {"stage": "pretrain", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "train": copy.deepcopy(train if train is not None
                                      else VIT_TRAIN_TINY)}
        for k, v in over.items():
            cfg[k] = v
        return cfg

    def test_the_vit_step2_config_is_accepted(self):
        built = adapter.to_run_config(self.vit_config(), self.out)
        self.assertEqual(built["arch"], "vit")
        self.assertEqual(built["model"]["embed_dim"], 16)
        self.assertEqual(built["model"]["hole_size"], 16)
        self.assertEqual(built["training"]["save_at_epochs"], [1, 2])

    def test_the_native_path_has_no_top_level_arch(self):
        built = adapter.to_run_config(self.config(), self.out)
        self.assertNotIn("arch", built)

    def test_a_bad_arch_is_refused_by_name(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(
                self.vit_config(train={**VIT_TRAIN_TINY, "arch": "vitt"}),
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

    def test_a_native_knob_does_not_leak_into_the_vit_path(self):
        for key in ("loss_type", "mask_size", "use_adversarial", "momentum",
                    "save_freq"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(
                        self.vit_config(train={**VIT_TRAIN_TINY, key: 1}),
                        self.out)
                self.assertIn(key, str(e.exception))

    def test_a_vit_knob_does_not_leak_into_the_native_path(self):
        for key in ("embed_dim", "hole_size", "min_lr", "save_at_epochs"):
            with self.subTest(key=key):
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(self.config(train={key: 1}), self.out)
                self.assertIn(key, str(e.exception))


class TestTheVitModel(unittest.TestCase):
    def _model(self):
        vm = load("vit_context_encoder",
                  METHOD / "models" / "vit_context_encoder.py")
        return vm.build_vit_context_encoder(**VIT_MODEL_ARGS)

    @needs_timm
    def test_forward_returns_pred_mask_and_features(self):
        import torch
        pred, mask, feat = self._model()(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(pred.shape), (2, N_PATCHES, PATCH_PIXELS))
        self.assertEqual(tuple(mask.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(feat.shape), (2, N_PATCHES, FEATURE_DIM_VIT))

    @needs_timm
    def test_get_features_is_the_mean_patch_token(self):
        import torch
        feats = self._model().get_features(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(feats.shape), (2, FEATURE_DIM_VIT))

    @needs_timm
    def test_the_hole_extractors_agree_on_shape(self):
        import torch
        model = self._model()
        pred, _, _ = model(torch.randn(2, 3, 32, 32))
        ph = model.extract_predicted_hole(pred)
        th = model.extract_target_hole(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(ph.shape), (2, 3, 16, 16))
        self.assertEqual(tuple(th.shape), (2, 3, 16, 16))

    @needs_timm
    def test_encoder_pt_holds_only_the_trunk(self):
        got = adapter.extract_encoder(self._model().state_dict())
        self.assertTrue(got)
        self.assertTrue(all(k.startswith("encoder.") for k in got))
        for stray in ("decoder", "mask_token", "decoder_pred", "decoder_embed",
                      "decoder_pos_embed"):
            self.assertFalse([k for k in got if k.startswith(stray)])

    @needs_timm
    def test_load_encoder_round_trips_the_trunk(self):
        import torch
        saved = adapter.extract_encoder(self._model().state_dict())
        cfg = {"train": {"arch": "vit", **VIT_MODEL_ARGS}}
        model = adapter.load_encoder(saved, cfg)
        loaded = model.state_dict()
        pairs = 0
        for k, want in saved.items():
            got = loaded.get(k)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{k} came back changed")
        self.assertGreater(pairs, 0, "no saved weight reached the trunk")


class TestAVitStep2Smoke(Base):
    def _adapter(self, cfg_dict, out):
        import os
        cfg = self.tmp / (out.name + ".json")
        cfg.write_text(json.dumps(cfg_dict), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(out)], cwd=METHOD, env=env,
            capture_output=True, text=True)
        return cfg, r

    @needs_timm
    def test_pretrain_milestones_then_probe_passes_contract(self):
        tiny_imagenet(self.tmp / "data")
        pre = self.tmp / "pre_out"
        _, r = self._adapter(
            {"stage": "pretrain", "seed": 0,
             "data_root": str(self.tmp / "data"), "device": "cpu",
             "train": copy.deepcopy(VIT_TRAIN_TINY)}, pre)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((pre / "encoder.pt").is_file())
        for n in (1, 2):
            self.assertTrue((pre / f"encoder_epoch{n}.pt").is_file(),
                            f"milestone encoder_epoch{n}.pt not written")

        tiny_classified(self.tmp / "eval")
        ev = self.tmp / "eval_out"
        cfg, r = self._adapter(
            {"stage": "linear_eval", "seed": 0,
             "data_root": str(self.tmp / "eval"), "device": "cpu",
             "encoder": str(pre / "encoder_epoch2.pt"),
             "train": copy.deepcopy(VIT_EVAL_TINY)}, ev)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out", str(ev),
             "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        m = json.loads((ev / "metrics.json").read_text())["metrics"]
        self.assertIn("final_linear_probe_top1_accuracy", m)
        self.assertFalse((ev / "encoder.pt").exists())


if __name__ == "__main__":
    unittest.main()
