#!/usr/bin/env python3
"""Specification for cosmos3_super (NVIDIA Cosmos3-Super; nvidia/Cosmos3-Super).

A **pure eval-only** port, the `transformers`-sourced sibling of `data2vec2` and
`sam3`: a single `linear_eval` stage and no pretraining at all. Cosmos3-Super is a
video world-foundation model; the capture's Step-3 VideoGen evaluation probes it as
an as-is frozen backbone -- only its Qwen3-VL **vision encoder**'s patch tokens,
mean-pooled, fit by a linear classifier (the 64B MoT / DiT / VAE are never loaded).
The from-scratch pretraining is the excluded step, so the port reuses the released
checkpoint. The representation is a genuine learned feature, so the number is
comparable (the multimodal "pretrained-backbone reuse" row).

So this port ships no `encoder.pt` from training; `linear_eval` probes a frozen
backbone built from `transformers.Qwen3VLVisionModel`. There is **no author
submodule**: the model class is `transformers`' (a pinned pip dependency,
`transformers==5.16.1`), and the weights are a sha256-pinned download recorded as
`backbone_artifact` in provenance.json. Unlike `sam3`, the released checkpoint IS
directly loadable by the HF class -- `Qwen3VLVisionModel.from_pretrained(dir,
local_files_only=True)` on the `vision_encoder/` directory (config.json +
model.safetensors) -- so no trunk conversion is needed and a `save_pretrained` ->
`from_pretrained` round-trip is exact. The hermetic smoke leaves `ckpt` empty and
builds a **random tiny** `Qwen3VLVisionModel`, so nothing is downloaded. Licence:
OpenMDW-1.1 (public, not gated); nothing is copied here.
"""

from __future__ import annotations

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
METHOD = ROOT / "methods" / "cosmos3_super"
BIN = ROOT / "bin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import transformers                                # noqa: F401
    from transformers import Qwen3VLVisionModel        # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS,
    "cosmos3_super needs torch, torchvision and transformers "
    "(with Qwen3VLVisionModel)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("cosmos3_super_adapter", METHOD / "adapter" / "__init__.py")

# A frozen backbone small enough to probe on a CPU: a random tiny Cosmos3-Super
# vision tower (ckpt empty) -- 2 small blocks at 32px (patch 16 -> a 2x2 token
# grid), embed_dim 32, 2 heads. The shipped config pins the official
# nvidia/Cosmos3-Super vision encoder (hidden 1152, depth 27) via
# bin/fetch-weights.py.
EVAL_TRAIN = {"name": "nvidia/Cosmos3-Super", "ckpt": "", "img_size": 32,
              "patch_size": 16, "embed_dim": 32, "depth": 2, "num_heads": 2,
              "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}

EMBED_DIM = 32


def tiny_split(root: Path, per: int = 3) -> Path:
    """A labelled ImageFolder with train/ and val/, two classes each."""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    for split in ("train", "val"):
        for label, cls in enumerate(("c0", "c1")):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per):
                base = np.full((48, 48, 3), label * 120, dtype="uint8")
                noise = rng.randint(0, 64, (48, 48, 3), dtype="uint8")
                Image.fromarray((base + noise).astype("uint8")).save(
                    d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cosmos3-super-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"),
               "device": "cpu", "train": dict(EVAL_TRAIN)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg


class TestItIsAPureEvalOnlyPort(unittest.TestCase):
    def test_the_only_stage_is_linear_eval(self):
        self.assertEqual(adapter.STAGES, ("linear_eval",))

    def test_the_method_name_carries_no_number_and_no_step_word(self):
        self.assertEqual(adapter.METHOD, "cosmos3_super")

    def test_there_is_no_pinned_submodule(self):
        # The backbone is a transformers pip dependency, not a git submodule, so
        # the adapter records no UPSTREAM (and provenance records no upstream).
        self.assertFalse(hasattr(adapter, "UPSTREAM"))


class TestThePinnedBackbone(unittest.TestCase):
    def prov(self) -> dict:
        return json.loads((METHOD / "provenance.json").read_text())

    def test_the_backbone_artifact_is_pinned_by_sha256(self):
        art = self.prov()["backbone_artifact"]
        for key in ("url", "filename", "sha256"):
            self.assertIn(key, art)
        self.assertEqual(len(art["sha256"]), 64)
        self.assertTrue(art["url"].startswith("https://"))

    def test_provenance_records_no_submodule_upstream(self):
        # A no-submodule method must not claim a git upstream, or
        # test_port_completeness's both-places rule fails against the adapter.
        prov = self.prov()
        self.assertNotIn("upstream", prov)


class TestConfigTranslation(Base):
    def test_linear_eval_is_accepted(self):
        adapter.to_run_config(self.eval_config(), out=self.out)

    def test_a_missing_setting_is_refused_by_name(self):
        for key in EVAL_TRAIN:
            with self.subTest(key=key):
                cfg = self.eval_config()
                cfg["train"] = {k: v for k, v in EVAL_TRAIN.items() if k != key}
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_setting_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.eval_config(train={"grad_clip": 1.0}),
                                  out=self.out)
        self.assertIn("grad_clip", str(e.exception))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.eval_config(stage="pretrain"), out=self.out)

    def test_a_config_that_sets_output_is_refused(self):
        cfg = self.eval_config()
        cfg["output"] = {"result_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))

    def test_an_unknown_top_level_key_is_refused(self):
        cfg = self.eval_config()
        cfg["mystery"] = 1
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("mystery", str(e.exception))

    def test_an_unknown_device_is_refused(self):
        with self.assertRaises(adapter.ConfigError):
            adapter.to_run_config(self.eval_config(device="tpu"), out=self.out)


class TestTheEvalProducesNoEncoder(Base):
    def _reason(self, cfg):
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return adapter._absent_reason(p)

    def test_linear_eval_declares_no_encoder(self):
        self.assertTrue(self._reason(self.eval_config()))


class TestTheMetricsAreComparable(unittest.TestCase):
    def test_every_mapped_name_is_in_the_vocabulary(self):
        for target in adapter.LINEAR_EVAL_METRIC_NAMES.values():
            if target is not None:
                self.assertIn(target, adapterlib.METRIC_VOCABULARY)

    def test_the_probe_accuracies_are_comparable_names(self):
        mapped = set(adapter.LINEAR_EVAL_METRIC_NAMES.values())
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, mapped)
            self.assertEqual(adapterlib.METRIC_VOCABULARY[name],
                             adapterlib.COMPARABLE)


class TestTheBackboneRepresentation(Base):
    def evaluator(self):
        return load("cosmos3_super_eval",
                    METHOD / "evaluate_linear_cosmos3_super.py")

    @needs_deps
    def test_the_feature_is_one_vector_per_image(self):
        import torch
        ev = self.evaluator()
        model = ev.build_model(dict(EVAL_TRAIN), torch.device("cpu"))
        feats = ev.extract_feature(model, torch.zeros(2, 3, 32, 32),
                                   torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))


class TestTheCheckpointLoadsDirectly(Base):
    """The real-run path: the released checkpoint is directly loadable by the HF
    class -- `Qwen3VLVisionModel.from_pretrained` on the vision_encoder directory
    -- with no trunk conversion, and a `save_pretrained` -> `from_pretrained`
    round-trip is exact. So a real run does not leave the backbone randomly
    initialised, and the from_pretrained path is covered without the (public but
    1.1GB) official weights by building a tiny model, saving it, and loading it
    back."""

    def evaluator(self):
        return load("cosmos3_super_eval",
                    METHOD / "evaluate_linear_cosmos3_super.py")

    @needs_deps
    def test_a_saved_checkpoint_directory_loads_and_forwards_finite(self):
        import torch
        ev = self.evaluator()
        # Build a tiny random tower and persist it as the official layout does:
        # a directory with config.json + model.safetensors.
        random_model = ev.build_model(dict(EVAL_TRAIN), torch.device("cpu"))
        ckpt_dir = self.tmp / "vision_encoder"
        random_model.save_pretrained(str(ckpt_dir))
        self.assertTrue((ckpt_dir / "config.json").is_file())
        self.assertTrue((ckpt_dir / "model.safetensors").is_file())

        # A ckpt path -> load it back (not a fresh random init). The round-trip
        # is exact, so the two towers agree bit for bit on the same input.
        loaded = ev.build_model({**EVAL_TRAIN, "ckpt": str(ckpt_dir)},
                                torch.device("cpu"))
        x = torch.zeros(2, 3, 32, 32)
        a = ev.extract_feature(random_model, x, torch.device("cpu"))
        b = ev.extract_feature(loaded, x, torch.device("cpu"))
        self.assertTrue(bool(torch.isfinite(b).all()))
        self.assertEqual(tuple(b.shape), (2, EMBED_DIM))
        self.assertLess(float((a - b).abs().max()), 1e-4)

    @needs_deps
    def test_a_missing_checkpoint_directory_is_refused(self):
        import torch
        ev = self.evaluator()
        with self.assertRaises((FileNotFoundError, OSError, RuntimeError, ValueError)):
            ev.build_model({**EVAL_TRAIN, "ckpt": str(self.tmp / "nope")},
                           torch.device("cpu"))


class TestTheDeviceIsResolved(Base):
    """Referenced by the device mutation spec."""

    def evaluator(self):
        return load("cosmos3_super_eval",
                    METHOD / "evaluate_linear_cosmos3_super.py")

    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        ev = self.evaluator()
        with mock.patch.object(ev.torch.cuda, "is_available",
                               return_value=False):
            with self.assertRaises(RuntimeError):
                ev.resolve_device("cuda")
            self.assertEqual(ev.resolve_device("cpu").type, "cpu")
            self.assertEqual(ev.resolve_device("auto").type, "cpu")

    @needs_deps
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        from unittest import mock
        ev = self.evaluator()
        with mock.patch.object(ev.torch.cuda, "is_available",
                               return_value=True):
            self.assertEqual(ev.resolve_device("cpu").type, "cpu")
            self.assertEqual(ev.resolve_device("auto").type, "cuda")


class TestALinearEvalSmoke(Base):
    def run_adapter(self, **over):
        tiny_split(self.tmp / "data")
        cfg = self.tmp / "resolved.json"
        cfg.write_text(json.dumps(self.eval_config(**over)), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return cfg, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(cfg),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    @needs_deps
    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_reports_the_comparable_probe_numbers(self):
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, m)

    @needs_deps
    def test_it_produces_no_encoder_and_says_so(self):
        self.run_adapter()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)

    @needs_deps
    def test_the_same_config_twice_gives_the_same_classifier(self):
        """Two runs of one config must agree bit for bit, compared by the
        manifest's recorded hashes over every artifact."""
        base = self.tmp
        digests = []
        for name in ("a", "b"):
            self.out = base / name
            self.run_adapter()
            man = json.loads((self.out / "run_manifest.json").read_text())
            digests.append({a["path"]: a["sha256"] for a in man["artifacts"]})
        self.assertEqual(digests[0], digests[1])

    @unittest.skipUnless(HAVE_DEPS and torch.cuda.is_available(),
                         "no CUDA device; the GPU path cannot be exercised here")
    def test_the_probe_runs_on_cuda(self):
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


# The feature-provider smoke builds a random tiny Cosmos3-Super vision tower and
# persists it in the official layout (a vision_encoder/ directory with config.json
# + model.safetensors), then hands the provider the model.safetensors file -- the
# single-file `backbone_artifact` the driver hashes. The provider takes its parent
# directory and loads it with from_pretrained. The tiny tower's hidden_size is 32,
# so the mean-pooled patch-token feature is 32-d (the real tower is 1152-d).
PROVIDER_DIM = EMBED_DIM


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. cosmos3_super is eval-only,
    so it has no `adapter.load_encoder`: the provider sets the shipped config's
    `ckpt` to the handed-in checkpoint and lets the eval module's `build_model`
    load it with `Qwen3VLVisionModel.from_pretrained`, then extracts the
    mean-pooled patch-token feature -- raw, before the probe's normalise -- one
    hidden_size-d row per val image, with honest meta.

    The released checkpoint is a `vision_encoder/` directory (config.json +
    model.safetensors), but the driver hands (and hashes) a single file, and the
    `backbone_artifact` is exactly the single `model.safetensors`. So the encoder
    fixture is built by saving a tiny random tower with `save_pretrained` and
    passing its `model.safetensors` file as `encoder_path`; the provider takes the
    parent directory for `from_pretrained`. Random weights do not affect the
    shape-and-plumbing this proves.
    """

    def evaluator(self):
        return load("cosmos3_super_eval",
                    METHOD / "evaluate_linear_cosmos3_super.py")

    def _make_encoder(self) -> Path:
        """A tiny random tower saved in the official vision_encoder/ layout;
        returns the model.safetensors file (the driver's single-file encoder)."""
        import torch
        ev = self.evaluator()
        model = ev.build_model(dict(EVAL_TRAIN), torch.device("cpu"))
        ckpt_dir = self.tmp / "vision_encoder"
        model.save_pretrained(str(ckpt_dir))
        safetensors = ckpt_dir / "model.safetensors"
        self.assertTrue(safetensors.is_file())
        return safetensors

    def _provider(self):
        return load("cosmos3_super_feature_provider",
                    METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_features_one_per_val_image(self):
        if not (METHOD / "feature_provider.py").is_file():
            self.skipTest("cosmos3_super provider not yet present")
        import numpy as np
        data_root = tiny_split(self.tmp / "data")
        encoder_pt = self._make_encoder()

        prov = self._provider()
        feats, labels, meta = prov.extract_val_features(
            encoder_path=str(encoder_pt), data_root=str(data_root),
            split="val", device="cpu", batch_size=2, num_workers=0)

        feats = np.asarray(feats)
        self.assertEqual(feats.ndim, 2)
        self.assertEqual(feats.shape[0], 6, "6 val images expected")
        self.assertEqual(feats.shape[1], PROVIDER_DIM,
                         "Cosmos3-Super patch-token-mean feature is hidden_size-d")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], PROVIDER_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads it,
        with the encoder's sha256 recorded in meta."""
        if not (METHOD / "feature_provider.py").is_file():
            self.skipTest("cosmos3_super provider not yet present")
        import hashlib
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        data_root = tiny_split(self.tmp / "data")
        encoder_pt = self._make_encoder()

        record = {"method": METHOD.name, "status": "ready",
                  "provider": str(METHOD / "feature_provider.py"),
                  "encoder": str(encoder_pt)}
        out = self.tmp / "features"
        updated = driver.extract_one(
            record, data_root=str(data_root), split="val", out=out,
            device="cpu", batch_size=2, num_workers=0)

        self.assertEqual(updated["status"], "ok", updated.get("reason", ""))
        method_out = out / METHOD.name
        feats = np.load(method_out / "features.npy")
        labels = np.load(method_out / "labels.npy")
        meta = json.loads((method_out / "meta.json").read_text())
        self.assertEqual(feats.shape, (6, PROVIDER_DIM))
        self.assertEqual(labels.shape[0], 6)
        self.assertEqual(meta["encoder_sha256"],
                         hashlib.sha256(encoder_pt.read_bytes()).hexdigest())

    @needs_deps
    def test_the_isolated_driver_run_extracts_this_method_end_to_end(self):
        """The whole driver, real subprocess, real provider -- catches the class
        of regression where the isolated worker cannot see a repository-root
        module the provider needs (the worker puts ROOT on sys.path)."""
        if not (METHOD / "feature_provider.py").is_file():
            self.skipTest("cosmos3_super provider not yet present")
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        data_root = tiny_split(self.tmp / "data")
        encoder_pt = self._make_encoder()
        out = self.tmp / "features"
        manifest = driver.run(
            METHOD.parent, data_root=str(data_root), split="val", out=out,
            encoders={METHOD.name: str(encoder_pt)}, encoders_root=None,
            device="cpu", batch_size=2, num_workers=0,
            venvs_root=ROOT / ".venvs")

        rec = {r["method"]: r for r in manifest["records"]}[METHOD.name]
        self.assertEqual(rec["status"], "ok", rec.get("reason", ""))
        feats = np.load(out / METHOD.name / "features.npy")
        self.assertEqual(feats.shape, (6, PROVIDER_DIM))


if __name__ == "__main__":
    unittest.main()
