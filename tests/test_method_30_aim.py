#!/usr/bin/env python3
"""Specification for 30_aim (AIM; El-Nouby et al., 2024; arXiv:2401.08541).

An **eval-only** port (a `linear_eval` stage and no step 1), the AIM sibling of
28_dinov2 / 36_franca. In the capture, AIM's "Step 1" is an as-is SSL comparison:
the official pretrained AIM-600M (ViT-H/14) is downloaded and a linear probe is fit
on frozen features (the average of the last few transformer blocks, patch
mean-pooled), because the from-scratch data (DFN-2B+, ~2B uncurated images) is not
public. That from-scratch autoregressive pretraining is the excluded step.

So this port ships no `encoder.pt` from training; `linear_eval` probes a frozen,
hash-pinned downloaded backbone -- a genuine SSL representation, so the number is
comparable. The upstream `apple/ml-aim` is pinned under `third_party/ml-aim`,
imported not copied. A real run needs the official checkpoint (a hash-pinned
download); the hermetic smoke builds a **random tiny** AIM, so nothing is
downloaded. Licence: apple-amlr (non-commercial research); nothing under it is
copied here (the code is a submodule, the weights a download).
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
from _checkout import needs_checkout         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHOD = ROOT / "methods" / "30_aim"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "ml-aim"
UPSTREAM_PKG = UPSTREAM / "aim-v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import huggingface_hub                             # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "30_aim needs torch, torchvision, huggingface_hub")

try:
    import torch as _torch                              # noqa: F401
    import torchvision as _tv                           # noqa: F401
    import numpy as _np                                 # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# The unified Step-2 is the lab's own torch-only AIM; it does not need the apple
# ml-aim package (huggingface_hub), unlike the eval-only Step-1.
needs_torch = unittest.skipUnless(
    HAVE_TORCH, "30_aim Step-2 needs torch, torchvision and numpy")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("aim_adapter", METHOD / "adapter" / "__init__.py")

# A frozen backbone small enough to probe on a CPU: a random tiny AIM (ckpt empty)
# -- 4 small blocks at 32px (patch 16 -> a 2x2 token grid), embed_dim 32, the last
# 2 blocks averaged. The shipped config pins the official AIM-600M (ViT-H/14,
# embed 1536, 24 blocks) via bin/fetch-weights.py.
EVAL_TRAIN = {"name": "aim_tiny", "ckpt": "", "img_size": 32, "patch_size": 16,
              "embed_dim": 32, "num_blocks": 4, "num_heads": 4,
              "num_feature_layers": 2,
              "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}

EMBED_DIM = 32

# The unified Step-2 (from-scratch) recipe, tiny for a CPU smoke: an AIM ViT with
# embed_dim 32 / depth 2 at 32px (patch 16 -> a 2x2 token grid), a 1-block head,
# the last block averaged. The shipped config is the real ViT-B/16 recipe.
VIT_EMBED = 32
UNIFIED_PRETRAIN = {
    "model": {"name": "aim_base_patch16", "img_size": 32, "patch_size": 16,
              "embed_dim": VIT_EMBED, "depth": 2, "num_heads": 2, "mlp_ratio": 2.0,
              "head_depth": 1, "head_dim": 32, "prefix_fraction_min": 0.10,
              "prefix_fraction_max": 0.50},
    "data": {"num_workers": 0},
    "training": {"epochs": 1, "batch_size": 2, "lr": 6.0e-4, "min_lr": 0.0,
                 "warmup_epochs": 0, "beta1": 0.9, "beta2": 0.95,
                 "weight_decay": 0.05, "grad_clip": 1.0, "save_at_epochs": []},
}
UNIFIED_EVAL = {"recipe": "unified", "img_size": 32, "patch_size": 16,
                "embed_dim": VIT_EMBED, "depth": 2, "num_heads": 2, "mlp_ratio": 2.0,
                "num_feature_layers": 2, "epochs": 2, "batch_size": 2,
                "num_workers": 0, "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}


def _deep_pretrain(**over) -> dict:
    import copy
    cfg = {"stage": "pretrain", "seed": 0, "device": "cpu",
           **copy.deepcopy(UNIFIED_PRETRAIN)}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def _submodule_present() -> bool:
    return (UPSTREAM_PKG / "aim" / "v1" / "torch" / "models.py").is_file()


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
        self.tmp = Path(tempfile.mkdtemp(prefix="aim-"))
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


class TestThePinnedUpstream(unittest.TestCase):
    @needs_checkout
    def test_the_adapter_records_the_checked_out_commit(self):
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("the submodule is not checked out here")
        self.assertEqual(r.stdout.strip(), adapter.UPSTREAM["commit"])

    def test_provenance_agrees_with_the_adapter(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(prov["upstream"]["commit"], adapter.UPSTREAM["commit"])
        self.assertIn("apple/ml-aim", adapter.UPSTREAM["repo"])
        self.assertNotIn("fork_of", prov["upstream"])

    def test_the_backbone_artifact_is_pinned_by_sha256(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        art = prov["backbone_artifact"]
        for key in ("url", "filename", "sha256", "bytes", "license"):
            self.assertIn(key, art)
        self.assertEqual(len(art["sha256"]), 64)
        self.assertTrue(art["url"].startswith("https://"))

    def test_the_licence_is_recorded_as_non_commercial(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        # The apple-amlr licence is non-commercial; the port must say so plainly.
        blob = json.dumps(prov).lower()
        self.assertIn("apple-amlr", blob)
        self.assertIn("non-commercial", blob)


class TestConfigTranslation(Base):
    def test_linear_eval_is_accepted(self):
        adapter.to_run_config(self.eval_config(), out=self.out)

    def test_it_has_pretrain_and_linear_eval(self):
        self.assertEqual(adapter.STAGES, ("pretrain", "linear_eval"))

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
            adapter.to_run_config(self.eval_config(stage="step3"), out=self.out)

    def test_a_config_that_sets_output_is_refused(self):
        cfg = self.eval_config()
        cfg["output"] = {"result_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))

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
        return load("aim_eval", METHOD / "evaluate_linear_aim.py")

    @needs_deps
    def test_the_feature_is_one_vector_per_image(self):
        import torch
        if not _submodule_present():
            self.skipTest("the ml-aim submodule is not checked out here")
        ev = self.evaluator()
        model = ev.build_model(UPSTREAM_PKG, dict(EVAL_TRAIN),
                               torch.device("cpu"))
        feats = ev.extract_feature(model, torch.zeros(2, 3, 32, 32),
                                   dict(EVAL_TRAIN), torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))


class TestTheDeviceIsResolved(Base):
    """Referenced by the device mutation spec."""

    def evaluator(self):
        return load("aim_eval", METHOD / "evaluate_linear_aim.py")

    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        ev = self.evaluator()
        with mock.patch.object(ev.torch.cuda, "is_available",
                               return_value=False):
            with self.assertRaises(RuntimeError):
                ev.resolve_device("cuda", 0)
            self.assertEqual(ev.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(ev.resolve_device("auto", 0).type, "cpu")

    @needs_deps
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        from unittest import mock
        ev = self.evaluator()
        with mock.patch.object(ev.torch.cuda, "is_available",
                               return_value=True):
            self.assertEqual(ev.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(ev.resolve_device("auto", 0).type, "cuda")


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
        if not _submodule_present():
            self.skipTest("the ml-aim submodule is not checked out here")
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_deps
    def test_it_reports_the_comparable_probe_numbers(self):
        if not _submodule_present():
            self.skipTest("the ml-aim submodule is not checked out here")
        self.run_adapter()
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        for name in ("best_linear_probe_top1_accuracy",
                     "final_linear_probe_top1_accuracy",
                     "best_linear_probe_top5_accuracy",
                     "final_linear_probe_top5_accuracy"):
            self.assertIn(name, m)

    @needs_deps
    def test_it_produces_no_encoder_and_says_so(self):
        if not _submodule_present():
            self.skipTest("the ml-aim submodule is not checked out here")
        self.run_adapter()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)

    @needs_deps
    def test_the_manifest_records_the_pinned_upstream(self):
        if not _submodule_present():
            self.skipTest("the ml-aim submodule is not checked out here")
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)

    @needs_deps
    def test_the_same_config_twice_gives_the_same_classifier(self):
        """The guarantee applies to this stage too: it has its own RNG --
        feature extraction shuffles and the probe is initialised -- so two runs
        of one config must agree bit for bit, compared by the manifest's
        recorded hashes over every artifact."""
        if not _submodule_present():
            self.skipTest("the ml-aim submodule is not checked out here")
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
        if not _submodule_present():
            self.skipTest("the ml-aim submodule is not checked out here")
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


class TestTheUnifiedPretrainConfig(Base):
    """The from-scratch Step-2 pretrain config (nested)."""

    def test_pretrain_is_accepted(self):
        built = adapter.to_run_config(_deep_pretrain(
            data_root=str(self.tmp / "data")), out=self.out)
        self.assertEqual(built["model"]["embed_dim"], VIT_EMBED)
        self.assertEqual(built["training"]["save_at_epochs"], [])
        self.assertEqual(built["data"]["train_path"],
                         str(self.tmp / "data" / "train"))

    def test_a_missing_block_key_is_refused_by_name(self):
        for block, key in (("training", "lr"), ("model", "embed_dim"),
                           ("model", "prefix_fraction_min"), ("data", "num_workers")):
            with self.subTest(key=key):
                cfg = _deep_pretrain(data_root=str(self.tmp / "data"))
                del cfg[block][key]
                with self.assertRaises(adapter.ConfigError) as e:
                    adapter.to_run_config(cfg, out=self.out)
                self.assertIn(key, str(e.exception))

    def test_an_unknown_block_key_is_refused(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"))
        cfg["training"]["mystery"] = 1
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("mystery", str(e.exception))

    def test_a_non_aim_base_name_is_refused(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"),
                             model={"name": "aim_huge"})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("aim_huge", str(e.exception))


class TestTheUnifiedEvalConfig(Base):
    def unified_eval_config(self, **over) -> dict:
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "encoder": str(self.tmp / "encoder.pt"), "train": dict(UNIFIED_EVAL)}
        for k, v in over.items():
            if k == "train" and v:
                cfg["train"] = {**cfg["train"], **v}
            elif k != "train":
                cfg[k] = v
        return cfg

    def test_unified_eval_is_accepted(self):
        adapter.to_run_config(self.unified_eval_config(), out=self.out)

    def test_unified_eval_needs_an_encoder(self):
        cfg = self.unified_eval_config()
        del cfg["encoder"]
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("encoder", str(e.exception))

    def test_a_download_key_on_the_unified_eval_is_refused(self):
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(self.unified_eval_config(train={"ckpt": ""}),
                                  out=self.out)
        self.assertIn("ckpt", str(e.exception))


class TestTheUnifiedModel(unittest.TestCase):
    def models(self):
        return load("aim_models", METHOD / "models" / "__init__.py")

    @needs_torch
    def test_extract_then_load_round_trips_the_trunk(self):
        import torch
        m = self.models()
        model = m.AIMViT(img_size=32, patch_size=16, embed_dim=VIT_EMBED, depth=2,
                         num_heads=2, mlp_ratio=2.0, head_depth=1, head_dim=32)
        enc_state = adapter.extract_encoder(model.state_dict())
        self.assertTrue(enc_state)
        self.assertFalse(any(k.startswith("predictor.") for k in enc_state))
        cfg = {"train": {"img_size": 32, "patch_size": 16, "embed_dim": VIT_EMBED,
                         "depth": 2, "num_heads": 2, "mlp_ratio": 2.0}}
        loaded = adapter.load_encoder(enc_state, cfg).state_dict()
        pairs = 0
        for key, want in enc_state.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


@needs_torch
class TestAUnifiedPretrainSmoke(Base):
    def run_adapter(self, **over):
        tiny_split(self.tmp / "data")          # provides data/train/{c0,c1}
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"), **over)
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return p, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    def test_it_completes_and_satisfies_the_contract(self):
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertTrue((self.out / "encoder.pt").is_file())

    def test_each_milestone_encoder_is_written(self):
        _, r = self.run_adapter(training={"epochs": 2, "save_at_epochs": [1, 2]})
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((self.out / "encoder_epoch1.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch2.pt").is_file())


@needs_torch
class TestAUnifiedLinearEvalSmoke(Base):
    def _step2_encoder(self) -> Path:
        first = TestAUnifiedPretrainSmoke("test_it_completes_and_satisfies_the_contract")
        first.tmp, first.out = self.tmp, self.tmp / "s2out"
        _, r = first.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        return first.out / "encoder.pt"

    def run_eval(self):
        enc = self._step2_encoder()
        cfg = {"stage": "linear_eval", "seed": 0,
               "data_root": str(self.tmp / "data"), "device": "cpu",
               "encoder": str(enc), "train": dict(UNIFIED_EVAL)}
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return p, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    def test_it_completes_and_reports_comparable_numbers(self):
        cfg, r = self.run_eval()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        m = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("final_linear_probe_top1_accuracy", m)
        self.assertFalse((self.out / "encoder.pt").exists())


if __name__ == "__main__":
    unittest.main()
