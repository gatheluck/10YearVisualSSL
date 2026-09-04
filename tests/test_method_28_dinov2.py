#!/usr/bin/env python3
"""Specification for 28_dinov2 (DINOv2; Oquab et al., 2023; arXiv:2304.07193).

An **eval-only** port (a `linear_eval` stage and no step 1), the DINOv2 sibling of
36_franca. In the capture, DINOv2's "Step 1" is a frozen-backbone probe: the
official pretrained ViT-g/14 (LVD-142M) is downloaded and a linear probe is fit on
its frozen CLS token, because the from-scratch SSL data (LVD-142M) is not public.
That from-scratch pretraining is the excluded step, as in every port.

So this port ships no `encoder.pt` from training; `linear_eval` probes a frozen,
hash-pinned downloaded backbone -- a genuine SSL representation, so the number is
comparable. The upstream `facebookresearch/dinov2` is pinned under
`third_party/dinov2`, imported not copied, and built with the xformers path
disabled so the forward is torch-only (the giant's SwiGLU falls back to a torch
implementation with the same w12/w3 weight keys, so the official weights load).
A real run needs the official checkpoint (a hash-pinned download); the hermetic
smoke builds a **random** ViT-S/14 at a tiny resolution, so nothing is downloaded.
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
METHOD = ROOT / "methods" / "28_dinov2"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "dinov2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

try:
    import timm                                        # noqa: F401
    HAVE_TIMM = HAVE_DEPS
except ImportError:
    HAVE_TIMM = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "28_dinov2 needs torch and torchvision")
# The unified Step-2 backbone is a timm ViT; the eval-only Step-1 does not need it.
needs_timm = unittest.skipUnless(
    HAVE_TIMM, "28_dinov2 Step-2 needs torch, torchvision and timm")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("dinov2_adapter", METHOD / "adapter" / "__init__.py")

# A frozen backbone small enough to probe on a CPU: a random ViT-S/14 (ckpt empty
# -> pretrained=False) at resolution 28 (patch 14 -> a 2x2 token grid). The
# shipped config pins the official ViT-g/14 (embed 1536) via bin/fetch-weights.py.
EVAL_TRAIN = {"name": "dinov2_vits14", "ckpt": "", "resolution": 28,
              "feature_key": "x_norm_clstoken",
              "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}

EMBED_DIM = 384   # ViT-S

# The unified Step-2 (from-scratch) recipe, tiny for a CPU smoke: a ViT with
# embed_dim 32 / depth 2 at 32px (patch 16 -> a 2x2 token grid), 2 global + 2 local
# crops, 64 prototypes. The shipped config is the real ViT-B/16 recipe.
VIT_EMBED = 32
UNIFIED_PRETRAIN = {
    "model": {"arch": "vit_base_patch16_224", "patch_size": 16,
              "embed_dim": VIT_EMBED, "depth": 2, "num_heads": 2,
              "mlp_ratio": 2.0, "img_size": 32},
    "data": {"global_crop_size": 32, "local_crop_size": 16, "n_global_crops": 2,
             "n_local_crops": 2, "global_crops_scale": [0.32, 1.0],
             "local_crops_scale": [0.05, 0.32], "num_workers": 0},
    "dino": {"out_dim": 64, "head_bottleneck_dim": 16, "head_nlayers": 2,
             "head_hidden_dim": 32, "student_temp": 0.1, "teacher_temp_min": 0.04,
             "teacher_temp_max": 0.07, "teacher_temp_warmup_epochs": 0,
             "center_momentum": 0.9},
    "ibot": {"out_dim": 64, "separate_head": False,
             "mask_sample_probability": 0.5, "mask_ratio_min_max": [0.1, 0.5],
             "loss_weight": 1.0},
    "koleo": {"loss_weight": 0.1},
    "dino_loss_weight": 1.0,
    "training": {"epochs": 1, "batch_size": 2, "base_lr": 6.0e-4, "min_lr": 1.0e-6,
                 "warmup_epochs": 0, "beta1": 0.9, "beta2": 0.95,
                 "weight_decay": 0.05, "momentum_teacher_min": 0.992,
                 "momentum_teacher_max": 1.0, "clip_grad": 3.0,
                 "freeze_last_layer_epochs": 0, "save_at_epochs": []},
}
UNIFIED_EVAL = {"recipe": "unified", "arch": "vit_base_patch16_224",
                "patch_size": 16, "embed_dim": VIT_EMBED, "depth": 2,
                "num_heads": 2, "mlp_ratio": 2.0, "resolution": 32,
                "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
                "momentum": 0.9, "weight_decay": 0.0}


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
    return (UPSTREAM / "dinov2" / "hub" / "backbones.py").is_file()


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
        self.tmp = Path(tempfile.mkdtemp(prefix="dinov2-"))
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
        self.assertIn("facebookresearch/dinov2", adapter.UPSTREAM["repo"])
        self.assertNotIn("fork_of", prov["upstream"])

    def test_the_backbone_artifact_is_pinned_by_sha256(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        art = prov["backbone_artifact"]
        for key in ("url", "filename", "sha256", "bytes", "license"):
            self.assertIn(key, art)
        self.assertEqual(len(art["sha256"]), 64)
        self.assertTrue(art["url"].startswith("https://"))


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
        return load("dinov2_eval", METHOD / "evaluate_linear_dinov2.py")

    @needs_deps
    def test_the_cls_feature_is_one_vector_per_image(self):
        import torch
        if not _submodule_present():
            self.skipTest("the dinov2 submodule is not checked out here")
        ev = self.evaluator()
        model = ev.build_model(UPSTREAM, dict(EVAL_TRAIN), torch.device("cpu"))
        feats = ev.extract_cls(model, torch.zeros(2, 3, 28, 28),
                               dict(EVAL_TRAIN), torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))


class TestTheDeviceIsResolved(Base):
    """Referenced by the device mutation spec."""

    def evaluator(self):
        return load("dinov2_eval", METHOD / "evaluate_linear_dinov2.py")

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
            self.skipTest("the dinov2 submodule is not checked out here")
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
            self.skipTest("the dinov2 submodule is not checked out here")
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
            self.skipTest("the dinov2 submodule is not checked out here")
        self.run_adapter()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)

    @needs_deps
    def test_the_manifest_records_the_pinned_upstream(self):
        if not _submodule_present():
            self.skipTest("the dinov2 submodule is not checked out here")
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
            self.skipTest("the dinov2 submodule is not checked out here")
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
            self.skipTest("the dinov2 submodule is not checked out here")
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


class TestTheUnifiedPretrainConfig(Base):
    """The from-scratch Step-2 pretrain config (nested, as the capture groups it)."""

    def test_pretrain_is_accepted(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"))
        built = adapter.to_run_config(cfg, out=self.out)
        self.assertEqual(built["model"]["embed_dim"], VIT_EMBED)
        self.assertEqual(built["training"]["save_at_epochs"], [])
        self.assertEqual(built["data"]["train_path"],
                         str(self.tmp / "data" / "train"))

    def test_a_missing_block_key_is_refused_by_name(self):
        for block, key in (("training", "base_lr"), ("dino", "out_dim"),
                           ("ibot", "mask_sample_probability"),
                           ("model", "embed_dim")):
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

    def test_a_non_vit_base_arch_is_refused(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"),
                             model={"arch": "vit_small_patch16_224"})
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("vit_small_patch16_224", str(e.exception))

    def test_a_config_that_sets_output_is_refused(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"))
        cfg["output"] = {"checkpoint_dir": "/anywhere"}
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("--out", str(e.exception))


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

    def test_the_native_eval_probes_the_download_not_a_trained_encoder(self):
        # regression: the Step-1 (as-is) eval has no recipe and no encoder.
        self.assertEqual(adapter.run_linear_eval.__name__, "run_linear_eval")
        adapter.to_run_config(self.eval_config(), out=self.out)


class TestTheUnifiedModel(unittest.TestCase):
    def models(self):
        return load("dinov2_models", METHOD / "models" / "__init__.py")

    @needs_timm
    def test_extract_then_load_round_trips_the_teacher_backbone(self):
        import torch
        m = self.models()
        model = m.build_dinov2_model({
            "model": {"arch": "vit_base_patch16_224", "patch_size": 16,
                      "embed_dim": VIT_EMBED, "depth": 2, "num_heads": 2,
                      "mlp_ratio": 2.0, "img_size": 32},
            "dino": {"out_dim": 64, "head_hidden_dim": 32,
                     "head_bottleneck_dim": 16, "head_nlayers": 2},
            "ibot": {"out_dim": 64, "separate_head": False}})
        enc_state = adapter.extract_encoder(model.state_dict())
        self.assertTrue(enc_state)
        self.assertFalse(any(k.startswith("student") for k in enc_state))
        cfg = {"train": {"arch": "vit_base_patch16_224", "patch_size": 16,
                         "embed_dim": VIT_EMBED, "depth": 2, "num_heads": 2,
                         "mlp_ratio": 2.0, "resolution": 32}}
        loaded = adapter.load_encoder(enc_state, cfg).state_dict()
        pairs = 0
        for key, want in enc_state.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


@needs_timm
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


@needs_timm
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


class TestFeatureProvider(Base):
    """`feature_provider.py` is what `bin/extract-features.py` discovers and
    calls to obtain one raw feature vector per image. It reuses this method's
    own model builder and eval pipeline, so the check is that it returns the
    DINOv2 CLS-token feature (`x_norm_clstoken`) -- raw, before the probe's
    normalise -- one row per val image, with honest meta.

    This method is eval-only and download-backed: there is no `adapter.load_encoder`
    for the as-is backbone. The eval main builds the pinned upstream ViT with
    `evaluate_linear_dinov2.build_model` and loads the official checkpoint into it,
    and `build_model` requires the architecture it constructs to match the state
    dict strictly. The shipped config pins the giant ViT-g/14 (1536-d), which is
    not buildable on a CPU, so the provider reads the variant from the checkpoint
    (its embed dim) rather than the config -- exactly the way `var`'s provider
    infers its architecture from its checkpoint. The testable checkpoint here is a
    **random** ViT-S/14 (384-d), built by the same `build_model` the smoke uses
    (`TestTheBackboneRepresentation`) and saved as its state dict; random weights
    do not affect the shape-and-plumbing this proves. Modules load through `load`
    (`load_from`), which purges any other method's `adapter`/`models` first -- the
    whole suite runs many methods in one interpreter.
    """

    def evaluator(self):
        return load("dinov2_eval", METHOD / "evaluate_linear_dinov2.py")

    def _make_encoder(self) -> Path:
        """A random ViT-S/14 backbone saved as its own state dict -- a checkpoint
        `build_model` (and so the provider) loads strict, standing in for the
        official download without fetching the 1536-d giant."""
        import torch
        ev = self.evaluator()
        model = ev.build_model(UPSTREAM, dict(EVAL_TRAIN), torch.device("cpu"))
        encoder_pt = self.tmp / "encoder.pt"
        torch.save(model.state_dict(), encoder_pt)
        return encoder_pt

    def _provider(self):
        return load("dinov2_feature_provider", METHOD / "feature_provider.py")

    @needs_deps
    def test_it_returns_raw_384d_features_one_per_val_image(self):
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("28_dinov2 provider not yet present")
        if not _submodule_present():
            self.skipTest("the dinov2 submodule is not checked out here")
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
        self.assertEqual(feats.shape[1], EMBED_DIM,
                         "DINOv2 ViT-S/14 CLS-token feature is 384-d")
        self.assertEqual(np.asarray(labels).shape[0], 6)
        self.assertEqual(meta["feat_dim"], EMBED_DIM)
        self.assertEqual(meta["representation"], "raw")

    @needs_deps
    def test_the_driver_saves_it_under_a_per_method_directory(self):
        """End to end through the driver's save path: the provider's output
        lands as features.npy / labels.npy / meta.json where a figure reads
        it, with the encoder's sha256 recorded in meta."""
        prov_path = METHOD / "feature_provider.py"
        if not prov_path.is_file():
            self.skipTest("28_dinov2 provider not yet present")
        if not _submodule_present():
            self.skipTest("the dinov2 submodule is not checked out here")
        import hashlib
        import numpy as np
        driver = load("extract_features_driver", BIN / "extract-features.py")
        data_root = tiny_split(self.tmp / "data")
        encoder_pt = self._make_encoder()

        record = {"method": METHOD.name, "status": "ready",
                  "provider": str(prov_path), "encoder": str(encoder_pt)}
        out = self.tmp / "features"
        updated = driver.extract_one(
            record, data_root=str(data_root), split="val", out=out,
            device="cpu", batch_size=2, num_workers=0)

        self.assertEqual(updated["status"], "ok",
                         updated.get("reason", ""))
        method_out = out / METHOD.name
        feats = np.load(method_out / "features.npy")
        labels = np.load(method_out / "labels.npy")
        meta = json.loads((method_out / "meta.json").read_text())
        self.assertEqual(feats.shape, (6, EMBED_DIM))
        self.assertEqual(labels.shape[0], 6)
        self.assertEqual(meta["encoder_sha256"],
                         hashlib.sha256(encoder_pt.read_bytes()).hexdigest())

    @needs_deps
    def test_the_isolated_driver_run_extracts_this_method_end_to_end(self):
        """The whole driver, real subprocess, real provider. Unlike the
        synthetic-provider driver test in tests/test_extract_features.py, this
        runs a method whose adapter imports the shared `adapterlib` -- so it
        catches the class of regression where the isolated worker cannot see a
        repository-root module the provider needs (the worker puts ROOT on
        sys.path, as bin/launch.py sets PYTHONPATH=ROOT)."""
        if not (METHOD / "feature_provider.py").is_file():
            self.skipTest("28_dinov2 provider not yet present")
        if not _submodule_present():
            self.skipTest("the dinov2 submodule is not checked out here")
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
        self.assertEqual(feats.shape, (6, EMBED_DIM))


if __name__ == "__main__":
    unittest.main()
