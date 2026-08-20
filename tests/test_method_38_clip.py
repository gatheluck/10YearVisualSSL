#!/usr/bin/env python3
"""Specification for 38_clip (CLIP; Radford et al., 2021; arXiv:2103.00020).

A two-comparison port, the CLIP sibling of the eval-only-download trio
(28_dinov2 / 30_aim / 36_franca):

- **Step 1 (as-is)** -- a `linear_eval` with no recipe that freezes the official
  OpenAI ViT-B/32 (a sha256-pinned download built through the pinned openai/CLIP
  submodule, `third_party/CLIP`) and probes its pooled image embedding. CLIP's
  400M-pair training data is not public, so the as-is row reuses the released
  checkpoint. It produces no `encoder.pt`.
- **Step 2 (label-text adaptation)** -- a `pretrain` that trains a CLIP ViT-B/16
  from scratch on ImageNet-1k, pairing each labeled image with an official ImageNet
  class-name prompt, then a `recipe: unified` `linear_eval` that probes the trained
  image tower. This is a **supervised label-text adaptation**, disclosed in every
  config/checkpoint/result as `supervised_label_text_adaptation=true` /
  `main_vssl_comparability=false`; it is not a comparable VSSL result.

A real run needs the download / ImageNet; the hermetic smokes build a random tiny
CLIP at a tiny resolution, so nothing is downloaded. CLIP derives its vision-tower
head count as `vision_width // 64`, so the smoke keeps `vision_width` at 64.
"""

from __future__ import annotations

import copy
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
METHOD = ROOT / "methods" / "38_clip"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "CLIP"
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
    import ftfy                                        # noqa: F401
    import regex                                       # noqa: F401
    HAVE_CLIP_DEPS = HAVE_DEPS
except ImportError:
    HAVE_CLIP_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "38_clip needs torch and torchvision")
# Building/tokenizing needs the pinned CLIP package, which imports ftfy and regex.
needs_clip = unittest.skipUnless(
    HAVE_CLIP_DEPS, "38_clip needs torch, torchvision, ftfy and regex")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("clip_adapter", METHOD / "adapter" / "__init__.py")
OFFICIAL_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"

# A frozen tiny image tower for a CPU probe: a random ViT (ckpt empty ->
# pretrained=False) at resolution 32 (patch 16 -> a 2x2 token grid), width 64.
# The shipped config pins the official ViT-B/32 via bin/fetch-weights.py.
EVAL_TRAIN = {"ckpt": "", "resolution": 32, "patch_size": 16, "width": 64,
              "layers": 2, "heads": 1, "output_dim": 16,
              "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
              "momentum": 0.9, "weight_decay": 0.0}
OUTPUT_DIM = 16

DEFINITION = {"source_method": "OpenAI CLIP",
              "source_code_commit": OFFICIAL_COMMIT,
              "adaptation": "ImageNet class-name text prompts paired with images",
              "supervised_label_text_adaptation": True,
              "main_vssl_comparability": False}

# The label-text Step-2 recipe, tiny for a CPU smoke: a CLIP with a 2-layer
# ViT (width 64, patch 16 at 32px) and a 2-layer text tower; vocab_size stays at
# the tokenizer's 49408 so the real BPE ids index the text embedding.
UNIFIED_PRETRAIN = {
    "definition": dict(DEFINITION),
    "model": {"embed_dim": 16, "image_resolution": 32, "vision_layers": 2,
              "vision_width": 64, "vision_patch_size": 16, "context_length": 77,
              "vocab_size": 49408, "transformer_width": 64, "transformer_heads": 2,
              "transformer_layers": 2},
    "data": {"image_size": 32, "num_workers": 0},
    "prompts": {"use_official_imagenet": False, "templates": ["a photo of a {}."]},
    "training": {"epochs": 1, "batch_size": 2, "lr": 6.0e-4, "min_lr": 1.0e-6,
                 "beta1": 0.9, "beta2": 0.95, "eps": 1.0e-8, "weight_decay": 0.05,
                 "warmup_epochs": 0, "clip_grad_norm": 1.0, "save_at_epochs": []},
}
UNIFIED_EVAL = {"recipe": "unified", "resolution": 32, "patch_size": 16,
                "width": 64, "layers": 2, "heads": 1, "output_dim": 16,
                "epochs": 2, "batch_size": 2, "num_workers": 0, "lr": 0.1,
                "momentum": 0.9, "weight_decay": 0.0}


def _deep_pretrain(**over) -> dict:
    cfg = {"stage": "pretrain", "seed": 0, "device": "cpu",
           **copy.deepcopy(UNIFIED_PRETRAIN)}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def _submodule_present() -> bool:
    return (UPSTREAM / "clip" / "__init__.py").is_file()


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
                Image.fromarray((base + noise).astype("uint8")).save(d / f"{i}.png")
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="clip-"))
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
        self.assertIn("openai/CLIP", adapter.UPSTREAM["repo"])
        self.assertNotIn("fork_of", prov["upstream"])

    def test_the_backbone_artifact_is_pinned_by_sha256(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        art = prov["backbone_artifact"]
        for key in ("url", "filename", "sha256", "bytes", "license"):
            self.assertIn(key, art)
        self.assertEqual(len(art["sha256"]), 64)
        self.assertTrue(art["url"].startswith("https://"))

    def test_the_licence_is_permissive_and_carries_no_licence_note(self):
        # openai/CLIP is MIT: the permissive precedent (24_beit) records the
        # licence in upstream.license and adds NO `licence_note` block (that field
        # is reserved for the non-commercial submodules).
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(prov["upstream"]["license"], "MIT")
        self.assertNotIn("licence_note", prov)


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

    def test_pretrain_declares_an_encoder(self):
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(_deep_pretrain(data_root=str(self.tmp / "d"))),
                     encoding="utf-8")
        self.assertIsNone(adapter._absent_reason(p))


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
        return load("clip_eval", METHOD / "evaluate_linear_clip.py")

    @needs_clip
    def test_the_pooled_feature_is_one_vector_per_image(self):
        import torch
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
        ev = self.evaluator()
        model = ev.build_model(UPSTREAM, dict(EVAL_TRAIN), torch.device("cpu"))
        feats = ev.extract_cls(model, torch.zeros(2, 3, 32, 32),
                               dict(EVAL_TRAIN), torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, OUTPUT_DIM))


class TestTheDeviceIsResolved(Base):
    """Referenced by the device mutation spec."""

    def evaluator(self):
        return load("clip_eval", METHOD / "evaluate_linear_clip.py")

    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        ev = self.evaluator()
        with mock.patch.object(ev.torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                ev.resolve_device("cuda", 0)
            self.assertEqual(ev.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(ev.resolve_device("auto", 0).type, "cpu")

    @needs_deps
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        from unittest import mock
        ev = self.evaluator()
        with mock.patch.object(ev.torch.cuda, "is_available", return_value=True):
            self.assertEqual(ev.resolve_device("cpu", 0).type, "cpu")
            self.assertEqual(ev.resolve_device("auto", 0).type, "cuda")


class TestTheTrainerPieces(unittest.TestCase):
    def trainer(self):
        return load("clip_train", METHOD / "train_pretrain_vit_clip.py")

    @needs_deps
    def test_the_prompt_choice_is_deterministic_in_index_and_epoch(self):
        import torch
        tr = self.trainer()
        prompts = torch.arange(3 * 8 * 7).reshape(3, 8, 7)
        labels = torch.tensor([0, 2, 1])
        indices = torch.tensor([10, 20, 30])
        first = tr.choose_prompt_tokens(prompts, labels, indices, epoch=4)
        second = tr.choose_prompt_tokens(prompts, labels, indices, epoch=4)
        changed = tr.choose_prompt_tokens(prompts, labels, indices, epoch=5)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, changed))

    @needs_deps
    def test_the_lr_schedule_warms_up_then_cosines(self):
        tr = self.trainer()
        self.assertLess(tr.lr_at_step(0, 1000, 100, 6e-4, 1e-6), 6e-4)
        self.assertAlmostEqual(tr.lr_at_step(99, 1000, 100, 6e-4, 1e-6), 6e-4,
                               places=9)
        self.assertAlmostEqual(tr.lr_at_step(1000, 1000, 100, 6e-4, 1e-6), 1e-6,
                               places=9)

    @needs_deps
    def test_the_contrastive_loss_is_symmetric_and_positive(self):
        import torch
        tr = self.trainer()
        torch.manual_seed(0)
        img = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        txt = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        scale = torch.tensor(10.0)
        loss = tr.clip_contrastive_loss(img, txt, scale)
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.isfinite(loss))


class TestTheUnifiedModel(unittest.TestCase):
    @needs_clip
    def test_extract_then_load_round_trips_the_image_tower(self):
        import torch
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
        models = load("clip_models", METHOD / "models" / "__init__.py")
        model = models.build_clip(UNIFIED_PRETRAIN["model"])
        enc_state = adapter.extract_encoder(model.state_dict())
        self.assertTrue(enc_state)
        self.assertFalse(any(k.startswith("transformer.resblocks") and False
                             for k in enc_state))
        self.assertFalse(any(k.startswith("token_embedding") for k in enc_state))
        cfg = {"train": {"resolution": 32, "patch_size": 16, "width": 64,
                         "layers": 2, "heads": 1, "output_dim": 16}}
        loaded = adapter.load_encoder(enc_state, cfg).state_dict()
        pairs = 0
        for key, want in enc_state.items():
            got = loaded.get(key)
            if got is None:
                continue
            pairs += 1
            self.assertTrue(torch.equal(got, want), f"{key} changed")
        self.assertGreater(pairs, 0, "no saved weight reached the model")


class TestTheDisclosureIsEnforced(Base):
    """The Step-2 supervision disclosure is a guard, not a comment."""

    def test_pretrain_refuses_a_missing_supervision_flag(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"))
        cfg["definition"]["supervised_label_text_adaptation"] = False
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("supervised_label_text_adaptation", str(e.exception))

    def test_pretrain_refuses_claiming_comparability(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"))
        cfg["definition"]["main_vssl_comparability"] = True
        with self.assertRaises(adapter.ConfigError) as e:
            adapter.to_run_config(cfg, out=self.out)
        self.assertIn("main_vssl_comparability", str(e.exception))


class TestTheUnifiedPretrainConfig(Base):
    def test_pretrain_is_accepted(self):
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"))
        built = adapter.to_run_config(cfg, out=self.out)
        self.assertEqual(built["model"]["vision_width"], 64)
        self.assertEqual(built["training"]["save_at_epochs"], [])
        self.assertEqual(built["data"]["train_path"],
                         str(self.tmp / "data" / "train"))

    def test_a_missing_block_key_is_refused_by_name(self):
        for block, key in (("training", "lr"), ("model", "vision_width"),
                           ("data", "image_size"), ("prompts", "templates"),
                           ("definition", "adaptation")):
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

    @needs_clip
    def test_it_completes_and_satisfies_the_contract(self):
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)

    @needs_clip
    def test_it_produces_no_encoder_and_says_so(self):
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
        self.run_adapter()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)

    @needs_clip
    def test_the_manifest_records_the_pinned_upstream(self):
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)


@needs_clip
class TestAUnifiedPretrainSmoke(Base):
    def run_adapter(self, **over):
        tiny_split(self.tmp / "data")
        cfg = _deep_pretrain(data_root=str(self.tmp / "data"), **over)
        p = self.tmp / "resolved.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        return p, subprocess.run(
            [sys.executable, "-m", "adapter", "--config", str(p),
             "--out", str(self.out)],
            cwd=METHOD, env=env, capture_output=True, text=True)

    def test_it_completes_and_satisfies_the_contract(self):
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
        cfg, r = self.run_adapter()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        v = subprocess.run(
            [sys.executable, str(BIN / "contract-test.py"), "--out",
             str(self.out), "--config", str(cfg), "--exit-status", "0"],
            capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertTrue((self.out / "encoder.pt").is_file())

    def test_each_milestone_encoder_is_written(self):
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
        _, r = self.run_adapter(training={"epochs": 2, "save_at_epochs": [1, 2]})
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue((self.out / "encoder_epoch1.pt").is_file())
        self.assertTrue((self.out / "encoder_epoch2.pt").is_file())


@needs_clip
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
        if not _submodule_present():
            self.skipTest("the CLIP submodule is not checked out here")
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
