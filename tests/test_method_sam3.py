#!/usr/bin/env python3
"""Specification for sam3 (Meta SAM 3; facebook/sam3; ai.meta.com/research/sam3).

A **pure eval-only** port, the `transformers`-sourced sibling of `data2vec2`: a
single `linear_eval` stage and no pretraining at all. SAM 3 is a promptable
segmentation foundation model; the capture's Step-3 CompEval evaluates it as an
as-is frozen backbone -- its vision encoder's patch tokens, mean-pooled, probed by
a linear classifier (no text/box prompts). The from-scratch pretraining is the
excluded step, so the port reuses the released checkpoint. The representation is a
genuine learned feature, so the number is comparable (the multimodal
"pretrained-backbone reuse" row).

So this port ships no `encoder.pt` from training; `linear_eval` probes a frozen
backbone built from `transformers.Sam3ViTModel`. There is **no author submodule**:
the model class is `transformers`' (a pinned pip dependency, `transformers==5.16.1`),
and the weights are a sha256-pinned download recorded as `backbone_artifact` in
provenance.json. Because the official `sam3.pt` uses ViTDet-style trunk keys
(fused qkv, a CLS in `pos_embed`) that do not match `Sam3ViTModel`, a real run
converts them (`sam3_trunk.load_official_trunk`); the hermetic smoke leaves `ckpt`
empty and builds a **random tiny** `Sam3ViTModel`, so nothing is downloaded.
Licence: the SAM 3 weights are Meta's, gated; nothing is copied here.
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
METHOD = ROOT / "methods" / "sam3"
BIN = ROOT / "bin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import transformers                                # noqa: F401
    from transformers import Sam3ViTModel              # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "sam3 needs torch, torchvision and transformers (with Sam3ViTModel)")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("sam3_adapter", METHOD / "adapter" / "__init__.py")

# A frozen backbone small enough to probe on a CPU: a random tiny SAM3 ViT
# (ckpt empty) -- 2 small blocks at 28px (patch 14 -> a 2x2 token grid), embed_dim
# 32, 2 heads. The shipped config pins the official ViT-L SAM3 (embed 1024, depth
# 32) via bin/fetch-weights.py.
EVAL_TRAIN = {"name": "facebook/sam3", "ckpt": "", "img_size": 28,
              "patch_size": 14, "embed_dim": 32, "depth": 2, "num_heads": 2,
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
        self.tmp = Path(tempfile.mkdtemp(prefix="sam3-"))
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
        self.assertEqual(adapter.METHOD, "sam3")

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
        return load("sam3_eval", METHOD / "evaluate_linear_sam3.py")

    @needs_deps
    def test_the_feature_is_one_vector_per_image(self):
        import torch
        ev = self.evaluator()
        model = ev.build_model(dict(EVAL_TRAIN), torch.device("cpu"))
        feats = ev.extract_feature(model, torch.zeros(2, 3, 28, 28),
                                   torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))


class TestTheTrunkConverter(Base):
    """The real-run path: the official ViTDet-style trunk (fused qkv, a CLS in
    pos_embed) is converted onto Sam3ViTModel's split-projection, no-CLS layout so
    the backbone is not left randomly initialised."""

    def trunk(self):
        return load("sam3_trunk", METHOD / "sam3_trunk.py")

    def _official_trunk(self, H=32, depth=2, mlp=64, grid=2):
        """A synthetic official-format trunk state dict (fused qkv, CLS in
        pos_embed, a patch-embed bias) with a leading detector.* prefix."""
        import torch
        g = torch.Generator().manual_seed(0)
        n = grid * grid
        sd = {
            "patch_embed.proj.weight": torch.randn(H, 3, 14, 14, generator=g),
            "patch_embed.proj.bias": torch.zeros(H),
            "pos_embed": torch.randn(1, 1 + n, H, generator=g),  # CLS + patches
            "ln_pre.weight": torch.randn(H, generator=g),
            "ln_pre.bias": torch.randn(H, generator=g),
        }
        for i in range(depth):
            p = f"blocks.{i}."
            for nm, shape in (("norm1.weight", (H,)), ("norm1.bias", (H,)),
                              ("norm2.weight", (H,)), ("norm2.bias", (H,)),
                              ("attn.qkv.weight", (3 * H, H)),
                              ("attn.qkv.bias", (3 * H,)),
                              ("attn.proj.weight", (H, H)),
                              ("attn.proj.bias", (H,)),
                              ("mlp.fc1.weight", (mlp, H)), ("mlp.fc1.bias", (mlp,)),
                              ("mlp.fc2.weight", (H, mlp)), ("mlp.fc2.bias", (H,))):
                sd[p + nm] = torch.randn(*shape, generator=g)
        return {"detector.backbone.vision_backbone.trunk." + k: v
                for k, v in sd.items()}

    @needs_deps
    def test_the_official_trunk_loads_with_no_backbone_weight_missing(self):
        import torch
        from transformers import Sam3ViTConfig, Sam3ViTModel
        tk = self.trunk()
        raw = self._official_trunk()
        converted = tk.convert_official_trunk_to_hf(tk._strip_prefix(raw))
        converted.pop("_unused_patch_embed_bias", None)
        cfg = Sam3ViTConfig(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, patch_size=14, image_size=28,
            pretrain_image_size=28, window_size=24,
            global_attn_indexes=[7, 15, 23, 31])
        model = Sam3ViTModel(cfg)
        result = model.load_state_dict(converted, strict=False)
        missing = [k for k in result.missing_keys if "rope" not in k]
        self.assertEqual(missing, [], f"backbone weights left missing: {missing}")
        self.assertEqual(list(result.unexpected_keys), [])
        model.eval()
        with torch.no_grad():
            out = model(pixel_values=torch.zeros(1, 3, 28, 28)).last_hidden_state
        self.assertTrue(bool(torch.isfinite(out).all()))

    @needs_deps
    def test_a_checkpoint_with_no_trunk_keys_is_refused(self):
        tk = self.trunk()
        with self.assertRaises(RuntimeError):
            tk._strip_prefix({"detector.something.else": 1, "head.weight": 2})


class TestTheDeviceIsResolved(Base):
    """Referenced by the device mutation spec."""

    def evaluator(self):
        return load("sam3_eval", METHOD / "evaluate_linear_sam3.py")

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


if __name__ == "__main__":
    unittest.main()
