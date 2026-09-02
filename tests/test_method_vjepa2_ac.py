#!/usr/bin/env python3
"""Specification for vjepa2_ac (V-JEPA 2-AC; Assran et al., Meta FAIR, 2025;
arXiv:2506.09985 -- the action-conditioned world model).

A **pure eval-only** port: a single `linear_eval` stage and no pretraining. It is
the frozen-backbone sibling of `vjepa2`, but faithful in a way that one is not: the
ViT is the **pinned facebookresearch/vjepa2 submodule** (third_party/vjepa2),
imported not copied, and it runs V-JEPA 2's **real rotary-position attention**
(`use_rope=True`), reproducing the capture's number rather than approximating it
with plain attention. V-JEPA 2-AC's encoder is architecturally the V-JEPA 2 ViT-g
(embed_dim 1408, depth 40, 22 heads), pretrained with an action-conditioned world
model on robot-manipulation video (Droid) and evaluated here as a general visual
backbone.

The representation, on a still image. V-JEPA 2 consumes a video clip
`(B, 3, T, H, W)` via a Conv3d tubelet patch embed; the capture's linear eval feeds
a **still image** replicated `tubelet_size` times along the temporal axis (one
temporal token; never PyAV / a video dataset), runs the ViT and mean-pools the
tokens. This port keeps exactly that.

The checkpoint, stated plainly. The public `vjepa2-ac-vitg.pt` is a training-state
dict whose `encoder` sub-dict carries the ViT weights under a `module.` prefix
(plus a `predictor` action-conditioned predictor, `opt`, `scaler`, `target_encoder`
and scalars, all dropped). Its `encoder.module.*` tensors, with `module.`/`backbone.`
stripped, load into the submodule's `vit_giant_xformers` with **zero missing or
unexpected keys** (measured: 484 == 484), so any mismatch is a hard error, not a
silently half-loaded backbone. A real run loads the hash-pinned download (passed as
`ckpt`); the hermetic smoke leaves `ckpt` empty and builds a **random tiny**
`vit_tiny` (still `use_rope=True`) from the config's `arch`, so nothing is
downloaded and the pipeline runs on a CPU.

Licence: MIT (measured from the official facebookresearch/vjepa2 repository;
the capture's backbone comment labels it CC-BY-NC-4.0, an error inherited from
V-JEPA 1, and MIT is recorded with that correction noted in provenance.json).
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
METHOD = ROOT / "methods" / "vjepa2_ac"
BIN = ROOT / "bin"
UPSTREAM = ROOT / "third_party" / "vjepa2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    import numpy                                       # noqa: F401
    import einops                                      # noqa: F401
    import timm                                        # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "vjepa2_ac needs torch, torchvision, numpy, einops, timm")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("vjepa2_ac_adapter", METHOD / "adapter" / "__init__.py")

# A frozen backbone small enough to probe on a CPU: a random tiny V-JEPA 2 ViT
# (ckpt empty) built through the submodule as `vit_tiny` (embed_dim 192, the
# smallest factory) with `use_rope=True` -- the real rotary forward, at 32px
# (patch 16 -> a 2x2 spatial grid) on a 2-frame clip (tubelet 2 -> one temporal
# token). The shipped config pins vjepa2-ac-vitg.pt via bin/fetch-weights.py and
# builds `vit_giant_xformers` (embed_dim 1408, depth 40).
EVAL_TRAIN = {"name": "facebook/vjepa2-ac-vitg", "ckpt": "", "arch": "vit_tiny",
              "img_size": 32, "patch_size": 16, "num_frames": 64,
              "tubelet_size": 2, "epochs": 2, "batch_size": 2, "num_workers": 0,
              "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

EMBED_DIM = 192   # vit_tiny embed dim


def _submodule_present() -> bool:
    return (UPSTREAM / "src" / "models" / "vision_transformer.py").is_file()


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
        self.tmp = Path(tempfile.mkdtemp(prefix="vjepa2ac-"))
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
        self.assertEqual(adapter.METHOD, "vjepa2_ac")

    def test_it_pins_the_vjepa2_submodule(self):
        # Unlike the self-contained sibling vjepa2, this port builds the ViT from
        # the pinned facebookresearch/vjepa2 submodule, so it records an UPSTREAM
        # (and provenance an upstream) -- test_port_completeness's both-places rule.
        self.assertTrue(hasattr(adapter, "UPSTREAM"))
        self.assertIn("facebookresearch/vjepa2", adapter.UPSTREAM["repo"])
        self.assertEqual(len(adapter.UPSTREAM["commit"]), 40)


class TestThePinnedUpstream(unittest.TestCase):
    @needs_checkout
    def test_the_adapter_records_the_checked_out_commit(self):
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("the submodule is not checked out here")
        self.assertEqual(r.stdout.strip(), adapter.UPSTREAM["commit"])

    def test_provenance_agrees_and_records_mit_with_the_correction(self):
        prov = json.loads((METHOD / "provenance.json").read_text())
        self.assertEqual(prov["upstream"]["commit"], adapter.UPSTREAM["commit"])
        self.assertIn("facebookresearch/vjepa2", adapter.UPSTREAM["repo"])
        # The licence is MIT (measured), recorded with the capture's
        # CC-BY-NC mislabel called out rather than propagated.
        blob = json.dumps(prov).lower()
        self.assertIn("cc-by-nc", blob)      # names the capture's error
        self.assertIn("mit", blob)


class TestThePinnedBackbone(unittest.TestCase):
    def prov(self) -> dict:
        return json.loads((METHOD / "provenance.json").read_text())

    def test_the_backbone_artifact_is_pinned_by_sha256(self):
        art = self.prov()["backbone_artifact"]
        for key in ("url", "filename", "sha256"):
            self.assertIn(key, art)
        self.assertEqual(len(art["sha256"]), 64)
        self.assertTrue(art["url"].startswith("https://"))

    def test_provenance_records_the_official_ac_weights_and_licence(self):
        art = self.prov()["backbone_artifact"]
        self.assertIn("vjepa2-ac", json.dumps(self.prov()).lower())
        self.assertEqual(art["license"], "MIT")


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
        return load("vjepa2_ac_eval", METHOD / "evaluate_linear_vjepa2_ac.py")

    @needs_deps
    def test_the_feature_is_one_vector_per_image(self):
        import torch
        if not _submodule_present():
            self.skipTest("the vjepa2 submodule is not checked out here")
        ev = self.evaluator()
        model = ev.build_model(dict(EVAL_TRAIN), torch.device("cpu"))
        feats = ev.extract_feature(model, torch.zeros(2, 3, 32, 32),
                                   torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))
        self.assertTrue(torch.isfinite(feats).all())

    @needs_deps
    def test_the_forward_uses_rotary_position_embeddings(self):
        # The whole point of this port over the sibling: the real rope forward.
        import torch
        if not _submodule_present():
            self.skipTest("the vjepa2 submodule is not checked out here")
        ev = self.evaluator()
        model = ev.build_model(dict(EVAL_TRAIN), torch.device("cpu"))
        self.assertTrue(getattr(model, "use_rope", False))


class TestTheCheckpointLoadsFaithfully(Base):
    """The official AC checkpoint's `encoder` tensors must load into the ViT.

    Exercised without the 11.8 GB download: a tiny V-JEPA 2 ViT's own weights are
    written in the checkpoint's shape (a .pt training-state dict whose `encoder`
    sub-dict carries every key `module.`-prefixed, plus a decoy `predictor` that
    must be dropped) and read back. Referenced by the checkpoint mutation.
    """

    def evaluator(self):
        return load("vjepa2_ac_eval", METHOD / "evaluate_linear_vjepa2_ac.py")

    def _fake_ckpt(self, ev, drop: "str | None" = None,
                   with_predictor: bool = True) -> Path:
        import torch
        ref = ev.build_model(dict(EVAL_TRAIN, ckpt=""), torch.device("cpu"))
        enc = {f"module.{k}": v.clone().contiguous()
               for k, v in ref.state_dict().items()}
        if drop is not None:
            key = next(k for k in enc if k.endswith(drop))
            del enc[key]
        state = {"encoder": enc, "epoch": 1, "loss": 0.0}
        if with_predictor:
            # The official checkpoint carries predictor.* (the action-conditioned
            # JEPA predictor); it is training machinery and must be dropped, not
            # treated as an unexpected encoder key.
            state["predictor"] = {"module.predictor_embed.weight":
                                  torch.zeros(4, 4)}
        path = self.tmp / "vjepa2_ac.pt"
        torch.save(state, str(path))
        return path

    @needs_deps
    def test_a_well_formed_checkpoint_loads_and_probes(self):
        import torch
        if not _submodule_present():
            self.skipTest("the vjepa2 submodule is not checked out here")
        ev = self.evaluator()
        ckpt = self._fake_ckpt(ev)
        model = ev.build_model(dict(EVAL_TRAIN, ckpt=str(ckpt)),
                               torch.device("cpu"))
        feats = ev.extract_feature(model, torch.zeros(2, 3, 32, 32),
                                   torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))
        self.assertTrue(torch.isfinite(feats).all())

    @needs_deps
    def test_a_checkpoint_missing_an_encoder_weight_is_refused(self):
        import torch
        if not _submodule_present():
            self.skipTest("the vjepa2 submodule is not checked out here")
        ev = self.evaluator()
        ckpt = self._fake_ckpt(ev, drop="norm.weight")
        with self.assertRaises(RuntimeError):
            ev.build_model(dict(EVAL_TRAIN, ckpt=str(ckpt)),
                           torch.device("cpu"))


class TestTheDeviceIsResolved(Base):
    """Referenced by the device mutation spec."""

    def evaluator(self):
        return load("vjepa2_ac_eval", METHOD / "evaluate_linear_vjepa2_ac.py")

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
        if not _submodule_present():
            self.skipTest("the vjepa2 submodule is not checked out here")
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
            self.skipTest("the vjepa2 submodule is not checked out here")
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
            self.skipTest("the vjepa2 submodule is not checked out here")
        self.run_adapter()
        self.assertFalse((self.out / "encoder.pt").exists())
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["stage"], "linear_eval")
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        self.assertIn("encoder_absent_reason", man)

    @needs_deps
    def test_the_manifest_records_the_pinned_upstream(self):
        if not _submodule_present():
            self.skipTest("the vjepa2 submodule is not checked out here")
        self.run_adapter()
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["upstream"], adapter.UPSTREAM)

    @needs_deps
    def test_the_same_config_twice_gives_the_same_classifier(self):
        """Two runs of one config must agree bit for bit, compared by the
        manifest's recorded hashes over every artifact."""
        if not _submodule_present():
            self.skipTest("the vjepa2 submodule is not checked out here")
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
            self.skipTest("the vjepa2 submodule is not checked out here")
        cfg, r = self.run_adapter(device="cuda")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("cuda", r.stdout.lower())


if __name__ == "__main__":
    unittest.main()
