#!/usr/bin/env python3
"""Specification for cae (Context Autoencoder; Chen et al., 2022; arXiv:2202.04200).

A **pure eval-only** port: a single `linear_eval` stage and no pretraining at all,
the frozen-backbone sibling of the download-and-probe methods (eva02 / aimv2 /
data2vec2 / 36_franca). CAE's "Step 3 as-is" comparison freezes the pretrained
vision backbone (a BEiT-architecture ViT-B/16) and fits a linear probe on its
**CLS token**, because the from-scratch context-autoencoder pretraining (IN-1k,
many GPU-days) is the excluded step. The number is a genuine SSL representation,
so it is comparable.

The checkpoint, stated plainly. The capture design-of-record logs CAE as a
deficiency (DEF-02): the checkpoint its own doc names (`hujinwen/cae-base`) is not
a valid HuggingFace identifier, the official weights are Baidu-only, so the capture
pipeline fell back to a BEiT-v2 proxy and explicitly refused to label it CAE. This
port does NOT ship a proxy. It pins a real, publicly downloadable CAE ViT-B
checkpoint that the capture missed: OpenMMLab's mmselfsup **reproduction**
(`download.openmmlab.com/.../cae_vit-base-p16_...`, Apache-2.0). It is a faithful
reproduction, not the paper authors' released weights, and provenance.json says so.

Unlike data2vec2 (transformers) or eva02/aimv2 (timm), neither timm nor transformers
carries a CAE model class, and the checkpoint is in mmselfsup format. So the model
is a **small self-contained BEiT-style ViT** in the method's own evaluate script
(no mmcv/mmpretrain in the fleet), and the checkpoint's `backbone.*` tensors load
into it directly (the mmengine bookkeeping the checkpoint pickles is read with a
tolerant unpickler, so no mmengine is needed). A real run loads the hash-pinned
download (passed as `ckpt`) into that ViT; the hermetic smoke leaves `ckpt` empty
and builds a **random tiny** ViT from the config's architecture keys, so nothing
is downloaded and the pipeline runs on a CPU.
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
METHOD = ROOT / "methods" / "cae"
BIN = ROOT / "bin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                            # noqa: E402

try:
    import torch                                       # noqa: F401
    import torchvision                                 # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(
    HAVE_DEPS, "cae needs torch and torchvision")


def load(name: str, path: Path):
    return load_from(METHOD, name, path)


adapter = load("cae_adapter", METHOD / "adapter" / "__init__.py")

# A frozen backbone small enough to probe on a CPU: a random tiny CAE ViT (ckpt
# empty) -- 2 small blocks at 32px (patch 16 -> a 2x2 token grid), embed_dim 32,
# 2 heads. The shipped config pins the OpenMMLab CAE ViT-B (embed 768, 12 blocks)
# via bin/fetch-weights.py.
EVAL_TRAIN = {"name": "cae_vit-base-p16 (OpenMMLab mmselfsup reproduction)",
              "ckpt": "", "img_size": 32, "patch_size": 16, "embed_dim": 32,
              "depth": 2, "num_heads": 2, "epochs": 2, "batch_size": 2,
              "num_workers": 0, "lr": 0.1, "momentum": 0.9, "weight_decay": 0.0}

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
        self.tmp = Path(tempfile.mkdtemp(prefix="cae-"))
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
        self.assertEqual(adapter.METHOD, "cae")

    def test_there_is_no_pinned_submodule(self):
        # The model is a self-contained ViT in the evaluate script, not a git
        # submodule, so the adapter records no UPSTREAM (and provenance records
        # no upstream).
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

    def test_provenance_is_honest_that_the_checkpoint_is_a_reproduction(self):
        # The capture logged CAE as DEF-02 (no official public checkpoint). This
        # port pins the OpenMMLab reproduction and must say so, not pass it off as
        # the authors' weights.
        blob = json.dumps(self.prov()).lower()
        self.assertIn("reproduction", blob)
        self.assertIn("openmmlab", blob)


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
        return load("cae_eval", METHOD / "evaluate_linear_cae.py")

    @needs_deps
    def test_the_feature_is_one_vector_per_image(self):
        import torch
        ev = self.evaluator()
        model = ev.build_model(dict(EVAL_TRAIN), torch.device("cpu"))
        feats = ev.extract_feature(model, torch.zeros(2, 3, 32, 32),
                                   torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))


class TestTheCheckpointLoadsFaithfully(Base):
    """The mmselfsup checkpoint's `backbone.*` tensors must load into the ViT.

    Exercised without the 1.1 GB download: a tiny CAE ViT's own weights are
    written in the checkpoint's shape (nested under `state_dict`, every key
    prefixed `backbone.`) and read back. Referenced by the checkpoint mutation.
    """

    def evaluator(self):
        return load("cae_eval", METHOD / "evaluate_linear_cae.py")

    def _fake_ckpt(self, ev, drop: "str | None" = None) -> Path:
        import torch
        ref = ev.build_model(dict(EVAL_TRAIN, ckpt=""), torch.device("cpu"))
        sd = {f"backbone.{k}": v for k, v in ref.state_dict().items()}
        if drop is not None:
            key = next(k for k in sd if k.endswith(drop))
            del sd[key]
        path = self.tmp / "fake.pth"
        torch.save({"state_dict": sd}, path)
        return path

    @needs_deps
    def test_a_well_formed_checkpoint_loads_and_probes(self):
        import torch
        ev = self.evaluator()
        ckpt = self._fake_ckpt(ev)
        model = ev.build_model(dict(EVAL_TRAIN, ckpt=str(ckpt)),
                               torch.device("cpu"))
        feats = ev.extract_feature(model, torch.zeros(2, 3, 32, 32),
                                   torch.device("cpu"))
        self.assertEqual(tuple(feats.shape), (2, EMBED_DIM))
        self.assertTrue(torch.isfinite(feats).all())

    @needs_deps
    def test_a_checkpoint_missing_a_backbone_weight_is_refused(self):
        import torch
        ev = self.evaluator()
        ckpt = self._fake_ckpt(ev, drop="cls_token")
        with self.assertRaises(RuntimeError):
            ev.build_model(dict(EVAL_TRAIN, ckpt=str(ckpt)),
                           torch.device("cpu"))


class TestTheDeviceIsResolved(Base):
    """Referenced by the device mutation spec."""

    def evaluator(self):
        return load("cae_eval", METHOD / "evaluate_linear_cae.py")

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
