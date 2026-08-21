#!/usr/bin/env python3
"""Specification for the downstream NYUv2 depth task (docs/DOWNSTREAM.md).

The third cross-method downstream task, reusing the pilot's pieces: the frozen
spatial-backbone interface and the downstream contract. It attaches a DPT-style
progressive-upsampling depth head on the frozen backbone's feature map and reports
RMSE / AbsRel. The hermetic smoke builds a random tiny ViT and a synthetic HDF5
`.mat`, so it downloads nothing and trains no backbone.
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from downstream import contract                                    # noqa: E402

try:
    import numpy as np                                    # noqa: F401
    import torch                                          # noqa: F401
    import torchvision                                    # noqa: F401
    import h5py                                           # noqa: F401
    from downstream import nyuv2                           # noqa: E402
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

try:
    import timm                                           # noqa: F401
    HAVE_NYUV2 = HAVE_DEPS
except ImportError:
    HAVE_NYUV2 = False

needs_deps = unittest.skipUnless(HAVE_DEPS, "downstream NYUv2 needs torch/torchvision/h5py")
needs_nyuv2 = unittest.skipUnless(
    HAVE_NYUV2, "downstream NYUv2 needs torch/torchvision/timm/h5py")

SMOKE_BACKBONE = {"kind": "vit", "encoder": "", "arch": "vit_base_patch16_224",
                  "img_size": 64, "patch_size": 16, "embed_dim": 64, "depth": 2,
                  "num_heads": 2}
SMOKE_PROBE = {"epochs": 1, "batch_size": 2, "lr": 0.001, "num_workers": 0,
               "image_size": 64, "head_hidden_dim": 32, "max_train_samples": 2,
               "max_val_samples": 2, "max_steps_per_epoch": 0}


def smoke_config(data_root: Path, **over) -> dict:
    cfg = {"task": "nyuv2_depth", "seed": 0, "device": "cpu",
           "data_root": str(data_root), "backbone": dict(SMOKE_BACKBONE),
           "probe": dict(SMOKE_PROBE)}
    for k, v in over.items():
        cfg[k] = v
    return cfg


def tiny_nyuv2(data_root: Path, n: int = 6) -> Path:
    """A minimal labelled NYUv2 HDF5: images (n,3,48,48) + depths (n,48,48)."""
    import h5py
    import numpy as np
    rng = np.random.RandomState(0)
    mat = data_root / "labeled" / "nyu_depth_v2_labeled.mat"
    mat.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(mat), "w") as handle:
        handle.create_dataset(
            "images", data=rng.randint(0, 256, (n, 3, 48, 48)).astype("float32"))
        # positive finite depths (metres), a few zeros so the valid mask matters
        depths = rng.uniform(0.5, 5.0, (n, 48, 48)).astype("float32")
        depths[:, :4, :4] = 0.0
        handle.create_dataset("depths", data=depths)
    return data_root


class TestTheTaskMetricsAreRegistered(unittest.TestCase):
    def test_rmse_and_absrel_are_in_the_downstream_vocabulary(self):
        for name in ("nyuv2_rmse", "nyuv2_absrel"):
            self.assertIn(name, contract.DOWNSTREAM_METRICS)
            self.assertEqual(contract.DOWNSTREAM_METRICS[name], contract.COMPARABLE)


@needs_deps
class TestConfigValidation(unittest.TestCase):
    def base(self, **over) -> dict:
        return smoke_config(Path("/tmp/nowhere"), **over)

    def test_a_valid_config_is_accepted(self):
        nyuv2.validate_config(self.base())

    def test_an_unknown_top_key_is_refused(self):
        cfg = self.base()
        cfg["mystery"] = 1
        with self.assertRaises(nyuv2.ConfigError) as e:
            nyuv2.validate_config(cfg)
        self.assertIn("mystery", str(e.exception))

    def test_setting_the_output_is_refused(self):
        cfg = self.base()
        cfg["output"] = "/anywhere"
        with self.assertRaises(nyuv2.ConfigError) as e:
            nyuv2.validate_config(cfg)
        self.assertIn("--out", str(e.exception))

    def test_a_bad_device_is_refused(self):
        with self.assertRaises(nyuv2.ConfigError):
            nyuv2.validate_config(self.base(device="tpu"))

    def test_an_unported_backbone_kind_is_refused(self):
        cfg = self.base()
        cfg["backbone"] = {**SMOKE_BACKBONE, "kind": "resnet50"}
        with self.assertRaises(nyuv2.ConfigError) as e:
            nyuv2.validate_config(cfg)
        self.assertIn("resnet50", str(e.exception))

    def test_a_missing_probe_key_is_refused_by_name(self):
        cfg = self.base()
        cfg["probe"] = {k: v for k, v in SMOKE_PROBE.items() if k != "head_hidden_dim"}
        with self.assertRaises(nyuv2.ConfigError) as e:
            nyuv2.validate_config(cfg)
        self.assertIn("head_hidden_dim", str(e.exception))

    def test_the_split_keeps_the_capture_split_for_the_full_file(self):
        train, val = nyuv2.split_indices(nyuv2.FULL_SIZE)
        self.assertEqual(len(train), nyuv2.TRAIN_SIZE)
        self.assertEqual(len(val), nyuv2.FULL_SIZE - nyuv2.TRAIN_SIZE)

    def test_a_small_file_is_split_in_half(self):
        train, val = nyuv2.split_indices(6)
        self.assertEqual((train, val), ([0, 1, 2], [3, 4, 5]))


class TestTheDeviceIsResolved(unittest.TestCase):
    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                nyuv2.resolve_device("cuda")
            self.assertEqual(nyuv2.resolve_device("cpu").type, "cpu")
            self.assertEqual(nyuv2.resolve_device("auto").type, "cpu")

    @needs_deps
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        from unittest import mock
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(nyuv2.resolve_device("cpu").type, "cpu")


class TestTheDepthHead(unittest.TestCase):
    @needs_nyuv2
    def test_the_depth_head_predicts_one_positive_map_per_image(self):
        from downstream.spatial_backbones import build_frozen_backbone
        b = build_frozen_backbone(dict(SMOKE_BACKBONE), torch.device("cpu"))
        model = nyuv2.FrozenDepthModel(b, hidden_dim=32)
        pred = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(pred.shape), (2, 1, 64, 64))
        # softplus output is strictly positive, a depth.
        self.assertTrue(bool((pred > 0).all()))


class TestASmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ds-nyu-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def run_entrypoint(self, **over):
        tiny_nyuv2(self.tmp / "data")
        cfg = smoke_config(self.tmp / "data", **over)
        cfg_path = self.tmp / "resolved.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "downstream.nyuv2", "--config", str(cfg_path),
             "--out", str(self.out)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        return cfg_path, r

    @needs_nyuv2
    def test_it_completes_and_satisfies_the_downstream_contract(self):
        cfg_path, r = self.run_entrypoint()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        ok, violations = contract.verify(self.out, cfg_path, r.returncode)
        self.assertTrue(ok, violations)
        metrics = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("nyuv2_rmse", metrics)
        self.assertIn("nyuv2_absrel", metrics)

    @needs_nyuv2
    def test_a_subset_run_is_not_recordable(self):
        _, r = self.run_entrypoint()
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        results = json.loads((self.out / "results.json").read_text())
        self.assertFalse(results["record_value"])
        self.assertTrue(results["subset_or_smoke"])

    @needs_nyuv2
    def test_a_refused_config_exits_two_and_writes_no_manifest(self):
        _, r = self.run_entrypoint(device="tpu")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertFalse((self.out / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
