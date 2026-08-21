#!/usr/bin/env python3
"""Specification for the downstream ADE20K segmentation pilot (docs/DOWNSTREAM.md).

The first cross-method downstream task: freeze a backbone, fit a 1x1-conv readout,
report mIoU. It exercises the reusable pieces the rest of the battery (COCO,
NYUv2, SSv2) will share — the frozen spatial-backbone interface
(`forward_features(x) -> [B,C,h,w]` + `out_channels`), the downstream contract
(`run_manifest.json` + `metrics.json`, checked by `downstream.contract.verify`),
and the hermetic smoke (a random tiny ViT + synthetic data, downloading and
training-a-backbone nothing).
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
# contract.py is standard-library only, so it always imports. ade20k.py defines
# torch nn.Module subclasses, so it imports only where torch is present -- the
# torch-dependent tests below are skipped otherwise (and the module stays
# importable in the torch-free base env and the without-git suite scan).
from downstream import contract                                    # noqa: E402

try:
    import numpy                                          # noqa: F401
    import torch                                          # noqa: F401
    import torchvision                                    # noqa: F401
    from PIL import Image                                 # noqa: F401
    from downstream import ade20k                          # noqa: E402
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

try:
    import timm                                           # noqa: F401
    HAVE_TIMM = HAVE_DEPS
except ImportError:
    HAVE_TIMM = False

needs_deps = unittest.skipUnless(HAVE_DEPS, "downstream ADE20K needs torch/torchvision/PIL")
needs_timm = unittest.skipUnless(HAVE_TIMM, "downstream ADE20K ViT backbone needs timm")

SMOKE_BACKBONE = {"kind": "vit", "encoder": "", "arch": "vit_base_patch16_224",
                  "img_size": 32, "patch_size": 16, "embed_dim": 64, "depth": 2,
                  "num_heads": 2}
SMOKE_PROBE = {"epochs": 1, "batch_size": 2, "lr": 0.001, "num_workers": 0,
               "image_size": 32, "max_train_samples": 4, "max_val_samples": 4,
               "max_steps_per_epoch": 0}


def smoke_config(data_root: Path, **over) -> dict:
    cfg = {"task": "ade20k_segmentation", "seed": 0, "device": "cpu",
           "data_root": str(data_root), "backbone": dict(SMOKE_BACKBONE),
           "probe": dict(SMOKE_PROBE)}
    for k, v in over.items():
        cfg[k] = v
    return cfg


def tiny_ade(root: Path, per: int = 4) -> Path:
    """A tiny ADE20K-shaped tree: images/*.jpg + annotations/*.png, 3 label ids."""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    for split in ("training", "validation"):
        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per):
            arr = rng.randint(0, 256, (40, 40, 3), dtype="uint8")
            Image.fromarray(arr, "RGB").save(img_dir / f"img{i}.jpg")
            mask = rng.randint(0, 3, (40, 40), dtype="uint8")   # ids 0,1,2
            Image.fromarray(mask, "L").save(ann_dir / f"img{i}.png")
    return root


class TestTheDownstreamContract(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp(prefix="ds-"))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def test_the_vocabulary_has_both_kinds_and_the_task_metric_is_comparable(self):
        kinds = set(contract.DOWNSTREAM_METRICS.values())
        self.assertEqual(kinds, {contract.COMPARABLE, contract.PER_TASK})
        self.assertEqual(contract.DOWNSTREAM_METRICS["ade20k_miou"],
                         contract.COMPARABLE)

    def test_write_metrics_refuses_a_name_outside_the_vocabulary(self):
        with self.assertRaises(ValueError) as e:
            contract.write_metrics(self.out, {"x": 1.0}, {"x": "made_up_metric"})
        self.assertIn("made_up_metric", str(e.exception))

    def test_verify_passes_a_good_run_and_fails_an_unlisted_file(self):
        cfg = self.out / "config.json"
        cfg.write_text(json.dumps({"seed": 0}), encoding="utf-8")
        contract.write_metrics(self.out, {"miou": 1.0}, {"miou": "ade20k_miou"})
        contract.write_manifest(
            self.out, task="ade20k_segmentation", method_ref="random-smoke",
            status="ok", config_sha256=contract.sha256_bytes(cfg.read_bytes()),
            started_at="t0", finished_at="t1", seed=0, backbone={"kind": "vit"})
        ok, violations = contract.verify(self.out, cfg, exit_status=0)
        self.assertTrue(ok, violations)
        # An output the manifest never listed is a reproducibility hole.
        (self.out / "stray.txt").write_text("x", encoding="utf-8")
        ok2, violations2 = contract.verify(self.out, cfg, exit_status=0)
        self.assertFalse(ok2)
        self.assertTrue(any("unlisted" in m for m in violations2), violations2)

    def test_verify_fails_when_the_exit_status_is_nonzero(self):
        cfg = self.out / "config.json"
        cfg.write_text(json.dumps({"seed": 0}), encoding="utf-8")
        contract.write_metrics(self.out, {"miou": 1.0}, {"miou": "ade20k_miou"})
        contract.write_manifest(
            self.out, task="ade20k_segmentation", method_ref="r", status="ok",
            config_sha256=contract.sha256_bytes(cfg.read_bytes()),
            started_at="t0", finished_at="t1", seed=0, backbone={})
        ok, violations = contract.verify(self.out, cfg, exit_status=1)
        self.assertFalse(ok)


@needs_deps
class TestConfigValidation(unittest.TestCase):
    def base(self, **over) -> dict:
        return smoke_config(Path("/tmp/nowhere"), **over)

    def test_a_valid_config_is_accepted(self):
        ade20k.validate_config(self.base())

    def test_an_unknown_top_key_is_refused(self):
        cfg = self.base()
        cfg["mystery"] = 1
        with self.assertRaises(ade20k.ConfigError) as e:
            ade20k.validate_config(cfg)
        self.assertIn("mystery", str(e.exception))

    def test_setting_the_output_is_refused(self):
        cfg = self.base()
        cfg["output"] = "/anywhere"
        with self.assertRaises(ade20k.ConfigError) as e:
            ade20k.validate_config(cfg)
        self.assertIn("--out", str(e.exception))

    def test_a_bad_device_is_refused(self):
        with self.assertRaises(ade20k.ConfigError):
            ade20k.validate_config(self.base(device="tpu"))

    def test_an_unported_backbone_kind_is_refused(self):
        cfg = self.base()
        cfg["backbone"] = {**SMOKE_BACKBONE, "kind": "resnet50"}
        with self.assertRaises(ade20k.ConfigError) as e:
            ade20k.validate_config(cfg)
        self.assertIn("resnet50", str(e.exception))

    def test_a_missing_probe_key_is_refused_by_name(self):
        cfg = self.base()
        cfg["probe"] = {k: v for k, v in SMOKE_PROBE.items() if k != "epochs"}
        with self.assertRaises(ade20k.ConfigError) as e:
            ade20k.validate_config(cfg)
        self.assertIn("epochs", str(e.exception))


class TestTheDeviceIsResolved(unittest.TestCase):
    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                ade20k.resolve_device("cuda")
            self.assertEqual(ade20k.resolve_device("cpu").type, "cpu")
            self.assertEqual(ade20k.resolve_device("auto").type, "cpu")

    @needs_deps
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        from unittest import mock
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(ade20k.resolve_device("cpu").type, "cpu")


class TestTheSpatialBackbone(unittest.TestCase):
    @needs_timm
    def test_the_frozen_vit_yields_a_spatial_map(self):
        from downstream.spatial_backbones import build_frozen_backbone
        b = build_frozen_backbone(dict(SMOKE_BACKBONE), torch.device("cpu"))
        self.assertEqual(b.out_channels, 64)
        feat = b.forward_features(torch.zeros(2, 3, 32, 32))
        # 32px input at patch 16 -> a 2x2 token grid, embed_dim 64 channels.
        self.assertEqual(tuple(feat.shape), (2, 64, 2, 2))
        self.assertFalse(any(p.requires_grad for p in b.parameters()))

    @needs_timm
    def test_an_unported_kind_is_a_clear_not_implemented(self):
        from downstream.spatial_backbones import build_frozen_backbone
        with self.assertRaises(NotImplementedError):
            build_frozen_backbone({"kind": "clip_vit"}, torch.device("cpu"))

    @needs_timm
    def test_the_seg_model_outputs_per_pixel_class_logits(self):
        from downstream.spatial_backbones import build_frozen_backbone
        b = build_frozen_backbone(dict(SMOKE_BACKBONE), torch.device("cpu"))
        model = ade20k.FrozenSegModel(b)
        logits = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (2, ade20k.NUM_CLASSES, 32, 32))


class TestASmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ds-ade-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def run_entrypoint(self, **over):
        tiny_ade(self.tmp / "data")
        cfg = smoke_config(self.tmp / "data", **over)
        cfg_path = self.tmp / "resolved.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "downstream.ade20k", "--config", str(cfg_path),
             "--out", str(self.out)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        return cfg_path, r

    @needs_timm
    def test_it_completes_and_satisfies_the_downstream_contract(self):
        cfg_path, r = self.run_entrypoint()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        ok, violations = contract.verify(self.out, cfg_path, r.returncode)
        self.assertTrue(ok, violations)
        man = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(man["status"], "ok", man.get("error", ""))
        metrics = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("ade20k_miou", metrics)
        self.assertIn("ade20k_pixel_accuracy", metrics)

    @needs_timm
    def test_a_subset_run_is_not_recordable(self):
        # The smoke subsets the data, so its number must be stamped non-recordable.
        _, r = self.run_entrypoint()
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        results = json.loads((self.out / "results.json").read_text())
        self.assertFalse(results["record_value"])
        self.assertTrue(results["subset_or_smoke"])

    @needs_timm
    def test_a_refused_config_exits_two_and_writes_no_manifest(self):
        _, r = self.run_entrypoint(device="tpu")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertFalse((self.out / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
