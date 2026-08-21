#!/usr/bin/env python3
"""Specification for the downstream COCO detection task (docs/DOWNSTREAM.md).

The second cross-method downstream task, reusing the pilot's pieces: the frozen
spatial-backbone interface (`forward_features`/`forward -> [B,C,h,w]` +
`out_channels`) and the downstream contract (`downstream.contract`). It attaches a
torchvision Faster R-CNN head on the frozen backbone's single feature map and
reports COCO bbox mAP via `pycocotools`. The hermetic smoke builds a random tiny
ViT and a synthetic 2-image COCO, so it downloads nothing and trains no backbone.
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
    import numpy                                          # noqa: F401
    import torch                                          # noqa: F401
    import torchvision                                    # noqa: F401
    from PIL import Image                                 # noqa: F401
    from downstream import coco                            # noqa: E402
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

try:
    import timm                                           # noqa: F401
    import pycocotools                                    # noqa: F401
    HAVE_COCO = HAVE_DEPS
except ImportError:
    HAVE_COCO = False

needs_deps = unittest.skipUnless(HAVE_DEPS, "downstream COCO needs torch/torchvision/PIL")
needs_coco = unittest.skipUnless(
    HAVE_COCO, "downstream COCO needs torch/torchvision/timm/pycocotools")

SMOKE_BACKBONE = {"kind": "vit", "encoder": "", "arch": "vit_base_patch16_224",
                  "img_size": 64, "patch_size": 16, "embed_dim": 64, "depth": 2,
                  "num_heads": 2}
SMOKE_DETECTOR = {"epochs": 1, "batch_size": 2, "lr": 0.005, "num_workers": 0,
                  "min_size": 64, "max_size": 64, "anchor_sizes": [8, 16, 32],
                  "max_train_samples": 2, "max_val_samples": 2,
                  "max_steps_per_epoch": 0}


def smoke_config(data_root: Path, **over) -> dict:
    cfg = {"task": "coco_detection", "seed": 0, "device": "cpu",
           "data_root": str(data_root), "backbone": dict(SMOKE_BACKBONE),
           "detector": dict(SMOKE_DETECTOR)}
    for k, v in over.items():
        cfg[k] = v
    return cfg


def tiny_coco(root: Path, per: int = 2) -> Path:
    """A minimal COCO tree: 64x64 images + an instances json with one box each."""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(0)
    categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    for split in ("train2017", "val2017"):
        img_dir = root / "images" / split
        ann_dir = root / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for i in range(per):
            name = f"img{i}.jpg"
            Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype="uint8"),
                            "RGB").save(img_dir / name)
            images.append({"id": i + 1, "file_name": name,
                           "width": 64, "height": 64})
            annotations.append({"id": i + 1, "image_id": i + 1,
                                "category_id": (i % 2) + 1,
                                "bbox": [8, 8, 24, 24], "area": 576,
                                "iscrowd": 0})
        (ann_dir / f"instances_{split}.json").write_text(
            json.dumps({"images": images, "annotations": annotations,
                        "categories": categories}), encoding="utf-8")
    return root


class TestConfigValidationExists(unittest.TestCase):
    def test_the_task_metrics_are_in_the_downstream_vocabulary(self):
        for name in ("coco_map", "coco_map_50"):
            self.assertIn(name, contract.DOWNSTREAM_METRICS)
            self.assertEqual(contract.DOWNSTREAM_METRICS[name], contract.COMPARABLE)


@needs_deps
class TestConfigValidation(unittest.TestCase):
    def base(self, **over) -> dict:
        return smoke_config(Path("/tmp/nowhere"), **over)

    def test_a_valid_config_is_accepted(self):
        coco.validate_config(self.base())

    def test_an_unknown_top_key_is_refused(self):
        cfg = self.base()
        cfg["mystery"] = 1
        with self.assertRaises(coco.ConfigError) as e:
            coco.validate_config(cfg)
        self.assertIn("mystery", str(e.exception))

    def test_setting_the_output_is_refused(self):
        cfg = self.base()
        cfg["output"] = "/anywhere"
        with self.assertRaises(coco.ConfigError) as e:
            coco.validate_config(cfg)
        self.assertIn("--out", str(e.exception))

    def test_a_bad_device_is_refused(self):
        with self.assertRaises(coco.ConfigError):
            coco.validate_config(self.base(device="tpu"))

    def test_an_unported_backbone_kind_is_refused(self):
        cfg = self.base()
        cfg["backbone"] = {**SMOKE_BACKBONE, "kind": "resnet50"}
        with self.assertRaises(coco.ConfigError) as e:
            coco.validate_config(cfg)
        self.assertIn("resnet50", str(e.exception))

    def test_a_missing_detector_key_is_refused_by_name(self):
        cfg = self.base()
        cfg["detector"] = {k: v for k, v in SMOKE_DETECTOR.items() if k != "min_size"}
        with self.assertRaises(coco.ConfigError) as e:
            coco.validate_config(cfg)
        self.assertIn("min_size", str(e.exception))


class TestTheDeviceIsResolved(unittest.TestCase):
    @needs_deps
    def test_asking_for_cuda_without_one_is_refused(self):
        from unittest import mock
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                coco.resolve_device("cuda")
            self.assertEqual(coco.resolve_device("cpu").type, "cpu")
            self.assertEqual(coco.resolve_device("auto").type, "cpu")

    @needs_deps
    def test_cpu_is_honoured_even_where_cuda_exists(self):
        from unittest import mock
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(coco.resolve_device("cpu").type, "cpu")


class TestTheDetector(unittest.TestCase):
    @needs_coco
    def test_the_frozen_detector_builds_and_keeps_the_backbone_frozen(self):
        model = coco.build_frozen_detector(dict(SMOKE_BACKBONE),
                                           dict(SMOKE_DETECTOR),
                                           torch.device("cpu"))
        self.assertFalse(any(p.requires_grad
                             for p in model.backbone.parameters()))
        model.eval()
        with torch.no_grad():
            out = model([torch.rand(3, 64, 64)])
        self.assertEqual(set(out[0]), {"boxes", "labels", "scores"})


class TestASmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ds-coco-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"

    def run_entrypoint(self, **over):
        tiny_coco(self.tmp / "data")
        cfg = smoke_config(self.tmp / "data", **over)
        cfg_path = self.tmp / "resolved.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run(
            [sys.executable, "-m", "downstream.coco", "--config", str(cfg_path),
             "--out", str(self.out)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        return cfg_path, r

    @needs_coco
    def test_it_completes_and_satisfies_the_downstream_contract(self):
        cfg_path, r = self.run_entrypoint()
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        ok, violations = contract.verify(self.out, cfg_path, r.returncode)
        self.assertTrue(ok, violations)
        metrics = json.loads((self.out / "metrics.json").read_text())["metrics"]
        self.assertIn("coco_map", metrics)
        self.assertIn("coco_map_50", metrics)

    @needs_coco
    def test_a_subset_run_is_not_recordable(self):
        _, r = self.run_entrypoint()
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        results = json.loads((self.out / "results.json").read_text())
        self.assertFalse(results["record_value"])
        self.assertTrue(results["subset_or_smoke"])

    @needs_coco
    def test_a_refused_config_exits_two_and_writes_no_manifest(self):
        _, r = self.run_entrypoint(device="tpu")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertFalse((self.out / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
