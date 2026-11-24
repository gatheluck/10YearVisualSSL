"""Exercise Figure 5 native semantic evaluation through the public CLI."""

import json
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
import zipfile
from io import BytesIO

try:
    import h5py
    import numpy as np
    from PIL import Image
    from scipy.io import savemat
    import torch
    from downstream import extended_segmentation as cli
except ImportError:
    cli = None


@unittest.skipIf(
    cli is None or find_spec("pycocotools") is None, "requires downstream dependencies"
)
class NativeSemanticEvaluation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.predictions = self.root / "predictions"
        self.predictions.mkdir()
        self.cfg = {
            "task": "figure5_native_semantic_evaluation",
            "device": "cpu",
            "population": "fixture",
            "predictions": str(self.predictions),
        }

    def score(self, name="out", *, status=0):
        config = self.root / "config.json"
        config.write_text(json.dumps(self.cfg))
        output = self.root / name
        self.assertEqual(
            cli.main(["--config", str(config), "--out", str(output)]), status
        )
        if status:
            self.assertFalse((output / "results.json").exists())
            return
        return json.loads((output / "results.json").read_text())

    def spacenet(self):
        self.cfg.update(dataset="spacenet2_vegas", data_root=str(self.root))
        images = self.root / "RGB-PanSharpen_jpg"
        images.mkdir()
        ann = self.root / "annotations"
        ann.mkdir()
        for split, identifier in [("train", 15), ("val", 27)]:
            filename = f"{split}.jpg"
            Image.new("RGB", (224, 224), "red").save(images / filename)
            polygon = [0, 0, 112, 0, 112, 224, 0, 224]
            data = {
                "categories": [{"id": 1, "name": "building"}],
                "images": [
                    {
                        "id": identifier,
                        "width": 224,
                        "height": 224,
                        "file_name": filename,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": identifier,
                        "category_id": 1,
                        "bbox": [0, 0, 112, 224],
                        "segmentation": [polygon],
                        "area": 25088,
                        "iscrowd": 0,
                    }
                ],
            }
            (ann / f"instance_{split}.json").write_text(json.dumps(data))
        np.save(self.predictions / "a.npy", np.zeros((224, 224), dtype=np.uint8))
        (self.predictions / "index.json").write_text(json.dumps({"27": "a.npy"}))

    def sun(self):
        self.cfg.update(
            dataset="sun_rgbd",
            archive=str(self.root / "images.zip"),
            toolbox=str(self.root / "toolbox.zip"),
            semantic_mat=str(self.root / "labels.mat"),
        )

        def mat(data):
            buffer = BytesIO()
            savemat(buffer, data)
            return buffer.getvalue()

        with zipfile.ZipFile(self.cfg["archive"], "w") as archive:
            metadata = np.empty(2, dtype=[("rgbpath", "O"), ("depthpath", "O")])
            for i, name in enumerate(("test", "train")):
                base = "SUNRGBD/" + name
                metadata[i] = (base + "/image/color.jpg", base + "/depth/depth.png")
                for member, image in [
                    (metadata[i][0], Image.new("RGB", (3, 2), "red")),
                    (metadata[i][1], Image.fromarray(np.full((2, 3), 8000, np.uint16))),
                ]:
                    buffer = BytesIO()
                    image.save(buffer, "PNG")
                    archive.writestr(member, buffer.getvalue())
        with zipfile.ZipFile(self.cfg["toolbox"], "w") as archive:
            archive.writestr(
                "SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat",
                mat({"SUNRGBDMeta": metadata}),
            )
            archive.writestr(
                "SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat",
                mat(
                    {
                        "alltrain": np.array(["SUNRGBD/train"], dtype=object),
                        "alltest": np.array(["SUNRGBD/test"], dtype=object),
                    }
                ),
            )
        gt = np.array([[1, 1, 0], [37, 37, 37]], np.uint8)
        with h5py.File(self.cfg["semantic_mat"], "w") as f:
            cells = f.create_dataset(
                "SUNRGBD2Dseg/seglabel", (1, 2), dtype=h5py.ref_dtype
            )
            for i, labels in enumerate((gt, np.ones_like(gt))):
                cells[0, i] = f.create_dataset("map" + str(i), data=labels.T).ref
        np.save(self.predictions / "a.npy", np.zeros(gt.shape, np.uint8))
        (self.predictions / "index.json").write_text(
            json.dumps({"SUNRGBD/test": "a.npy"})
        )

    def test_spacenet_building_and_background_are_equally_weighted(self):
        self.spacenet()
        report = self.score()
        self.assertEqual(report["metrics"]["miou"], 25.0)
        self.assertEqual(report["metrics"]["pixels"], 224**2)
        self.assertFalse(report["paper_score_verified"])

    def test_sun_uses_official_metadata_order_native_dimensions_and_void(self):
        self.sun()
        report = self.score()
        self.assertEqual(report["metrics"]["pixels"], 5)
        self.assertAlmostEqual(report["metrics"]["miou"], 20.0)
        self.assertFalse(report["paper_score_verified"])

    def test_sun_model_inputs_encode_metric_depth_and_validity(self):
        self.sun()
        self.score()
        from downstream.figure5_native_semantics import native_samples

        with native_samples(self.cfg, train=False) as samples:
            self.assertEqual(list(samples.ids), ["SUNRGBD/test"])
            image, mask = samples[0]
        self.assertEqual(tuple(image.shape), (5, 224, 224))
        torch.testing.assert_close(image[3], torch.full_like(image[3], 0.1))
        self.assertTrue(image[4].eq(1).all())
        self.assertEqual(mask.tolist(), [[0, 0, 255], [36, 36, 36]])
        with native_samples(self.cfg, train=True) as samples:
            self.assertEqual(list(samples.ids), ["SUNRGBD/train"])
            self.assertEqual(samples[0][1].shape, (224, 224))

    def test_spacenet_inputs_preserve_rgb_range_and_union_foreground(self):
        self.spacenet()
        self.score()
        from downstream.figure5_native_semantics import native_samples

        with native_samples(self.cfg, train=False) as samples:
            image, target = samples[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertGreater(image[0].min().item(), 0.98)
        self.assertLess(image[1:].max().item(), 0.02)
        self.assertTrue(target[:, :112].eq(1).all())
        self.assertTrue(target[:, 112:].eq(0).all())

    def test_incomplete_full_population_and_misaligned_maps_fail_closed(self):
        self.sun()
        self.cfg["population"] = "full"
        self.score("short", status=1)
        self.cfg["population"] = "fixture"
        np.save(self.predictions / "a.npy", np.zeros((224, 224), np.uint8))
        self.score("resized", status=1)
        (self.predictions / "index.json").write_text('{"SUNRGBD/train":"a.npy"}')
        self.score("wrong", status=1)


class NativeCI(unittest.TestCase):
    def test_downstream_environment_exercises_native_formats(self):
        from tests._checkout import is_checkout
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if not is_checkout() or not HAVE_YAML:
            self.skipTest("checkout and workflow parser required")
        steps = parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
        command = next(
            step["run"]
            for step in steps
            if step.get("name")
            == "Run Basic5 component contracts with downstream dependencies"
        )
        self.assertTrue(
            _runs_finetune_tests(command, module="tests.test_figure5_native_semantics")
        )
