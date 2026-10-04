"""PASCAL Context ontology, native lists and portable staging contracts."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAVE = all(
    importlib.util.find_spec(n)
    for n in ("scipy", "numpy", "PIL", "torch", "torchvision")
)


@unittest.skipUnless(HAVE, "semantic conversion dependencies required")
class TestContext(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.context_membership"),
            "native PC59 conversion missing",
        )
        from downstream import context_membership

        self.api = context_membership
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.voc = self.root / "VOC2010"
        self.context = self.root / "context"
        self.out = self.root / "converted"

    def fixture(self):
        import numpy as np
        from PIL import Image
        from scipy.io import savemat

        self.context.mkdir()
        self.labels = "0:background\n1:unselected\n" + "\n".join(
            f"{i}:{n}" for i, n in zip(self.api.NATIVE_IDS, self.api.CLASSES)
        )
        (self.context / "labels.txt").write_text(self.labels)
        for sp, name, color in [("train", "alpha", 20), ("val", "beta", 110)]:
            path = self.voc / "ImageSets/Main" / f"{sp}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + "\n")
            image = self.voc / "JPEGImages" / f"{name}.jpg"
            image.parent.mkdir(exist_ok=True)
            Image.fromarray(np.full((2, 3, 3), color, np.uint8)).save(image)
            savemat(
                self.context / (name + ".mat"),
                {"LabelMap": np.array([[0, 2, 458], [9, 1, 284]], np.uint16)},
            )

    def test_mapping_staging_and_real_semantic_loader(self):
        import numpy as np
        from PIL import Image

        self.fixture()
        self.api.convert(self.voc, self.context, self.out, fixture_counts=(1, 1))
        obj = json.loads((self.out / "samples.json").read_text())
        self.assertEqual(len(obj["classes"]), 59)
        self.assertEqual(obj["classes"][36], "person")
        row = obj["train"][0]
        np.testing.assert_array_equal(
            np.array(Image.open(self.out / row["mask"])), [[255, 0, 58], [1, 255, 36]]
        )
        self.assertEqual(
            (self.out / row["image"]).read_bytes(),
            (self.voc / "JPEGImages/alpha.jpg").read_bytes(),
        )
        from downstream.extended_segmentation import load_data

        train, val, _meta = load_data(
            self.out / "samples.json", self.out, "captured_fixed224_v1"
        )
        self.assertEqual((len(train), len(val)), (1, 1))
        self.assertEqual(train[0][1].shape, (224, 224))
        text = (self.out / "sources.json").read_text()
        self.assertNotIn(str(self.root), text)
        self.assertEqual(len(json.loads(text)["sources"]), 7)
        (self.context / "labels.txt").unlink()
        with self.assertRaises(FileExistsError):
            self.api.convert(self.voc, self.context, self.out, fixture_counts=(1, 1))

    def test_swapped_names_unknown_ids_and_missing_labelmap_fail_before_output(self):
        import numpy as np
        from scipy.io import savemat

        self.fixture()
        (self.context / "labels.txt").write_text(
            self.labels.replace("2:aeroplane", "2:wrong")
        )
        with self.assertRaises(ValueError):
            self.api.convert(self.voc, self.context, self.out, fixture_counts=(1, 1))
        self.assertFalse(self.out.exists())
        (self.context / "labels.txt").write_text(self.labels)
        for values in (
            {"LabelMap": np.full((2, 3), 459, np.uint16)},
            {"LabelMap": np.ones((2, 3), np.float32)},
            {"image": np.zeros((2, 3), np.uint8)},
        ):
            savemat(self.context / "alpha.mat", values)
            with self.assertRaises(ValueError):
                self.api.convert(
                    self.voc, self.context, self.out, fixture_counts=(1, 1)
                )
            self.assertFalse(self.out.exists())

    def test_lists_are_main_not_segmentation_and_independence_is_enforced(self):
        self.fixture()
        other = self.voc / "ImageSets/Segmentation"
        other.mkdir()
        (other / "train.txt").write_text("decoy\n")
        (self.voc / "ImageSets/Main/val.txt").write_text("alpha\n")
        with self.assertRaises(ValueError):
            self.api.convert(self.voc, self.context, self.out, fixture_counts=(1, 1))
        (self.voc / "ImageSets/Main/val.txt").write_text("beta\n")
        (self.voc / "JPEGImages/beta.jpg").write_bytes(
            (self.voc / "JPEGImages/alpha.jpg").read_bytes()
        )
        with self.assertRaises(ValueError):
            self.api.convert(self.voc, self.context, self.out, fixture_counts=(1, 1))
        self.assertFalse(self.out.exists())

    def test_population_paths_and_cli_fail_closed(self):
        self.fixture()
        with self.assertRaises(ValueError):
            self.api.convert(self.voc, self.context, self.out)
        self.assertFalse(self.out.exists())
        with self.assertRaises(ValueError):
            self.api.main(
                [
                    "--voc-root",
                    str(self.voc),
                    "--context-root",
                    str(self.context),
                    "--out",
                    str(self.out),
                ]
            )
        self.assertFalse(self.out.exists())
        (self.voc / "ImageSets/Main/train.txt").write_text("../alpha\n")
        with self.assertRaises(ValueError):
            self.api.convert(self.voc, self.context, self.out, fixture_counts=(1, 1))

    @unittest.skipUnless(
        importlib.util.find_spec("transformers"), "encoder dependencies required"
    )
    def test_staged_inputs_train_lp_ap_with_real_provider(self):
        from downstream import extended_segmentation as seg
        from tests.test_method_extended_structured import TestExecution as Structured

        self.fixture()
        self.api.convert(self.voc, self.context, self.out, fixture_counts=(1, 1))
        Structured.encoder(self)
        for adaptation in ("frozen", "attentive"):
            cfg = Structured.config(self, "pascal_context", adaptation)
            cfg.update(
                task="extended_semantic_segmentation",
                transform_profile="captured_fixed224_v1",
                data_root=str(self.out),
                samples=str(self.out / "samples.json"),
            )
            dest = self.root / adaptation
            dest.mkdir()
            result = seg.run(cfg, dest)
            self.assertGreater(result["pixels"], 0)
            report = json.loads((dest / "results.json").read_text())
            self.assertFalse(report["canonical_eligible"])
