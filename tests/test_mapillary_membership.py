"""Full native split joins, numeric labels and safe Mapillary staging."""

import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

HAVE = all(
    importlib.util.find_spec(n)
    for n in ("scipy", "numpy", "PIL", "torch", "torchvision")
)


@unittest.skipUnless(HAVE, "semantic conversion dependencies required")
class TestMapillary(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.mapillary_membership"),
            "Mapillary native staging missing",
        )
        from downstream import mapillary_membership

        self.api = mapillary_membership
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images.zip"
        self.labels = self.root / "labels.zip"
        self.out = self.root / "out"

    def fixture(self, missing=False, bad=False, duplicate=False, overlap=False):
        import numpy as np
        from PIL import Image

        with (
            zipfile.ZipFile(self.images, "w") as im,
            zipfile.ZipFile(self.labels, "w") as labels,
        ):
            for split, n in [("training", 3), ("validation", 1)]:
                for i in range(n):
                    b = io.BytesIO()
                    Image.fromarray(
                        np.full(
                            (2, 3, 3),
                            10 + i if split == "training" or overlap else 100,
                            np.uint8,
                        )
                    ).save(b, format="JPEG")
                    member = f"{split}/images/{i}.jpg"
                    im.writestr(member, b.getvalue())
                    if duplicate and split == "validation":
                        im.writestr(member, b.getvalue())
                    b = io.BytesIO()
                    mask = np.array([[0, 65, 255], [1, 2, 3]], np.uint8)
                    if bad:
                        mask[0, 0] = 66
                    Image.fromarray(mask).save(b, format="PNG")
                    if not (missing and split == "validation"):
                        labels.writestr(f"{split}/v1.2/labels/{i}.png", b.getvalue())
            # Decoys from a different ontology never replace v1.2 labels.
            labels.writestr("validation/v2.0/labels/orphan.png", b.getvalue())

    def test_all_listed_pairs_and_class65_are_preserved(self):
        import numpy as np
        from PIL import Image

        self.fixture()
        self.api.convert(self.images, self.labels, self.out, fixture_counts=(3, 1))
        obj = json.loads((self.out / "samples.json").read_text())
        self.assertEqual((len(obj["train"]), len(obj["validation"])), (3, 1))
        self.assertEqual(len(obj["classes"]), 66)
        np.testing.assert_array_equal(
            np.array(Image.open(self.out / obj["train"][0]["mask"])),
            [[0, 65, 255], [1, 2, 3]],
        )
        from downstream.extended_segmentation import load_data

        train, val, _ = load_data(
            self.out / "samples.json", self.out, "captured_fixed224_v1"
        )
        self.assertEqual(train[0][1].shape, (224, 224))
        self.assertEqual(len(val), 1)
        self.assertNotIn(str(self.root), (self.out / "sources.json").read_text())
        with self.assertRaises(FileExistsError):
            self.api.convert(self.images, self.labels, self.out, fixture_counts=(3, 1))

    def test_incomplete_join_invalid_target_and_image_leakage_fail(self):
        for kwargs in (
            {"missing": True},
            {"bad": True},
            {"duplicate": True},
            {"overlap": True},
        ):
            with self.subTest(kwargs=kwargs):
                self.fixture(**kwargs)
                with self.assertRaises(ValueError):
                    self.api.convert(
                        self.images, self.labels, self.out, fixture_counts=(3, 1)
                    )
                self.assertFalse(self.out.exists())

    def test_release_counts_and_cli_refuse_truncated_population(self):
        self.fixture()
        with self.assertRaises(ValueError):
            self.api.convert(self.images, self.labels, self.out)
        with self.assertRaises(ValueError):
            self.api.main(
                [
                    "--images",
                    str(self.images),
                    "--labels",
                    str(self.labels),
                    "--out",
                    str(self.out),
                ]
            )
        self.assertFalse(self.out.exists())

    @unittest.skipUnless(
        importlib.util.find_spec("transformers"), "encoder dependencies required"
    )
    def test_staged_inputs_train_lp_ap_with_real_provider(self):
        from downstream import extended_segmentation as seg
        from tests.test_method_extended_structured import TestExecution as Structured

        self.fixture()
        self.api.convert(self.images, self.labels, self.out, fixture_counts=(3, 1))
        Structured.encoder(self)
        for adaptation in ("frozen", "attentive"):
            cfg = Structured.config(self, "mapillary_vistas", adaptation)
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
