"""Observed memberships stay distinct from unresolved protocol declarations."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(
    all(importlib.util.find_spec(n) for n in ("PIL", "scipy")),
    "image/MAT dependencies required",
)
class Membership(unittest.TestCase):
    def setUp(self):
        from downstream import native_images

        self.api = native_images
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "source"
        self.out = self.base / "staged"

    def flowers(self):
        from PIL import Image

        self.root.mkdir()
        (self.root / "jpg").mkdir()
        for i in range(1, 7):
            Image.new("RGB", (13, 17), (i * 30, 20, 70)).save(
                self.root / f"jpg/image_{i:05d}.jpg"
            )
        self.sets = {"trnid": [2, 1], "valid": [4, 3], "tstid": [6, 5]}
        self.labels = [2, 1, 1, 2, 2, 1]
        self.save_mat()

    def save_mat(self):
        from scipy.io import savemat

        savemat(self.root / "setid.mat", self.sets)
        savemat(self.root / "imagelabels.mat", {"labels": self.labels})

    def kanji(self):
        from PIL import Image

        for split, offset in (("train", 20), ("val", 100)):
            for i, name in enumerate(("U+4E01", "U+4453")):
                folder = self.root / split / name
                folder.mkdir(parents=True)
                Image.new("RGB", (13, 17), (offset, i * 30, 70)).save(
                    folder / (split + ".png")
                )

    def convert(self, dataset):
        return self.api.convert(dataset, self.root, self.out, fixture_counts=(2, 2, 2))

    def test_flowers_preserves_trnid_test_labels_and_excludes_valid(self):
        self.flowers()
        # Prepared folders are intentionally inconsistent and must not be used.
        (self.root / "train").mkdir()
        (self.root / "train/irrelevant.txt").write_text("not the selected membership")
        data = self.convert("flowers102")
        self.assertEqual(data["classes"], ["1", "2"])
        for split, ids in (("train", (1, 2)), ("validation", (5, 6))):
            self.assertEqual(
                data[split],
                [
                    {"path": f"jpg/image_{i:05d}.jpg", "target": self.labels[i - 1] - 1}
                    for i in ids
                ],
            )
            for row in data[split]:
                self.assertEqual(
                    (self.out / row["path"]).read_bytes(),
                    (self.root / row["path"]).read_bytes(),
                )
        self.assertFalse((self.out / "jpg/image_00003.jpg").exists())
        evidence = json.loads(data["split_evidence"])
        self.assertEqual(evidence["official_train"], "setid.mat:trnid")
        self.assertEqual(evidence["excluded_validation_count"], 2)
        self.assertTrue(evidence["protocol_split_conflict"])
        self.assertEqual(
            set(evidence["annotation_sha256"]), {"setid.mat", "imagelabels.mat"}
        )
        self.assertFalse(evidence["authenticity_verified"])
        self.assertNotIn(str(self.root), json.dumps(data))
        with self.assertRaises(FileExistsError):
            self.convert("flowers102")

    def test_flowers_rejects_invalid_membership_before_publication(self):
        self.flowers()
        cases = [
            ({"trnid": [1, 1]}, self.labels),
            ({"valid": [1, 3]}, self.labels),
            ({"tstid": [5, 7]}, self.labels),
            ({"trnid": [0, 2]}, self.labels),
            ({"trnid": [1.5, 2]}, self.labels),
            ({"valid": [3]}, self.labels),
            ({}, self.labels[:-1]),
            ({}, [3, 1, 1, 2, 2, 1]),
            ({}, [1, 1, 1, 2, 2, 1]),
            ({}, [2, 1, 1, 1, 2, 1]),
            ({}, [2, 1, 1, 2, 1, 1]),
        ]
        for overrides, labels in cases:
            with self.subTest(overrides=overrides, labels=labels):
                self.sets = {
                    "trnid": [2, 1],
                    "valid": [4, 3],
                    "tstid": [6, 5],
                } | overrides
                self.labels = labels
                self.save_mat()
                with self.assertRaises(ValueError):
                    self.convert("flowers102")
                self.assertFalse(self.out.exists())

    def test_flowers_requires_mat_fields_and_all_listed_images(self):
        from scipy.io import savemat

        self.flowers()
        for payload in ({}, {"trnid": [], "valid": [3, 4], "tstid": [5, 6]}):
            savemat(self.root / "setid.mat", payload)
            with self.assertRaisesRegex(ValueError, "annotation"):
                self.convert("flowers102")
        self.save_mat()
        (self.root / "jpg/image_00003.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing source"):
            self.convert("flowers102")
        self.assertFalse(self.out.exists())

    def test_flowers_annotation_diagnostics_identify_the_violated_requirement(self):
        self.flowers()
        for overrides, labels, diagnostic in (
            ({"trnid": [1.5, 2]}, self.labels, "positive integer"),
            ({}, self.labels[:-1], "label count"),
            ({}, [3, 1, 1, 2, 2, 1], "out of range"),
            ({"valid": [1, 3]}, self.labels, "duplicate or overlapping"),
            ({"trnid": [1, 2, 3], "valid": [4]}, self.labels, "validation count"),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.sets = {
                    "trnid": [2, 1],
                    "valid": [4, 3],
                    "tstid": [6, 5],
                } | overrides
                self.labels = labels
                self.save_mat()
                with self.assertRaisesRegex(ValueError, diagnostic):
                    self.convert("flowers102")
                self.assertFalse(self.out.exists())

    def test_flowers_corrupt_mat_symlinks_and_selected_corrupt_images_fail(self):
        self.flowers()
        mat = self.root / "setid.mat"
        raw = mat.read_bytes()
        mat.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "unreadable MAT"):
            self.convert("flowers102")
        mat.unlink()
        mat.symlink_to(self.root / "imagelabels.mat")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.convert("flowers102")
        mat.unlink()
        mat.write_bytes(raw)
        (self.root / "jpg/image_00001.jpg").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "unreadable source image"):
            self.convert("flowers102")
        self.assertFalse(self.out.exists())

    def test_flowers_production_quota_checks_both_training_and_excluded_valid(self):
        from PIL import Image
        from scipy.io import savemat

        from downstream import flowers_membership

        self.root.mkdir()
        (self.root / "jpg").mkdir()
        for i in range(1, 45):
            Image.new("RGB", (9, 9), (i * 4, 40, 70)).save(
                self.root / f"jpg/image_{i:05d}.jpg"
            )
        savemat(
            self.root / "setid.mat",
            {
                "trnid": list(range(1, 21)),
                "valid": list(range(21, 41)),
                "tstid": list(range(41, 45)),
            },
        )
        labels = [1] * 10 + [2] * 10 + [1] * 10 + [2] * 10 + [1, 1, 2, 2]
        with patch.dict(flowers_membership.COUNTS, {"flowers102": (20, 4, 2)}):
            for offset in (9, 29):
                changed = labels.copy()
                changed[offset] = 2
                savemat(self.root / "imagelabels.mat", {"labels": changed})
                with self.assertRaisesRegex(ValueError, "ten images"):
                    self.api.convert("flowers102", self.root, self.out)
                self.assertFalse(self.out.exists())
            savemat(self.root / "imagelabels.mat", {"labels": labels})
            data = self.api.convert("flowers102", self.root, self.out)
            self.assertEqual(len(data["train"]), 20)
            self.assertEqual(len(data["validation"]), 4)
            self.assertFalse(json.loads(data["split_evidence"])["fixture"])

    def test_kanji_preserves_observed_folder_labels_and_discloses_limits(self):
        self.kanji()
        data = self.convert("kuzushiji_kanji")
        self.assertEqual(data["classes"], ["U+4453", "U+4E01"])
        self.assertEqual([r["target"] for r in data["train"]], [0, 1])
        evidence = json.loads(data["split_evidence"])
        self.assertEqual(
            evidence["count_basis"],
            "observed prepared benchmark; not the unsplit publisher release",
        )
        self.assertFalse(evidence["historical_membership_verified"])
        self.assertTrue(evidence["small_image_recipe_conflict"])
        self.assertEqual(evidence["transform_reference"], "captured_small_crop_v1")
        self.assertEqual(evidence["alternate_builder_transform"], "captured_rgb_rrc_v1")
        self.assertEqual(evidence["evaluation_per_class"], 5)
        self.assertFalse(evidence["upstream_membership_verified"])
        self.assertNotIn(str(self.root), json.dumps(data))

    def test_kanji_rejects_nonunicode_classes_and_overlapping_ids(self):
        self.kanji()
        for split in ("train", "val"):
            (self.root / split / "U+4453").rename(self.root / split / "class0")
        with self.assertRaisesRegex(ValueError, "class name"):
            self.convert("kuzushiji_kanji")
        for split in ("train", "val"):
            (self.root / split / "class0").rename(self.root / split / "U+4453")
        (self.root / "val/U+4453/val.png").rename(self.root / "val/U+4453/train.png")
        with self.assertRaisesRegex(ValueError, "identity overlap"):
            self.convert("kuzushiji_kanji")
        self.assertFalse(self.out.exists())

    def test_production_counts_and_per_class_checks_cannot_be_disabled(self):
        from downstream import prepared_images

        self.kanji()
        with self.assertRaisesRegex(ValueError, "class count"):
            self.api.convert("kuzushiji_kanji", self.root, self.out)
        with (
            patch.dict(prepared_images.COUNTS, {"kuzushiji_kanji": (2, 2, 2)}),
            self.assertRaisesRegex(ValueError, "per-class count"),
        ):
            self.api.convert("kuzushiji_kanji", self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_kanji_production_evaluation_quota_with_variable_train_sizes(self):
        from downstream import prepared_images

        self.kanji()
        for cls in ("U+4453", "U+4E01"):
            folder = self.root / "val" / cls
            for i in range(4):
                (folder / f"extra{i}.png").write_bytes(
                    (folder / "val.png").read_bytes()
                )
        folder = self.root / "train/U+4453"
        (folder / "additional.png").write_bytes((folder / "train.png").read_bytes())
        with patch.dict(prepared_images.COUNTS, {"kuzushiji_kanji": (3, 10, 2)}):
            data = self.api.convert("kuzushiji_kanji", self.root, self.out)
            self.assertEqual(len(data["train"]), 3)
            self.assertEqual(len(data["validation"]), 10)
            self.out = self.base / "unbalanced"
            (self.root / "val/U+4453/extra0.png").rename(
                self.root / "val/U+4E01/moved.png"
            )
            with self.assertRaisesRegex(ValueError, "per-class count"):
                self.api.convert("kuzushiji_kanji", self.root, self.out)


@unittest.skipUnless(
    all(
        importlib.util.find_spec(n)
        for n in ("torch", "torchvision", "transformers", "scipy")
    ),
    "downstream integration dependencies required",
)
class Training(unittest.TestCase):
    def test_both_memberships_execute_lp_ap_without_losing_evidence(self):
        import torch
        from transformers import SiglipVisionConfig, SiglipVisionModel

        from downstream import extended_classification as ex

        torch.set_num_threads(1)
        for dataset, fixture in (
            ("flowers102", "flowers"),
            ("kuzushiji_kanji", "kanji"),
        ):
            owner = Membership()
            owner.setUp()
            self.addCleanup(owner.doCleanups)
            getattr(owner, fixture)()
            data = owner.convert(dataset)
            encoder = owner.base / "encoder"
            SiglipVisionModel(
                SiglipVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=32,
                    patch_size=16,
                )
            ).save_pretrained(encoder)
            for adaptation in ("frozen", "attentive"):
                cfg = {
                    "task": ex.TASK,
                    "profile": ex.PROFILE,
                    "dataset": dataset,
                    "seed": 0,
                    "device": "cpu",
                    "data_root": str(owner.out),
                    "samples": str(owner.out / "samples.json"),
                    "transform_profile": "captured_small_crop_v1"
                    if dataset == "kuzushiji_kanji"
                    else "captured_rgb_rrc_v1",
                    "adaptation": adaptation,
                    "reader_profile": "captured_single_block_v1"
                    if adaptation == "attentive"
                    else None,
                    "backbone": {
                        "kind": "siglip2_g",
                        "arch": "fixture",
                        "encoder": str(encoder),
                        "img_size": 32,
                        "patch_size": 16,
                    },
                    "probe": {"epochs": 1, "batch_size": 2, "num_workers": 0},
                }
                out = owner.base / adaptation
                out.mkdir()
                ex.run(cfg, out)
                result = json.loads((out / "results.json").read_text())
                self.assertEqual(result["updates"], 1)
                self.assertEqual(result["final"]["images"], 2)
                self.assertEqual(
                    result["membership"]["split_evidence"], data["split_evidence"]
                )
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])
                self.assertTrue((out / "resume.pt").is_file())


class Delivery(unittest.TestCase):
    def test_profiles_pin_observed_populations_used_by_documented_cli(self):
        from downstream import flowers_membership, prepared_images

        # Measured benchmark inventories are not the publisher's unsplit totals.
        self.assertEqual(flowers_membership.COUNTS["flowers102"], (1020, 6149, 102))
        self.assertEqual(
            prepared_images.COUNTS["kuzushiji_kanji"], (126551, 9830, 1966)
        )

    def test_documented_examples_validate_and_tests_run_in_ci(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("flowers102", "kuzushiji_kanji"):
            path = root / f"docs/examples/extended_{dataset}.json"
            self.assertTrue(path.is_file(), "missing executable example")
            cfg = json.loads(path.read_text())
            self.assertEqual(cfg["dataset"], dataset)
            if importlib.util.find_spec("torchvision"):
                from downstream.extended_classification import validate_config

                validate_config(cfg)
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if HAVE_YAML:
            step = next(
                s["run"]
                for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
                if s.get("name")
                == "Run Basic5 component contracts with downstream dependencies"
            )
            self.assertTrue(
                _runs_finetune_tests(step, module="tests.test_extended_flowers_kanji")
            )
