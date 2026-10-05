"""Prepared membership preserves source labels, boundaries and provenance."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DATASETS = (
    "action40",
    "cifar10",
    "cifar100",
    "kmnist",
    "imagenet_1percent",
    "imagenet_10percent",
    "omniglot15",
)


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow required")
class PreparedImages(unittest.TestCase):
    def setUp(self):
        from downstream import native_images

        self.api = native_images
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "source"
        self.out = self.base / "staged"

    def fixture(self, dataset, *, split_local=False):
        from PIL import Image

        classes = (
            ("n00000002", "n00000001") if dataset.startswith("imagenet") else ("1", "0")
        )
        for split, value in (("train", 20), ("val", 80)):
            for i, cls in enumerate(classes):
                name = "0.png" if split_local else split + ".png"
                path = self.root / split / cls / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (13, 17), (value, i * 40, 90)).save(path)
        return sorted(classes)

    def convert(self, dataset="action40", **kwargs):
        return self.api.convert(
            dataset, self.root, self.out, fixture_counts=(2, 2, 2), **kwargs
        )

    def test_all_seven_memberships_labels_bytes_and_disclosures(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                self.root = self.base / dataset
                self.out = self.base / (dataset + "-out")
                classes = self.fixture(dataset)
                data = self.convert(dataset)
                self.assertEqual(data["classes"], classes)
                for split, source in (("train", "train"), ("validation", "val")):
                    self.assertEqual(
                        data[split],
                        [
                            {"path": f"{source}/{c}/{source}.png", "target": i}
                            for i, c in enumerate(classes)
                        ],
                    )
                    for row in data[split]:
                        self.assertEqual(
                            (self.out / row["path"]).read_bytes(),
                            (self.root / row["path"]).read_bytes(),
                        )
                ev = json.loads(data["split_evidence"])
                self.assertFalse(ev["authenticity_verified"])
                self.assertFalse(ev["upstream_membership_verified"])
                self.assertEqual(
                    ev["membership_policy"],
                    "prepared splits; no sampling or inferred membership",
                )
                self.assertEqual(ev["transform_reference"], "captured_rgb_rrc_v1")
                self.assertEqual(
                    ev["small_image_recipe_conflict"],
                    dataset in ("cifar10", "cifar100", "kmnist", "omniglot15"),
                )
                self.assertEqual(
                    ev["item_identity"],
                    "split/class/filename"
                    if dataset in ("cifar10", "cifar100", "kmnist")
                    else "class/filename",
                )
                self.assertEqual(
                    ev["publisher_partition"],
                    "not the disjoint-alphabet split"
                    if dataset == "omniglot15"
                    else "unverified",
                )
                self.assertNotIn(
                    str(self.root), (self.out / "samples.json").read_text()
                )
                self.assertNotIn(
                    str(self.root), (self.out / "sources.json").read_text()
                )

    def test_split_local_indices_are_distinct_but_global_ids_cannot_overlap(self):
        for dataset in DATASETS:
            self.root = self.base / dataset
            self.out = self.base / (dataset + "-out")
            self.fixture(dataset, split_local=True)
            if dataset in ("cifar10", "cifar100", "kmnist"):
                self.assertEqual(len(self.convert(dataset)["validation"]), 2)
            else:
                with self.assertRaisesRegex(ValueError, "identity overlap"):
                    self.convert(dataset)
                self.assertFalse(self.out.exists())

    def test_external_imagenet_evaluation_ignores_unused_subset_link(self):
        self.fixture("imagenet_1percent")
        external = self.base / "evaluation"
        external.mkdir()
        (self.root / "val").rename(external / "val")
        (self.root / "val").symlink_to(external / "val", target_is_directory=True)
        data = self.convert("imagenet_1percent", eval_data_root=external)
        self.assertEqual(len(data["validation"]), 2)
        rows = json.loads((self.out / "sources.json").read_text())["sources"]
        self.assertEqual(
            {r["source_role"] for r in rows}, {"training_root", "evaluation_root"}
        )
        self.assertNotIn(str(external), json.dumps(data))
        self.assertEqual(
            (self.out / data["validation"][0]["path"]).read_bytes(),
            (external / data["validation"][0]["path"]).read_bytes(),
        )

    def test_external_root_restricted_safe_and_read_only(self):
        self.fixture("imagenet_1percent")
        for dataset, root, diagnostic in (
            ("action40", self.root, "ImageNet subsets"),
            (
                "imagenet_1percent",
                self.base / "missing",
                "evaluation source must be a directory",
            ),
        ):
            with self.assertRaisesRegex(ValueError, diagnostic):
                self.convert(dataset, eval_data_root=root)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.api.convert(
                "imagenet_1percent",
                self.root,
                self.root / "out",
                eval_data_root=self.root,
                fixture_counts=(2, 2, 2),
            )
        other = self.base / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.api.convert(
                "imagenet_1percent",
                self.root,
                other / "out",
                eval_data_root=other,
                fixture_counts=(2, 2, 2),
            )
        link = self.base / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.convert("imagenet_1percent", eval_data_root=link)
        self.assertFalse(self.out.exists())

    def test_layout_missing_empty_nested_and_unexpected_files_are_refused(self):
        self.fixture("action40")
        for name in ("train/empty", "val/empty"):
            p = self.root / name
            p.mkdir()
            with self.assertRaisesRegex(ValueError, "empty class"):
                self.convert()
            p.rmdir()
        for name in ("train/0/nested/image.png", "val/0/readme.txt", "train/image.png"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "split/class/image"):
                self.convert()
            p.unlink()
        (self.root / "val").rename(self.root / "heldout")
        with self.assertRaisesRegex(ValueError, "missing split"):
            self.convert()
        self.assertFalse(self.out.exists())

    def test_empty_symlink_directories_are_refused_as_links(self):
        self.fixture("action40")
        empty = self.base / "empty"
        empty.mkdir()
        selected = self.root / "val/linked"
        selected.symlink_to(empty, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.convert()
        selected.unlink()
        (self.root / "val").rename(self.root / "saved")
        (self.root / "val").symlink_to(empty, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.convert()
        self.assertFalse(self.out.exists())

    def test_class_vocabulary_patterns_and_population_are_checked(self):
        self.fixture("action40")
        (self.root / "val/1").rename(self.root / "val/2")
        with self.assertRaisesRegex(ValueError, "class coverage"):
            self.convert()
        (self.root / "val/2").rename(self.root / "val/1")
        for dataset in ("kmnist", "imagenet_1percent"):
            (self.root / "train/1").rename(self.root / "train/invalid")
            (self.root / "val/1").rename(self.root / "val/invalid")
            with self.assertRaisesRegex(ValueError, "class name"):
                self.convert(dataset)
            (self.root / "train/invalid").rename(self.root / "train/1")
            (self.root / "val/invalid").rename(self.root / "val/1")
        with self.assertRaisesRegex(ValueError, "class count"):
            self.api.convert("action40", self.root, self.out, fixture_counts=(2, 2, 3))
        with self.assertRaisesRegex(ValueError, "split count"):
            self.api.convert("action40", self.root, self.out, fixture_counts=(3, 2, 2))
        self.assertFalse(self.out.exists())

    def test_release_quotas_are_not_bypassed_by_matching_total(self):
        from downstream import prepared_images

        self.fixture("action40")
        with (
            patch.dict(prepared_images.COUNTS, {"action40": (2, 2, 2)}),
            self.assertRaisesRegex(ValueError, "per-class count"),
        ):
            self.api.convert("action40", self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_production_evaluation_quota_and_variable_training_quota(self):
        from downstream import prepared_images

        self.fixture("imagenet_1percent")
        for cls in ("n00000001", "n00000002"):
            directory = self.root / "val" / cls
            raw = (directory / "val.png").read_bytes()
            for i in range(49):
                (directory / f"v{i}.png").write_bytes(raw)
        with patch.dict(prepared_images.COUNTS, {"imagenet_1percent": (2, 100, 2)}):
            data = self.api.convert("imagenet_1percent", self.root, self.out)
            self.assertEqual(len(data["validation"]), 100)
            self.assertFalse(json.loads(data["split_evidence"])["fixture"])
            self.out = self.base / "unbalanced"
            (self.root / "val/n00000001/v0.png").rename(
                self.root / "val/n00000002/extra.png"
            )
            with self.assertRaisesRegex(ValueError, "per-class count"):
                self.api.convert("imagenet_1percent", self.root, self.out)
            self.assertFalse(self.out.exists())

    def test_links_corruption_and_source_change_never_publish_manifest(self):
        self.fixture("action40")
        p = self.root / "train/0/train.png"
        original = p.read_bytes()
        p.unlink()
        p.symlink_to(self.root / "val/0/val.png")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.convert()
        p.unlink()
        p.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            self.convert()
        p.write_bytes(original)
        checked = self.api.checked_image

        def change(raw):
            checked(raw)
            p.write_bytes(b"changed")

        with (
            patch.object(self.api, "checked_image", side_effect=change),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            self.convert()
        self.assertFalse((self.out / "samples.json").exists())

    def test_content_overlap_disclosed_without_repartitioning_and_metadata_ignored(
        self,
    ):
        self.fixture("action40")
        (self.root / "val/0/val.png").write_bytes(
            (self.root / "train/0/train.png").read_bytes()
        )
        (self.root / "train/.DS_Store").write_bytes(b"metadata")
        data = self.convert()
        self.assertEqual(
            json.loads(data["split_evidence"])["cross_split_content_hashes"], 1
        )
        self.assertEqual(len(data["train"]) + len(data["validation"]), 4)
        with self.assertRaises(FileExistsError):
            self.convert()

    def test_cli_forwards_explicit_evaluation_root_but_rejects_fixture_overrides(self):
        cfg = self.base / "input.json"
        value = {
            "dataset": "imagenet_1percent",
            "data_root": str(self.root),
            "eval_data_root": str(self.base / "eval"),
        }
        cfg.write_text(json.dumps(value))
        with patch.object(self.api, "convert") as convert:
            self.assertEqual(
                self.api.main(["--config", str(cfg), "--out", str(self.out)]), 0
            )
            convert.assert_called_once_with(
                value["dataset"],
                value["data_root"],
                str(self.out),
                eval_data_root=value["eval_data_root"],
            )
        cfg.write_text(json.dumps(dict(value, fixture_counts=[2, 2, 2])))
        with self.assertRaises(ValueError):
            self.api.main(["--config", str(cfg), "--out", str(self.out)])


class Delivery(unittest.TestCase):
    def test_example_is_executable_and_ci_runs_this_module(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "docs/examples/extended_prepared_images.json"
        self.assertTrue(path.is_file(), "prepared image example missing")
        cfg = json.loads(path.read_text())
        self.assertEqual(cfg["dataset"], "action40")
        self.assertEqual(cfg["transform_profile"], "captured_rgb_rrc_v1")
        if importlib.util.find_spec("torchvision"):
            from downstream.extended_classification import validate_config

            validate_config(cfg)
        from tests.test_basic5_finetune_tasks import _runs_finetune_tests
        from tests.test_ci import HAVE_YAML, parsed

        if HAVE_YAML:
            command = next(
                s["run"]
                for s in parsed()["tests.yml"]["jobs"]["downstream"]["steps"]
                if s.get("name")
                == "Run Basic5 component contracts with downstream dependencies"
            )
            self.assertTrue(
                _runs_finetune_tests(
                    command, module="tests.test_extended_prepared_images"
                )
            )


@unittest.skipUnless(
    all(
        importlib.util.find_spec(n)
        for n in ("torch", "torchvision", "transformers", "PIL")
    ),
    "downstream integration dependencies required",
)
class Training(unittest.TestCase):
    def test_staged_seven_datasets_train_lp_ap_and_keep_evidence(self):
        import torch
        from transformers import SiglipVisionConfig, SiglipVisionModel

        from downstream import extended_classification as ex

        torch.set_num_threads(1)
        owner = PreparedImages()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
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
        for dataset in DATASETS:
            owner.root = owner.base / (dataset + "-source")
            owner.fixture(dataset)
            owner.out = owner.base / dataset
            data = owner.convert(dataset)
            for adaptation in ("frozen", "attentive"):
                out = owner.base / (dataset + "-" + adaptation)
                out.mkdir()
                cfg = {
                    "task": ex.TASK,
                    "profile": ex.PROFILE,
                    "dataset": dataset,
                    "seed": 0,
                    "device": "cpu",
                    "data_root": str(owner.out),
                    "samples": str(owner.out / "samples.json"),
                    "transform_profile": "captured_rgb_rrc_v1",
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
                ex.run(cfg, out)
                report = json.loads((out / "results.json").read_text())
                self.assertEqual(report["updates"], 1)
                self.assertEqual(report["final"]["images"], 2)
                self.assertEqual(
                    report["membership"]["split_evidence"], data["split_evidence"]
                )
                self.assertFalse(report["canonical_eligible"])
                self.assertFalse(report["record_value"])
                self.assertTrue((out / "resume.pt").is_file())
