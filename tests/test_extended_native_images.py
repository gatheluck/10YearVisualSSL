"""Native CLUE/SUN membership, byte preservation and disclosure behavior."""

import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow required")
class NativeImages(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.native_images"),
            "native CLUE/SUN staging is missing",
        )
        from downstream import native_images

        self.api = native_images
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.out = self.base / "staged"

    def image(self, color, fmt="PNG"):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (11, 9), color).save(buf, format=fmt)
        return buf.getvalue()

    def clue(self):
        for sp, n in (("train", 1), ("test", 2), ("valid", 3)):
            for i, cls in enumerate(("zebra", "ant")):
                # Same submission across splits, but distinct photographs.
                p = self.root / sp / cls / f"{i + 10}_{n}_bbox0.png"
                p.parent.mkdir(parents=True)
                p.write_bytes(self.image((n * 30, i * 80, 50)))

    def sun(self, *, duplicate=False):
        folder = self.root / "protocol_tar"
        (folder / "shards").mkdir(parents=True)
        rows = []
        for sp, n in (("train", 1), ("test", 2)):
            with tarfile.open(folder / "shards" / f"{sp}.tar", "w") as tar:
                for label, cls in enumerate(("zebra", "ant")):
                    raw = self.image((20 if duplicate else n * 20, label * 80, 1))
                    digest = hashlib.sha256(raw).hexdigest()
                    member = f"images/{digest}.png"
                    info = tarfile.TarInfo(member)
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))
                    rows.append(
                        {
                            "archive": f"shards/{sp}.tar",
                            "member": member,
                            "class_name": cls,
                            "label": label,
                            "split": sp,
                            "bytes": len(raw),
                            "content_sha256": digest,
                        }
                    )
        self.save_sun(rows)
        return rows

    def save_sun(self, rows):
        (self.root / "protocol_tar/manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )

    def convert(self, dataset):
        return self.api.convert(dataset, self.root, self.out, fixture_counts=(2, 2, 2))

    def test_all_clue_variants_keep_test_and_submission_disclosure(self):
        self.clue()
        for variant in ("egg", "feather", "footprint", "skulls", "stools"):
            self.out = self.base / variant
            data = self.convert("clue_" + variant)
            self.assertEqual(data["classes"], ["ant", "zebra"])
            self.assertEqual([r["target"] for r in data["train"]], [0, 1])
            self.assertTrue(
                all(r["path"].startswith("test/") for r in data["validation"])
            )
            self.assertFalse((self.out / "valid").exists())
            evidence = json.loads(data["split_evidence"])
            groups = evidence["submission_groups"]
            self.assertEqual(groups["status"], "cross_split_submission_overlap")
            self.assertEqual(groups["overlaps"]["test__train"], 2)
            self.assertEqual(groups["unknown"], {"train": 0, "test": 0, "valid": 0})
            for r in data["train"] + data["validation"]:
                self.assertEqual(
                    (self.out / r["path"]).read_bytes(),
                    (self.root / r["path"]).read_bytes(),
                )
            self.assertNotIn(str(self.root), (self.out / "sources.json").read_text())

    def test_clue_missing_test_unknown_class_and_symlinks_fail_before_output(self):
        self.clue()
        p = self.root / "test/ant/11_2_bbox0.png"
        original = p.read_bytes()
        p.unlink()
        with self.assertRaises(ValueError):
            self.convert("clue_egg")
        self.assertFalse(self.out.exists())
        p.write_bytes(original)
        (self.root / "test/ant").rename(self.root / "test/unknown")
        with self.assertRaisesRegex(ValueError, "unknown CLUE evaluation class"):
            self.convert("clue_egg")
        (self.root / "test/unknown").rename(self.root / "test/ant")
        p.unlink()
        p.symlink_to(self.root / "train/ant/11_1_bbox0.png")
        with self.assertRaises(ValueError):
            self.convert("clue_egg")
        self.assertFalse(self.out.exists())

    def test_clue_unknown_submission_ids_are_not_certified(self):
        self.clue()
        p = self.root / "test/ant/11_2_bbox0.png"
        p.rename(p.with_name("anonymous.png"))
        data = self.convert("clue_egg")
        ev = json.loads(data["split_evidence"])["submission_groups"]
        self.assertEqual(ev["unknown"]["test"], 1)
        self.assertFalse(ev["all_ids_known"])

    def test_clue_vocabulary_count_is_not_inferred(self):
        self.clue()
        with self.assertRaisesRegex(ValueError, "class count"):
            self.api.convert("clue_egg", self.root, self.out, fixture_counts=(2, 2, 3))
        self.assertFalse(self.out.exists())

    def test_sun_preserves_non_alphabetical_labels_and_occurrences(self):
        rows = self.sun(duplicate=True)
        data = self.convert("sun397")
        self.assertEqual(data["classes"], ["zebra", "ant"])
        ordered = [
            r
            for sp in ("train", "test")
            for r in sorted(
                (r for r in rows if r["split"] == sp),
                key=lambda r: r["archive"] + "/" + r["member"],
            )
        ]
        self.assertEqual(
            [r["target"] for r in data["train"]], [r["label"] for r in ordered[:2]]
        )
        self.assertEqual(
            len({r["path"] for sp in ("train", "validation") for r in data[sp]}), 4
        )
        ev = json.loads(data["split_evidence"])
        self.assertEqual(ev["cross_split_content_hashes"], 2)
        self.assertFalse(ev["upstream_filename_join_verified"])
        self.assertFalse(ev["authenticity_verified"])
        for row, source in zip(data["train"] + data["validation"], ordered):
            self.assertEqual(
                hashlib.sha256((self.out / row["path"]).read_bytes()).hexdigest(),
                source["content_sha256"],
            )

    def test_sun_per_split_coverage_and_duplicate_json_keys_fail(self):
        rows = self.sun()
        changed = [dict(r) for r in rows]
        changed[-1].update(label=0, class_name="zebra")
        self.save_sun(changed)
        with self.assertRaisesRegex(ValueError, "split class coverage"):
            self.convert("sun397")
        self.save_sun(rows)
        path = self.root / "protocol_tar/manifest.jsonl"
        path.write_text(
            path.read_text().replace('"label": 0', '"label": 1, "label": 0', 1)
        )
        with self.assertRaises(ValueError):
            self.convert("sun397")
        self.assertFalse(self.out.exists())

    def test_clue_content_overlap_policy_matches_selected_reference(self):
        self.clue()
        (self.root / "test/ant/11_2_bbox0.png").write_bytes(
            (self.root / "train/ant/11_1_bbox0.png").read_bytes()
        )
        for variant in ("footprint", "stools"):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.convert("clue_" + variant)
            self.assertFalse(self.out.exists())
        for variant in ("egg", "feather", "skulls"):
            self.out = self.base / variant
            data = self.convert("clue_" + variant)
            self.assertEqual(
                json.loads(data["split_evidence"])["cross_split_content_hashes"], 1
            )

    def test_sun_declared_hash_length_mapping_and_schema_are_enforced(self):
        rows = self.sun()
        for field, value in (
            ("bytes", 1),
            ("content_sha256", "0" * 64),
            ("label", True),
            ("class_name", "other"),
            ("split", "valid"),
            ("archive", "../escape.tar"),
            ("member", "../escape.png"),
        ):
            changed = [dict(r) for r in rows]
            changed[-1][field] = value
            self.save_sun(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.convert("sun397")
            self.assertFalse(self.out.exists())
        self.save_sun(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate SUN occurrence"):
            self.convert("sun397")

    def test_sun_rejects_ambiguous_tar_and_link_members(self):
        rows = self.sun()
        path = self.root / "protocol_tar/shards/train.tar"
        for link in (False, True):
            with tarfile.open(path, "w") as tar:
                for r in rows[:2]:
                    raw = self.image((20, r["label"] * 80, 1))
                    info = tarfile.TarInfo(r["member"])
                    info.size = len(raw)
                    if link:
                        info.type = tarfile.SYMTYPE
                        info.linkname = "/outside"
                    tar.addfile(info, None if link else io.BytesIO(raw))
                    if not link:
                        tar.addfile(info, io.BytesIO(raw))
            with (
                self.subTest(link=link),
                self.assertRaisesRegex(ValueError, "ambiguous or linked TAR"),
            ):
                self.convert("sun397")
            self.assertFalse(self.out.exists())

    def test_outputs_are_exclusive_and_inputs_cannot_be_overwritten(self):
        self.clue()
        data = self.convert("clue_egg")
        before = (self.out / "samples.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.convert("clue_egg")
        self.assertEqual((self.out / "samples.json").read_bytes(), before)
        self.out = self.root / "new"
        with self.assertRaises(ValueError):
            self.convert("clue_egg")
        self.assertFalse(self.out.exists())
        self.assertEqual(len(data["validation"]), 2)

    def test_cli_rejects_fixture_counts_and_incomplete_release(self):
        from unittest.mock import patch

        self.clue()
        cfg = self.base / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "dataset": "clue_egg",
                    "data_root": str(self.root),
                    "fixture_counts": [2, 2, 2],
                }
            )
        )
        with patch.object(self.api, "convert") as convert:
            with self.assertRaises(ValueError):
                self.api.main(["--config", str(cfg), "--out", str(self.out)])
            convert.assert_not_called()
        cfg.write_text(json.dumps({"dataset": "clue_egg", "data_root": str(self.root)}))
        with self.assertRaises(ValueError):
            self.api.main(["--config", str(cfg), "--out", str(self.out)])
        self.assertFalse(self.out.exists())

    def test_partial_write_never_publishes_success_manifest(self):
        from unittest.mock import patch

        self.clue()
        with (
            patch.object(Path, "write_bytes", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            self.convert("clue_egg")
        self.assertFalse((self.out / "samples.json").exists())

    def test_source_change_and_corrupt_image_refuse_success(self):
        from unittest.mock import patch

        self.clue()
        original = self.api.checked_image

        def changed(raw):
            original(raw)
            (self.root / "train/ant/11_1_bbox0.png").write_bytes(
                self.image((100, 2, 3))
            )

        with (
            patch.object(self.api, "checked_image", side_effect=changed),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            self.convert("clue_egg")
        self.assertFalse((self.out / "samples.json").exists())
        self.out = self.base / "next"
        (self.root / "train/ant/11_1_bbox0.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            self.convert("clue_egg")
        self.assertFalse(self.out.exists())

    def test_sun_hash_is_verified_against_actual_payload(self):
        rows = self.sun()
        for sp in ("train", "test"):
            with tarfile.open(self.root / f"protocol_tar/shards/{sp}.tar", "w") as tar:
                for row in (r for r in rows if r["split"] == sp):
                    raw = self.image((60, row["label"] * 80, 1))
                    row["bytes"] = len(raw)
                    info = tarfile.TarInfo(row["member"])
                    info.size = len(raw)
                    tar.addfile(info, io.BytesIO(raw))
        self.save_sun(rows)
        with self.assertRaisesRegex(ValueError, "content hash"):
            self.convert("sun397")
        self.assertFalse(self.out.exists())

    def test_schema_boundaries_and_release_quota(self):
        from unittest.mock import patch

        rows = self.sun()
        bad_values = (
            ("label", -1),
            ("label", 2),
            ("class_name", " "),
            ("class_name", " zebra"),
            ("content_sha256", 1),
            ("content_sha256", "bad"),
            ("bytes", True),
            ("bytes", 0),
            ("archive", "wrong.tar"),
            ("archive", "/absolute.tar"),
            ("archive", "shards//train.tar"),
            ("archive", "shards/./train.tar"),
            ("archive", "shards\\train.tar"),
            ("archive", ""),
            ("member", "images/" + rows[0]["content_sha256"] + ".exe"),
        )
        for field, value in bad_values:
            changed = [dict(r) for r in rows]
            changed[0][field] = value
            self.save_sun(changed)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.convert("sun397")
        for changed in ([{}] + rows[1:], []):
            self.save_sun(changed)
            with self.assertRaises(ValueError):
                self.convert("sun397")
        self.save_sun(rows)
        with (
            patch.dict(self.api.COUNTS, {"sun397": (2, 2, 2)}),
            self.assertRaisesRegex(ValueError, "50 occurrences"),
        ):
            self.api.convert("sun397", self.root, self.out)
        for counts in ((True, 2, 2), (0, 2, 2), (2, 2)):
            with self.assertRaises(ValueError):
                self.api.convert("sun397", self.root, self.out, fixture_counts=counts)
        with self.assertRaises(ValueError):
            self.convert("unknown")
        self.assertFalse(self.out.exists())

    def test_sun_field_diagnostics_precede_population_or_payload_errors(self):
        rows = self.sun()
        for field, value, diagnostic in (
            ("class_name", " ", "invalid SUN class"),
            ("split", "valid", "invalid SUN split"),
            ("content_sha256", "BAD", "invalid SUN hash"),
            ("content_sha256", "0" * 64, "content-addressed member"),
            ("bytes", True, "invalid SUN byte count"),
        ):
            changed = [dict(r) for r in rows]
            changed[0][field] = value
            self.save_sun(changed)
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(ValueError, diagnostic),
            ):
                self.convert("sun397")
            self.assertFalse(self.out.exists())

    def test_missing_sun_shard_member_and_complete_class_vocabulary(self):
        rows = self.sun()
        shard = self.root / "protocol_tar/shards/train.tar"
        saved = shard.read_bytes()
        shard.unlink()
        with self.assertRaises(ValueError):
            self.convert("sun397")
        with tarfile.open(shard, "w"):
            pass
        with self.assertRaises(ValueError):
            self.convert("sun397")
        shard.write_bytes(saved)
        changed = [dict(r, label=0, class_name="zebra") for r in rows]
        self.save_sun(changed)
        with self.assertRaisesRegex(ValueError, "class coverage"):
            self.convert("sun397")
        changed = [dict(r, class_name="same") for r in rows]
        self.save_sun(changed)
        with self.assertRaisesRegex(ValueError, "not unique"):
            self.convert("sun397")
        self.assertFalse(self.out.exists())


class Delivery(unittest.TestCase):
    def test_documented_examples_and_required_ci(self):
        root = Path(__file__).resolve().parents[1]
        guide = root / "docs/EXTENDED_NATIVE_IMAGES.md"
        self.assertTrue(guide.is_file(), "native image guide missing")
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
                    command, module="tests.test_extended_native_images"
                )
            )
        for name in ("clue", "sun397"):
            path = root / f"docs/examples/extended_{name}.json"
            self.assertTrue(path.is_file())
            cfg = json.loads(path.read_text())
            self.assertEqual(cfg["transform_profile"], "captured_rgb_rrc_v1")
            if importlib.util.find_spec("torchvision"):
                from downstream.extended_classification import validate_config

                validate_config(cfg)


@unittest.skipUnless(
    all(
        importlib.util.find_spec(n)
        for n in ("torch", "torchvision", "transformers", "PIL")
    ),
    "downstream integration dependencies required",
)
class Training(unittest.TestCase):
    def test_staged_six_datasets_train_lp_ap_and_keep_evidence(self):
        import torch
        from transformers import SiglipVisionConfig, SiglipVisionModel

        from downstream import extended_classification as ex

        torch.set_num_threads(1)
        owner = NativeImages()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        owner.clue()
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
        for dataset in owner.api.COUNTS:
            if dataset == "sun397":
                owner.sun()
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
