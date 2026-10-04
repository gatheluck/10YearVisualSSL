"""Native annotations must preserve identities, image splits and task geometry."""

import functools
import hashlib
import importlib.util
import io
import json
import operator
import tempfile
import unittest
import zipfile
from pathlib import Path

HAVE = all(importlib.util.find_spec(x) for x in ("torch", "torchvision", "numpy"))


@unittest.skipUnless(HAVE, "downstream dependencies required")
class TestMembership(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.structured_membership"),
            "Native structured membership builders are missing",
        )
        from downstream import structured_membership

        self.api = structured_membership
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    @staticmethod
    def png(value):
        import numpy as np
        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(np.full((16, 20, 3), value, dtype=np.uint8)).save(
            out, format="PNG"
        )
        return out.getvalue()

    def put(self, name, data):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())

    def test_cambridge_quaternion_is_camera_to_world_and_preserves_official_lists(self):
        with zipfile.ZipFile(self.root / "Room.zip", "w") as z:
            for i, sp in enumerate(("train", "test")):
                z.writestr(
                    f"Room/dataset_{sp}.txt",
                    f"Visual Landmark Dataset\nImageFile X Y Z W X Y Z\n{i}.png 1 2 3 0 0 0 1\n",
                )
                z.writestr(f"Room/{i}.png", self.png(i))
        m = self.api.build(
            "cambridge_landmarks", self.root, fixture_counts={"Room": (1, 1)}
        )
        self.assertEqual(m["scenes"], ["Room"])
        self.assertEqual(
            m["validation"][0]["image"], {"path": "Room.zip", "member": "Room/1.png"}
        )
        self.assertEqual(m["train"][0]["pose"][0], [-1.0, 0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            self.api.build("cambridge_landmarks", self.root)

    def test_seven_scenes_no_truncation_and_split_overlap_refused(self):
        for i, sp in enumerate(("Train", "Test")):
            seq = f"seq-0{i + 1}"
            self.put(f"room/{sp}Split.txt", f"sequence{i + 1}\n")
            with zipfile.ZipFile(self.root / f"room/{seq}.zip", "w") as z:
                for j in range(2):
                    z.writestr(f"{seq}/frame-{j:06}.color.png", self.png(i * 2 + j))
                    z.writestr(
                        f"{seq}/frame-{j:06}.pose.txt",
                        "1 0 0 1\n0 1 0 2\n0 0 1 3\n0 0 0 1\n",
                    )
        m = self.api.build("seven_scenes", self.root, fixture_counts={"room": (2, 2)})
        self.assertEqual(len(m["validation"]), 2)
        self.assertEqual(m["validation"][0]["pose"][0][3], 1.0)
        with self.assertRaises(ValueError):
            self.api.build("seven_scenes", self.root, fixture_counts={"room": (2, 3)})
        self.put("room/TestSplit.txt", "sequence1\n")
        with self.assertRaises(ValueError):
            self.api.build("seven_scenes", self.root, fixture_counts={"room": (2, 2)})

    def test_sr_renderer_does_not_leak_answer_and_joins_image_holdout(self):
        import csv

        fields = [
            "qid",
            "image",
            "qtype",
            "relation",
            "subject",
            "object1",
            "object2",
            "answer",
        ]
        rows = [
            [
                "a",
                "https://example.invalid/train/a.png",
                "orientation",
                "viewpoint",
                "cat",
                "dog",
                "",
                "left",
            ],
            [
                "b",
                "https://example.invalid/val/b.png",
                "multi_object",
                "above",
                "cat",
                "dog",
                "",
                "cat",
            ],
        ]
        text = io.StringIO()
        w = csv.writer(text)
        w.writerow(fields)
        w.writerows(rows)
        self.put("3dsrbench_v1.csv", text.getvalue())
        self.put("images/train/a.png", self.png(1))
        self.put("images/val/b.png", self.png(2))
        self.put(
            "sr_holdout.json",
            json.dumps(
                {
                    "train_ids": ["train/a.png"],
                    "eval_ids": ["val/b.png"],
                    "train_questions": 1,
                    "eval_questions": 1,
                }
            ),
        )
        m = self.api.build("three_d_srbench", self.root, fixture_counts=(1, 1))
        self.assertEqual(m["train"][0]["options"], ["front", "left", "back", "right"])
        self.assertEqual(m["train"][0]["target"], 1)
        r = dict(zip(fields, rows[0]))
        a = self.api.spatial_question(r)
        r["answer"] = "right"
        self.assertEqual(a, self.api.spatial_question(r))
        self.assertIn("not CircularEval", m["split_evidence"])
        self.put(
            "sr_holdout.json",
            json.dumps(
                {
                    "train_ids": ["train/a.png"],
                    "eval_ids": ["train/a.png"],
                    "train_questions": 1,
                    "eval_questions": 1,
                }
            ),
        )
        with self.assertRaises(ValueError):
            self.api.build("three_d_srbench", self.root, fixture_counts=(1, 1))

    def test_crowdpose_gt_crop_visibility_and_all_annotations(self):
        import tarfile

        (self.root / "archives").mkdir()
        with (
            tarfile.open(
                self.root / "archives/CrowdPose_annotations.tar.gz", "w:gz"
            ) as t,
            zipfile.ZipFile(self.root / "archives/CrowdPose_images.zip", "w") as z,
        ):
            for i, sp in enumerate(("train", "val")):
                kp = [[5.0, 6.0, 2.0]] * 13 + [[0.0, 0.0, 0.0]]
                obj = {
                    "images": [{"id": i, "file_name": f"{i}.png", "crowdIndex": 0.2}],
                    "annotations": [
                        {
                            "id": i,
                            "image_id": i,
                            "keypoints": functools.reduce(operator.iadd, kp, []),
                            "bbox": [0, 0, 10, 20],
                            "num_keypoints": 12,
                            "iscrowd": 0,
                        }
                    ],
                }
                raw = json.dumps(obj).encode()
                info = tarfile.TarInfo("crowdpose_" + sp + ".json")
                info.size = len(raw)
                t.addfile(info, io.BytesIO(raw))
                z.writestr(f"images/{i}.png", self.png(i))
        m = self.api.build("crowdpose", self.root)
        p = m["train"][0]["people"][0]
        self.assertEqual(p["crop"], [-7.5, -2.5, 25.0, 25.0])
        self.assertFalse(p["labeled"][-1])
        self.assertEqual(p["num_keypoints"], 12)

    def test_mpii_requires_pins_sidecar_join_and_image_partition(self):
        persons = []
        for i in range(2):
            persons.append(
                {
                    "id": str(i),
                    "image": f"{i}.png",
                    "validation": bool(i),
                    "joints": [[3.0, 4.0]] * 16,
                    "labeled": [True] * 16,
                    "crop": [0, 0, 20, 20],
                    "headbox": [0, 0, 10, 0],
                }
            )
            self.put(f"images/{i}.png", self.png(i))
        release = {
            "schema": "mpii-release-persons-v1",
            "persons": persons,
            "validation_persons": 1,
            "excluded_nonvalidation_persons_on_validation_images": [],
            "training_rule": "non-validation RELEASE training persons on images without a validation person",
            "mat_source_sha256": "a" * 64,
            "validation_source_sha256": "b" * 64,
        }
        raw = json.dumps(release).encode()
        self.put("archives/mpii_release_v1.json", raw)
        sidecar = {
            "schema": "mpii-hrnet-gt-val-v1",
            "source_sha256": {"release_json": hashlib.sha256(raw).hexdigest()},
            "columns": [
                {
                    "person_id": "1",
                    "pos_gt_src": [[4.0, 5.0]] * 16,
                    "jnt_missing": [0] * 16,
                    "headboxes_src": [[0, 0], [10, 0]],
                }
            ],
        }
        side = json.dumps(sidecar).encode()
        self.put("archives/gt.json", side)
        pins = {
            "release_sha256": hashlib.sha256(raw).hexdigest(),
            "gt_val_sidecar_sha256": hashlib.sha256(side).hexdigest(),
            "gt_val_sidecar_filename": "gt.json",
        }
        self.put("mpii_pins.json", json.dumps(pins))
        m = self.api.build("mpii_pose", self.root, fixture_counts=(1, 1))
        self.assertEqual(m["validation"][0]["people"][0]["pos_gt_src"][0], [4.0, 5.0])
        self.put("archives/gt.json", side + b" ")
        with self.assertRaises(ValueError):
            self.api.build("mpii_pose", self.root, fixture_counts=(1, 1))
