"""Behavioral checks for the manuscript's separate frontier evaluation."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from downstream import frontier

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "ImageNet-1k": "class_id",
    "SSv2": "class_id",
    "COCO": "category_ids",
    "ADE20K": "point_class_ids",
    "NYUv2": "pair_answers",
}


def fixture(dataset):
    sizes = {"ImageNet-1k": 1000, "SSv2": 174, "COCO": 80, "ADE20K": 150, "NYUv2": 0}
    samples, predictions = [], []
    for i in range(500):
        target = (
            [0, 1]
            if dataset == "COCO"
            else [0] * 32
            if dataset == "ADE20K"
            else [0] * 20
            if dataset == "NYUv2"
            else 0
        )
        sample = {"sample_id": f"sample-{i}", "target": target}
        if dataset == "SSv2":
            sample["frame_sha256"] = [f"{i * 16 + j:064x}" for j in range(16)]
        else:
            sample["input_sha256"] = f"{i:064x}"
        if dataset == "ADE20K":
            sample["points"] = [[j / 32, 0.5] for j in range(32)]
        if dataset == "NYUv2":
            sample["pairs"] = [[[j / 20, 0.25], [j / 20, 0.75]] for j in range(20)]
        samples.append(sample)
        predictions.append(
            {"sample_id": sample["sample_id"], FIELDS[dataset]: copy.deepcopy(target)}
        )
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "class_ids": list(range(sizes[dataset])),
        "samples": samples,
    }
    responses = [
        {"dataset": dataset, "predictions": predictions[start : start + 100]}
        for start in range(0, 500, 100)
    ]
    return manifest, responses


class FrontierEvaluation(unittest.TestCase):
    def test_all_five_perfect_scores_and_populations(self):
        for dataset in FIELDS:
            with self.subTest(dataset=dataset):
                m, r = fixture(dataset)
                result = frontier.evaluate(m, r)
                self.assertIsInstance(result, dict)
                self.assertEqual(result["score"], 100.0)
                self.assertEqual(result["samples"], 500)
                self.assertEqual(result["batches"], 5)
                self.assertFalse(result["paper_score_verified"])
                self.assertFalse(result["canonical_eligible"])
                self.assertFalse(result["record_value"])

    def test_top1_and_point_accuracy_use_all_decisions(self):
        for dataset, denominator in (
            ("ImageNet-1k", 500),
            ("SSv2", 500),
            ("ADE20K", 16000),
            ("NYUv2", 10000),
        ):
            with self.subTest(dataset=dataset):
                m, r = fixture(dataset)
                p = r[4]["predictions"][-1]
                if dataset in ("ImageNet-1k", "SSv2"):
                    p["class_id"] = 1
                else:
                    p[FIELDS[dataset]][-1] = 3 if dataset == "NYUv2" else 1
                result = frontier.evaluate(m, r)
                self.assertIsInstance(result, dict)
                self.assertEqual(result["correct"], denominator - 1)
                self.assertEqual(result["decisions"], denominator)
                self.assertAlmostEqual(
                    result["score"], 100 * (denominator - 1) / denominator
                )

    def test_coco_is_global_micro_f1_not_sample_f1(self):
        m, r = fixture("COCO")
        for s in m["samples"]:
            s["target"] = []
        for batch in r:
            for p in batch["predictions"]:
                p["category_ids"] = []
        m["samples"][0]["target"] = [0, 1]
        r[0]["predictions"][0]["category_ids"] = [0, 2]
        m["samples"][100]["target"] = [0]
        r[1]["predictions"][0]["category_ids"] = [0]
        result = frontier.evaluate(m, r)
        self.assertIsInstance(result, dict)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 1))
        self.assertAlmostEqual(result["score"], 200 / 3)
        m["samples"][0]["target"] = []
        m["samples"][100]["target"] = []
        for batch in r:
            for p in batch["predictions"]:
                p["category_ids"] = []
        with self.assertRaisesRegex(ValueError, "undefined"):
            frontier.evaluate(m, r)

    def test_rejects_incomplete_or_misaligned_responses(self):
        changes = [
            lambda m, r: r.pop(),
            lambda m, r: r[0]["predictions"].pop(),
            lambda m, r: r[0]["predictions"].reverse(),
            lambda m, r: r[0]["predictions"][0].update(sample_id="unknown"),
            lambda m, r: r[1].update(dataset="COCO"),
            lambda m, r: r[0]["predictions"][0].update(class_id=True),
            lambda m, r: r[0]["predictions"][0].update(class_id=1000),
            lambda m, r: r[0]["predictions"][0].update(class_id=0.0),
            lambda m, r: r[0]["predictions"][0].update(explanation="text"),
            lambda m, r: m["samples"].pop(),
            lambda m, r: m["samples"][1].update(sample_id="sample-0"),
            lambda m, r: m.update(schema_version=True),
            lambda m, r: m.update(dataset="unknown"),
            lambda m, r: m["class_ids"].pop(),
            lambda m, r: m["class_ids"].__setitem__(1, 0),
            lambda m, r: m["class_ids"].__setitem__(0, True),
            lambda m, r: m["samples"][0].update(target=1000),
            lambda m, r: m["samples"][0].update(input_sha256="invalid"),
            lambda m, r: m.update(extra=True),
        ]
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                m, r = fixture("ImageNet-1k")
                change(m, r)
                with self.assertRaises(ValueError):
                    frontier.evaluate(m, r)

    def test_dense_queries_and_category_sets_are_strict(self):
        cases = [
            ("COCO", lambda m, r: r[0]["predictions"][0].update(category_ids=[0, 0])),
            ("COCO", lambda m, r: m["samples"][0].update(target=[0, 0])),
            ("COCO", lambda m, r: r[0]["predictions"][0].update(category_ids=[80])),
            ("ADE20K", lambda m, r: r[0]["predictions"][0]["point_class_ids"].pop()),
            ("ADE20K", lambda m, r: m["samples"][0]["points"].pop()),
            ("ADE20K", lambda m, r: m["samples"][0]["points"][0].__setitem__(0, 1.1)),
            (
                "ADE20K",
                lambda m, r: m["samples"][0]["points"][0].__setitem__(0, float("nan")),
            ),
            ("ADE20K", lambda m, r: m["samples"][0]["points"][0].__setitem__(0, True)),
            ("NYUv2", lambda m, r: m["samples"][0]["pairs"].pop()),
            ("NYUv2", lambda m, r: m["samples"][0]["pairs"][0].pop()),
            ("NYUv2", lambda m, r: m["samples"][0]["target"].__setitem__(0, 3)),
            (
                "NYUv2",
                lambda m, r: r[0]["predictions"][0]["pair_answers"].__setitem__(0, 4),
            ),
            ("NYUv2", lambda m, r: m["samples"][0]["target"].pop()),
            ("SSv2", lambda m, r: m["samples"][0]["frame_sha256"].pop()),
        ]
        for dataset, change in cases:
            with self.subTest(dataset=dataset, change=change):
                m, r = fixture(dataset)
                change(m, r)
                with self.assertRaises(ValueError):
                    frontier.evaluate(m, r)

    def test_ids_need_not_be_zero_based_or_contiguous(self):
        m, r = fixture("COCO")
        m["class_ids"] = list(range(1, 81))
        for s in m["samples"]:
            s["target"] = [1, 80]
        for batch in r:
            for p in batch["predictions"]:
                p["category_ids"] = [80, 1]
        result = frontier.evaluate(m, r)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["score"], 100)

    def test_depth_abstentions_remain_in_denominator_and_equal_is_a_target(self):
        m, r = fixture("NYUv2")
        for sample in m["samples"]:
            sample["target"] = [0, 1, 2, 0, 1] * 4
        for batch in r:
            for p in batch["predictions"]:
                p["pair_answers"] = [0, 1, 2, 3, 1] * 4
        result = frontier.evaluate(m, r)
        self.assertEqual(result["correct"], 8000)
        self.assertEqual(result["decisions"], 10000)
        self.assertEqual(result["score"], 80)

    def test_manifest_and_response_fingerprints_preserve_order_and_queries(self):
        m, r = fixture("ADE20K")
        first = frontier.evaluate(m, r)
        m["samples"][0]["points"][0][0] = 1
        second = frontier.evaluate(m, r)
        self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(first["response_sha256"], second["response_sha256"])
        r[4]["predictions"][99]["point_class_ids"][31] = 1
        third = frontier.evaluate(m, r)
        self.assertNotEqual(second["response_sha256"][4], third["response_sha256"][4])
        self.assertEqual(second["response_sha256"][:4], third["response_sha256"][:4])

    def test_structural_boundary_types(self):
        changes = [
            lambda m, r: m["samples"].__setitem__(0, None),
            lambda m, r: m["samples"][0].update(sample_id=" "),
            lambda m, r: m["samples"][0].update(sample_id=0),
            lambda m, r: m.update(schema_version=2),
            lambda m, r: m.update(dataset=[]),
            lambda m, r: m.update(class_ids="invalid"),
            lambda m, r: m["class_ids"].__setitem__(0, -1),
            lambda m, r: m["samples"][0].update(input_sha256="A" * 64),
            lambda m, r: m["samples"][0].update(input_sha256=0),
            lambda m, r: r.__setitem__(0, {}),
            lambda m, r: r[0].update(predictions={}),
            lambda m, r: r[0]["predictions"].__setitem__(0, None),
        ]
        for change in changes:
            m, r = fixture("ImageNet-1k")
            change(m, r)
            with self.assertRaises(ValueError):
                frontier.evaluate(m, r)

    def test_invalid_manifest_cannot_be_validated_by_agreeing_responses(self):
        for identifier in ("sample-0", " "):
            m, r = fixture("ImageNet-1k")
            m["samples"][1]["sample_id"] = identifier
            r[0]["predictions"][1]["sample_id"] = identifier
            with self.assertRaises(ValueError):
                frontier.evaluate(m, r)
        for invalid in (-1, 999.5, True):
            m, r = fixture("ImageNet-1k")
            # Neither range membership nor uniqueness may mask the type guard.
            m["class_ids"][1] = 1001
            m["class_ids"][-1] = invalid
            with self.assertRaises(ValueError):
                frontier.evaluate(m, r)
        for dataset, key, value in (
            ("ADE20K", "point_class_ids", 0),
            ("NYUv2", "pair_answers", None),
            ("COCO", "category_ids", {}),
        ):
            m, r = fixture(dataset)
            r[0]["predictions"][0][key] = value
            with self.assertRaises(ValueError):
                frontier.evaluate(m, r)

    def test_cli_writes_report_without_overwriting_and_rejects_raw_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m, r = fixture("ImageNet-1k")
            (root / "manifest.json").write_text(json.dumps(m))
            paths = []
            for i, batch in enumerate(r):
                p = root / f"batch{i}.json"
                p.write_text(json.dumps(batch))
                paths.append(str(p))
            out = root / "report.json"
            cmd = [
                sys.executable,
                "-m",
                "downstream.frontier",
                "--manifest",
                str(root / "manifest.json"),
                "--responses",
                *paths,
                "--out",
                str(out),
            ]
            run = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, check=False, timeout=10, text=True
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(out.exists(), "CLI must publish an evaluation report")
            report = json.loads(out.read_text())
            self.assertEqual(report["score"], 100)
            import hashlib

            self.assertEqual(
                report["raw_manifest_sha256"],
                hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["raw_response_sha256"],
                [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths],
            )
            saved = out.read_bytes()
            self.assertNotEqual(
                subprocess.run(
                    cmd, cwd=ROOT, capture_output=True, check=False, timeout=10
                ).returncode,
                0,
            )
            self.assertEqual(out.read_bytes(), saved)
            for raw in (
                '{"dataset":"ImageNet-1k","dataset":"COCO"}',
                '{"dataset":NaN}',
                "```json\n{}\n```",
            ):
                Path(paths[0]).write_text(raw)
                failcmd = cmd[:-1] + [str(root / "failed.json")]
                self.assertNotEqual(
                    subprocess.run(
                        failcmd, cwd=ROOT, capture_output=True, check=False, timeout=10
                    ).returncode,
                    0,
                )
                self.assertFalse((root / "failed.json").exists())


if __name__ == "__main__":
    unittest.main()
