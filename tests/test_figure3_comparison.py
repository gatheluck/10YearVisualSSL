"""Compare full training outputs to the printed revised panel via its reporting CLI."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from tests import test_portable_figure3 as fixtures


class ComparisonTests(unittest.TestCase):
    root: Path
    setUp = fixtures.PortableFigure.setUp
    record = fixtures.PortableFigure.record

    def cell(
        self, task="coco", track="lora", seeds=(0, 1, 2), *, ranks=None, scores=None
    ):
        model = "raev2_k7"
        backbone = {
            "kind": model,
            "arch": "released",
            "img_size": 224,
            "patch_size": 16,
            "encoder": "weights",
        }

        def topology(result, published):
            if scores is not None:
                score = scores[int(result["metrics"]["bbox_ap"]) - 50]
                result["metrics"]["bbox_ap"] = score
                published["coco_map"] = score / 100
            result["world_size"] = (
                ranks if ranks is not None else (1 if task == "nyuv2" else 8)
            )

        records = [
            self.record(
                task, track, seed, alter=topology, config_extra={"backbone": backbone}
            )
            for seed in seeds
        ]
        return {"model": model, "task": task, "track": track, "runs": records}

    def report(self, cells):
        path = self.root / "plan.json"
        out = self.root / "comparison.json"
        path.write_text(
            json.dumps({"schema": "portable_figure3_comparison_v1", "cells": cells})
        )
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "downstream.revised_reporting",
                "--config",
                str(path),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertNotIn(str(self.root), out.read_text())
        report = json.loads(out.read_bytes())
        self.assertEqual(len(report["cells"]), 35)
        return {(r["model"], r["task"], r["track"]): r for r in report["cells"]}

    def test_bbox_and_unaligned_depth_in_printed_units(self):
        rows = self.report([self.cell(), self.cell("nyuv2", "dpt")])
        bbox = rows["raev2_k7", "coco", "lora"]
        self.assertEqual(bbox["observed"], {"mean": "51.00", "sd": "1.00"})
        self.assertEqual(bbox["expected"], {"mean": "47.37", "sd": "0.11"})
        self.assertEqual(bbox["status"], "mismatch")
        depth = rows["raev2_k7", "nyuv2", "dpt"]
        self.assertEqual(depth["observed"], {"mean": "0.410", "sd": "0.010"})
        self.assertEqual(depth["expected"], {"mean": "0.436", "sd": "0.002"})
        self.assertEqual(sum(r["status"] == "missing" for r in rows.values()), 33)

    def test_single_seed_does_not_publish_comparison(self):
        row = self.report([self.cell(seeds=(1,))])["raev2_k7", "coco", "lora"]
        self.assertEqual((row["status"], row["completed_seeds"]), ("incomplete", [1]))
        self.assertNotIn("observed", row)

    def test_wrong_rank_count_or_model_cannot_claim_paper_cell(self):
        wrong = self.cell("imagenet")
        wrong["model"] = "dinov3_hf"
        rows = self.report([self.cell(ranks=1), wrong])
        for k in [("raev2_k7", "coco", "lora"), ("dinov3_hf", "imagenet", "lora")]:
            self.assertEqual(rows[k]["status"], "invalid")
            self.assertNotIn("observed", rows[k])

    def test_match_requires_both_printed_statistics(self):
        row = self.report([self.cell(scores=[47.26, 47.37, 47.48])])[
            "raev2_k7", "coco", "lora"
        ]
        self.assertEqual(row["status"], "match")
        self.assertEqual(row["observed"], row["expected"])

    def test_matching_mean_with_different_sd_is_mismatch(self):
        row = self.report([self.cell(scores=[47.37] * 3)])["raev2_k7", "coco", "lora"]
        self.assertEqual(row["status"], "mismatch")
        self.assertEqual(row["observed"], {"mean": "47.37", "sd": "0.00"})

    def test_duplicate_cell_and_unknown_track_are_rejected(self):
        from downstream.figure3_comparison import compare_panel

        cell = self.cell(seeds=())
        for cells in [[cell, cell], [{**cell, "track": "attentive"}]]:
            with self.assertRaisesRegex(ValueError, "distinct cells"):
                compare_panel(cells)

    def test_documented_empty_campaign_can_be_reported(self):
        import re

        doc = Path(__file__).resolve().parents[1] / "docs/LORA_FULL_REPRODUCTION.md"
        plans = [
            json.loads(block)
            for block in re.findall(r"```json\n(.*?)\n```", doc.read_text(), re.DOTALL)
        ]
        (plan,) = [
            p for p in plans if p.get("schema") == "portable_figure3_comparison_v1"
        ]
        rows = self.report(plan["cells"])
        self.assertTrue(all(row["status"] == "missing" for row in rows.values()))
