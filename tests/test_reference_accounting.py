"""Historical results require explicit identities and final-epoch evidence."""

import copy
import hashlib
import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path


class Summaries(unittest.TestCase):
    def test_shared_statistics_keep_actual_n_and_missing_variance(self):
        from downstream import accounting

        self.assertTrue(
            hasattr(accounting, "summarize"), "shared measured-score summary missing"
        )
        self.assertEqual(
            accounting.summarize([0]), {"n": 1, "values": [0], "mean": 0, "std": None}
        )
        result = accounting.summarize([0, 2])
        self.assertEqual(result["mean"], 1)
        self.assertAlmostEqual(result["std"], math.sqrt(2))
        for values in ([], [True], [float("nan")], [float("inf")], ["1"]):
            with self.assertRaises(ValueError):
                accounting.summarize(values)


class ReferenceAccounting(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("downstream.reference_accounting"),
            "historical LP/AP audit missing",
        )
        self.api = importlib.import_module("downstream.reference_accounting")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0
        self.identity = {
            "dataset": "clue_footprint",
            "model": "raev2_k7",
            "track": "LINEAR",
            "protocol_id": "EXTEND_LINEAR_v1",
            "protocol_sha256": "a" * 64,
            "checkpoint_identity": {"sha256": "b" * 64},
            "recipe": "image_cls_100",
            "feature_layer": "K7 spatial mean",
            "evaluation_split": "test",
            "effective_batch": 256,
            "realized_lr": 0.1,
            "world_size": 1,
            "per_gpu_batch": 32,
            "accumulation": 8,
        }
        self.cell = {
            "cell_id": "table39-example",
            "identity": self.identity,
            "expected_seeds": [0, 1, 2],
            "epochs": 100,
            "evaluation_count": 2,
            "metric": {
                "field": "primary",
                "name": "top1",
                "unit": "percent",
                "direction": "higher",
            },
            "runs": [],
        }

    def artifact(self, name, obj):
        raw = json.dumps(obj, sort_keys=True).encode()
        path = self.root / name
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def run_record(self, seed, score, *, changes=None):
        self.serial += 1
        result = dict(
            self.identity,
            status="COMPLETED_VALID",
            seed=seed,
            protocol=self.identity["protocol_id"],
            epochs=100,
            final_scheduled_epoch=100,
            final_epoch_only=True,
            metrics={"primary": score, "n": 2},
            output_dir="/private/account/output",
            pbs_job_id="private-job",
        )
        result.update(changes or {})
        return {
            "seed": seed,
            "exit_status": 0,
            "result": self.artifact(f"result-{self.serial}.json", result),
            "completion": self.artifact(f"completed-{self.serial}.json", result),
            "resume_meta": self.artifact(
                f"resume-{self.serial}.json", {"epoch": 100, "step": 200}
            ),
        }

    def audit(self):
        return self.api.audit({"schema_version": 1, "cells": [self.cell]})["cells"][0]

    def test_actual_repeats_zero_scores_and_missing_runs_stay_distinct(self):
        self.cell["runs"] = [self.run_record(1, 2), self.run_record(0, 0)]
        result = self.audit()
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_seeds"], [2])
        self.assertEqual(result["completed_seeds"], [0, 1])
        self.assertEqual(result["summary"]["values"], [0, 2])
        self.assertEqual(
            result["valid_runs"][0]["artifact_sha256"],
            {
                k: self.cell["runs"][1][k]["sha256"]
                for k in ("result", "completion", "resume_meta")
            },
        )
        self.assertAlmostEqual(result["summary"]["std"], math.sqrt(2))
        self.assertFalse(result["canonical_eligible"])
        self.assertFalse(result["record_value"])
        self.assertFalse(result["paper_score_verified"])
        self.assertNotIn("/private/account", json.dumps(result))
        self.assertNotIn("private-job", json.dumps(result))
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_complete_lp_ap_cells_remain_separate_without_score_selection(self):
        self.cell["expected_seeds"] = [0]
        self.cell["runs"] = [self.run_record(0, 0)]
        first = copy.deepcopy(self.cell)
        self.identity.update(track="ATTENTIVE", protocol_id="EXTEND_ATTENTIVE_v1")
        self.cell["cell_id"] = "table40-example"
        self.cell["runs"] = [self.run_record(0, 20)]
        result = self.api.audit({"schema_version": 1, "cells": [first, self.cell]})
        self.assertEqual([c["summary"]["mean"] for c in result["cells"]], [0, 20])
        self.assertTrue(all(c["complete"] for c in result["cells"]))
        self.assertTrue(all(c["summary"]["std"] is None for c in result["cells"]))

    def test_invalid_identity_epoch_count_and_scores_are_not_averaged(self):
        changes = [
            {"status": "INCOMPLETE"},
            {"seed": 1},
            {"model": "different"},
            {"protocol": "wrong"},
            {"epochs": 99},
            {"final_scheduled_epoch": 99},
            {"final_epoch_only": False},
            {"metrics": {"primary": 0, "n": 1}},
            {"metrics": {"primary": True, "n": 2}},
            {"metrics": {"n": 2}},
            {"metrics": {"primary": 2, "n": True}},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.cell["runs"] = [self.run_record(0, 10, changes=change)]
                result = self.audit()
                self.assertFalse(result["complete"])
                self.assertEqual(result["completed_seeds"], [])
                self.assertEqual(result["invalid_runs"][0]["seed"], 0)
                self.assertIsNone(result["summary"])
                self.assertEqual(result["missing_seeds"], [1, 2])

    def test_each_pinned_identity_field_is_enforced(self):
        for key in self.identity:
            changed = {key: "different"}
            self.cell["runs"] = [self.run_record(0, 10, changes=changed)]
            with self.subTest(key=key):
                self.assertEqual(self.audit()["completed_seeds"], [])

    def test_final_epoch_metadata_and_completion_must_agree(self):
        for epoch in (99, 101, True):
            run = self.run_record(0, 10)
            run["resume_meta"] = self.artifact("resume.json", {"epoch": epoch})
            self.cell["runs"] = [run]
            self.assertEqual(self.audit()["completed_seeds"], [])
        run = self.run_record(0, 10)
        result = json.loads(Path(run["result"]["path"]).read_text())
        result["metrics"]["primary"] = 99
        run["completion"] = self.artifact("different.json", result)
        self.cell["runs"] = [run]
        self.assertEqual(self.audit()["completed_seeds"], [])

    def test_changed_missing_and_duplicate_json_artifacts_are_invalid(self):
        for kind in ("result", "completion", "resume_meta"):
            run = self.run_record(0, 10)
            Path(run[kind]["path"]).write_text("{}")
            self.cell["runs"] = [run]
            self.assertEqual(self.audit()["completed_seeds"], [])
            Path(run[kind]["path"]).unlink()
            self.assertEqual(self.audit()["completed_seeds"], [])
        for raw in (b'{"epoch":1,"epoch":100}', b'{"epoch":NaN}', b"[]"):
            run = self.run_record(0, 10)
            p = self.root / "malformed.json"
            p.write_bytes(raw)
            run["resume_meta"] = {
                "path": str(p),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            self.cell["runs"] = [run]
            self.assertEqual(self.audit()["completed_seeds"], [])

    def test_plan_mistakes_raise_instead_of_silently_dropping_runs(self):
        for seeds in ([], [0, 0], [True], [-1]):
            self.cell["expected_seeds"] = seeds
            with self.assertRaises(ValueError):
                self.audit()
        self.cell["expected_seeds"] = [0]
        for runs in ([self.run_record(1, 10)], [self.run_record(0, 10)] * 2):
            self.cell["runs"] = runs
            with self.assertRaises(ValueError):
                self.audit()
        self.cell["runs"] = []
        for key, value in (
            ("epochs", True),
            ("evaluation_count", 0),
            ("identity", {}),
            ("metric", {}),
        ):
            original = self.cell[key]
            self.cell[key] = value
            with self.assertRaises(ValueError):
                self.audit()
            self.cell[key] = original
        with self.assertRaises(ValueError):
            self.api.audit({"schema_version": 1, "cells": [self.cell, self.cell]})

    def test_cli_preserves_inputs_and_refuses_existing_output(self):
        self.cell["runs"] = [self.run_record(0, 5)]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"schema_version": 1, "cells": [self.cell]}))
        before = {p: p.read_bytes() for p in self.root.iterdir()}
        out = self.root / "report.json"
        self.assertEqual(self.api.main(["--config", str(plan), "--out", str(out)]), 0)
        self.assertFalse(json.loads(out.read_text())["cells"][0]["complete"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaises(FileExistsError):
            self.api.main(["--config", str(plan), "--out", str(out)])

    def test_report_pins_the_entire_plan_and_checked_population(self):
        plan = {"schema_version": 1, "cells": [self.cell]}
        report = self.api.audit(plan)
        digest = hashlib.sha256(
            json.dumps(
                plan, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self.assertEqual(report.get("plan_sha256"), digest)
        self.assertEqual(report["cells"][0].get("epochs"), 100)
        self.assertEqual(report["cells"][0].get("evaluation_count"), 2)
        self.assertEqual(
            report["cells"][0]["identity_sha256"],
            hashlib.sha256(self.api.canonical(self.identity)).hexdigest(),
        )

    def test_unsuccessful_or_unknown_exit_is_not_a_completed_experiment(self):
        for code in (1, None, True):
            run = self.run_record(0, 10)
            run["exit_status"] = code
            self.cell["runs"] = [run]
            result = self.audit()
            self.assertEqual(result["completed_seeds"], [])
            self.assertEqual(
                result["invalid_runs"][0]["reason"], "recorded successful exit required"
            )

    def test_plan_schema_and_identity_boundaries(self):
        for plan in (
            {},
            {"schema_version": True, "cells": [self.cell]},
            {"schema_version": 1, "cells": []},
        ):
            with self.assertRaises(ValueError):
                self.api.audit(plan)
        for key, value in (
            ("track", "FINETUNE"),
            ("recipe", None),
            ("checkpoint_identity", {}),
            ("protocol_sha256", "bad"),
        ):
            cell = copy.deepcopy(self.cell)
            cell["identity"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_cell(cell)
        for key, value in (
            ("cell_id", " "),
            ("runs", None),
            ("runs", {}),
            (
                "metric",
                {"field": "primary", "name": "x", "unit": "u", "direction": "unknown"},
            ),
        ):
            cell = copy.deepcopy(self.cell)
            cell[key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                self.api.validate_cell(cell)
        cell = copy.deepcopy(self.cell)
        cell["extra"] = 1
        with self.assertRaises(ValueError):
            self.api.validate_cell(cell)
        self.cell["runs"] = [{"seed": 0}]
        with self.assertRaises(ValueError):
            self.api.validate_cell(self.cell)

    def test_pinned_descriptor_and_unused_nonfinite_field(self):
        run = self.run_record(0, 10)
        original = json.loads(Path(run["resume_meta"]["path"]).read_text())
        original["unused"] = "changed without altering epoch"
        Path(run["resume_meta"]["path"]).write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.api.pinned(run["resume_meta"])
        for descriptor in ({}, {"path": run["result"]["path"], "sha256": "bad"}):
            with self.assertRaises(ValueError):
                self.api.pinned(descriptor)
        raw = b'{"epoch":100,"unused":NaN}'
        p = self.root / "bad-meta.json"
        p.write_bytes(raw)
        run["resume_meta"] = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()}
        self.cell["runs"] = [run]
        self.assertEqual(self.audit()["completed_seeds"], [])

    def test_metric_selection_is_explicit_and_does_not_convert_units(self):
        self.cell["metric"] = {
            "field": "rmse",
            "name": "depth_rmse",
            "unit": "m",
            "direction": "lower",
        }
        self.cell["runs"] = [
            self.run_record(
                0, 0, changes={"metrics": {"primary": 99, "rmse": 0.25, "n": 2}}
            )
        ]
        result = self.audit()
        self.assertEqual(result["summary"]["mean"], 0.25)
        self.assertEqual(result["metric"], self.cell["metric"])

    def test_recorded_blank_video_inputs_are_not_valid_measurements(self):
        for count in (1, -1, True, "0"):
            self.cell["runs"] = [
                self.run_record(
                    0,
                    10,
                    changes={
                        "metrics": {"primary": 10, "n": 2, "n_all_zero_clips": count}
                    },
                )
            ]
            result = self.audit()
            self.assertEqual(result["completed_seeds"], [])
            self.assertEqual(
                result["invalid_runs"][0]["reason"], "blank input disclosure is invalid"
            )
        self.cell["runs"] = [
            self.run_record(
                0,
                10,
                changes={"metrics": {"primary": 10, "n": 2, "n_all_zero_clips": 0}},
            )
        ]
        self.assertEqual(self.audit()["completed_seeds"], [0])


class Delivery(unittest.TestCase):
    def test_documented_example_is_a_valid_missing_run_audit(self):
        root = Path(__file__).resolve().parents[1]
        guide = root / "docs/REFERENCE_ACCOUNTING.md"
        self.assertTrue(guide.is_file(), "historical audit guide missing")
        path = root / "docs/examples/reference_accounting.json"
        self.assertTrue(path.is_file(), "historical audit plan missing")
        from downstream.reference_accounting import audit

        report = audit(json.loads(path.read_text()))
        self.assertFalse(report["cells"][0]["complete"])
        self.assertEqual(report["cells"][0]["missing_seeds"], [0, 1, 2])
