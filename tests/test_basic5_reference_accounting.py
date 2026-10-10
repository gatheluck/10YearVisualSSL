"""BasicFive native schemas must not masquerade as completed paper repeats."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from downstream import reference_accounting as api


class BasicFiveAccounting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0

    def cell(self, schema="basic5_epoch_v1", track="FINETUNE"):
        identity = {
            "dataset": "imagenet1k",
            "method_id": "example_encoder",
            "protocol_id": "BASIC5_FAIR_v1"
            if track == "LINEAR"
            else f"BASIC5_{track}_v1",
            "checkpoint_identity": {"sha256": "b" * 64},
            "feature_layer": "inspected_readout",
            "effective_batch": 1024,
            "realized_lr": 0.001,
            "world_size": 8,
        }
        if schema == "basic5_epoch_v1":
            identity.update(
                track=track,
                campaign="example_campaign",
                protocol_sha256="a" * 64,
                input_size=224,
                eval_views=1,
                per_gpu_batch=32,
                accumulation=4,
            )
        else:
            identity.update(
                protocol_hash="a" * 64,
                input_resolution=224,
                evaluation_views=1,
                micro_batch_per_gpu=32,
                grad_accum=4,
            )
        return {
            "cell_id": f"{schema}-{track}",
            "source_schema": schema,
            "track": track,
            "identity": identity,
            "expected_seeds": [0, 1, 2],
            "epochs": 100,
            "evaluation_count": 50000,
            "metric": {
                "field": "top1",
                "name": "top1",
                "unit": "percent",
                "direction": "higher",
            },
            "runs": [],
        }

    def artifact(self, value):
        self.serial += 1
        raw = json.dumps(value).encode()
        path = self.root / f"artifact-{self.serial}.json"
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def record(self, cell, seed=0, score=0, **changes):
        result = dict(
            cell["identity"],
            status="COMPLETED_VALID",
            seed=seed,
            smoke=False,
            metrics={"top1": score, "n": 50000},
            output_dir="/private/team/run",
            job_id="private-job",
        )
        meta = {"epoch": 100, "step": 1000}
        if cell["source_schema"] == "basic5_epoch_v1":
            result.update(epochs=100, final_epoch_only=True)
        else:
            result.update(
                canonical=True,
                epochs_scheduled=100,
                epochs_run=100,
                final_epoch_canonical=True,
            )
            meta.update(
                protocol=cell["identity"]["protocol_id"],
                dataset="imagenet1k",
                seed=seed,
            )
        result.update(changes)
        run = {
            "seed": seed,
            "exit_status": 0,
            "result": self.artifact(result),
            "resume_meta": self.artifact(meta),
        }
        if cell["source_schema"] == "basic5_epoch_v1":
            run["completion"] = self.artifact(result)
        return run

    def audit(self, cell):
        return api.audit({"schema_version": 1, "cells": [cell]})["cells"][0]

    def test_both_native_schemas_cover_lp_ap_ft_and_preserve_repeat_accounting(self):
        cells = []
        for schema in ("basic5_epoch_v1", "basic5_scheduled_v1"):
            for track in ("LINEAR", "ATTENTIVE", "FINETUNE"):
                cell = self.cell(schema, track)
                cell["runs"] = [self.record(cell, 1, 2), self.record(cell, 0, 0)]
                cells.append(cell)
        results = api.audit({"schema_version": 1, "cells": cells})["cells"]
        self.assertEqual(len(results), 6)
        for expected, result in zip(cells, results):
            self.assertEqual(result["source_schema"], expected["source_schema"])
            self.assertEqual(result["track"], expected["track"])
            self.assertEqual(result["completed_seeds"], [0, 1])
            self.assertEqual(result["missing_seeds"], [2])
            self.assertEqual(result["summary"]["values"], [0, 2])
            self.assertAlmostEqual(result["summary"]["std"], math.sqrt(2))
            self.assertFalse(result["complete"])
            self.assertFalse(result["paper_score_verified"])
            self.assertFalse(result["record_value"])
            self.assertFalse(result["canonical_eligible"])
            self.assertEqual(
                set(result["valid_runs"][0]["artifact_sha256"]),
                set(expected["runs"][0]) - {"seed", "exit_status"},
            )
            for secret in ("/private/team", "private-job", str(self.root)):
                self.assertNotIn(secret, json.dumps(result))

    def test_native_population_type_is_explicit_not_rounded_or_guessed(self):
        for schema in ("basic5_epoch_v1", "basic5_scheduled_v1"):
            cell = self.cell(schema)
            for n in (50000, 50000.0, 49999, 50000.5, True, None):
                with self.subTest(schema=schema, n=n):
                    cell["runs"] = [self.record(cell, metrics={"top1": 1, "n": n})]
                    expected = n == 50000 and (
                        type(n) is int or schema == "basic5_scheduled_v1"
                    )
                    self.assertEqual(
                        self.audit(cell)["completed_seeds"], [0] if expected else []
                    )
            cell["runs"] = [self.record(cell, metrics={"top1": 1})]
            self.assertIsNone(self.audit(cell)["summary"])

    def test_nonfinal_smoke_partial_and_noncanonical_records_are_invalid(self):
        for schema in ("basic5_epoch_v1", "basic5_scheduled_v1"):
            cell = self.cell(schema)
            changes = [
                {"smoke": True},
                {"smoke": 0},
                {"smoke": None},
                {"canonical": False},
                {"canonical": 1},
                {"status": "PARTIAL"},
                {"status": "REUSED"},
                {"status": "SMOKE_OK"},
            ]
            if schema == "basic5_epoch_v1":
                changes += [
                    {"epochs": 99},
                    {"epochs": True},
                    {"final_epoch_only": False},
                ]
            else:
                changes += [
                    {"epochs_scheduled": 99},
                    {"epochs_run": 99},
                    {"final_epoch_canonical": False},
                    {"canonical": None},
                ]
            for change in changes:
                with self.subTest(schema=schema, change=change):
                    cell["runs"] = [self.record(cell, **change)]
                    self.assertIsNone(self.audit(cell)["summary"])

    def test_all_identity_fields_and_scheduled_metadata_are_bound(self):
        for schema in ("basic5_epoch_v1", "basic5_scheduled_v1"):
            cell = self.cell(schema)
            for key in cell["identity"]:
                with self.subTest(schema=schema, key=key):
                    cell["runs"] = [self.record(cell, **{key: "different"})]
                    self.assertEqual(self.audit(cell)["completed_seeds"], [])
            for key, value in [("epoch", 99), ("epoch", True)] + (
                [
                    ("protocol", "other"),
                    ("dataset", "other"),
                    ("seed", 1),
                    ("seed", True),
                ]
                if schema == "basic5_scheduled_v1"
                else []
            ):
                run = self.record(cell)
                meta = json.loads(Path(run["resume_meta"]["path"]).read_text())
                meta[key] = value
                run["resume_meta"] = self.artifact(meta)
                cell["runs"] = [run]
                with self.subTest(schema=schema, meta=key, value=value):
                    self.assertEqual(self.audit(cell)["completed_seeds"], [])

    def test_bad_evidence_is_excluded_without_leaking_paths(self):
        for schema in ("basic5_epoch_v1", "basic5_scheduled_v1"):
            cell = self.cell(schema)
            for artifact in set(self.record(cell)) - {"seed", "exit_status"}:
                run = self.record(cell)
                Path(run[artifact]["path"]).write_text("{}")
                cell["runs"] = [run]
                self.assertEqual(self.audit(cell)["completed_seeds"], [])
            for exit_status in (None, 1, True):
                run = self.record(cell)
                run["exit_status"] = exit_status
                cell["runs"] = [run]
                self.assertEqual(self.audit(cell)["completed_seeds"], [])
            for score in (True, None, "1", float("nan")):
                cell["runs"] = [self.record(cell, score=score)]
                self.assertIsNone(self.audit(cell)["summary"])
        cell = self.cell()
        run = self.record(cell)
        run["completion"] = self.artifact({"status": "COMPLETED_VALID"})
        cell["runs"] = [run]
        self.assertIsNone(self.audit(cell)["summary"])

    def test_missing_scheduled_canonical_flag_and_metadata_are_not_inferred(self):
        cell = self.cell("basic5_scheduled_v1")
        for artifact, key in (
            ("result", "canonical"),
            ("resume_meta", "seed"),
            ("resume_meta", "protocol"),
            ("resume_meta", "dataset"),
        ):
            run = self.record(cell)
            value = json.loads(Path(run[artifact]["path"]).read_text())
            del value[key]
            run[artifact] = self.artifact(value)
            cell["runs"] = [run]
            with self.subTest(artifact=artifact, key=key):
                self.assertIsNone(self.audit(cell)["summary"])

    def test_linear_filename_label_is_not_the_recorded_protocol_identity(self):
        cell = self.cell(track="LINEAR")
        cell["identity"]["protocol_id"] = "BASIC5_LINEAR_v1"
        with self.assertRaises(ValueError):
            self.audit(cell)
        cell = self.cell()
        cell["identity"]["track"] = "LINEAR"
        with self.assertRaises(ValueError):
            self.audit(cell)

    def test_plan_requires_recognized_schema_track_identity_and_artifacts(self):
        for bad in (None, [], {}, "cell"):
            with self.subTest(cell=bad), self.assertRaises(ValueError):
                api.validate_cell(bad)
        for schema in ("basic5_epoch_v1", "basic5_scheduled_v1"):
            cell = self.cell(schema)
            for key in cell["identity"]:
                bad = copy.deepcopy(cell)
                del bad["identity"][key]
                with (
                    self.subTest(schema=schema, missing=key),
                    self.assertRaises(ValueError),
                ):
                    self.audit(bad)
            for key, value in [
                ("source_schema", "unknown"),
                ("source_schema", None),
                ("track", "OTHER"),
                ("track", "LINEAR"),
            ]:
                bad = copy.deepcopy(cell)
                bad[key] = value
                with (
                    self.subTest(schema=schema, key=key),
                    self.assertRaises(ValueError),
                ):
                    self.audit(bad)
            for field in (
                "protocol_sha256" if schema == "basic5_epoch_v1" else "protocol_hash",
            ):
                bad = copy.deepcopy(cell)
                bad["identity"][field] = "bad"
                with self.assertRaises(ValueError):
                    self.audit(bad)
            run = self.record(cell)
            del run["resume_meta"]
            cell["runs"] = [run]
            with self.assertRaises(ValueError):
                self.audit(cell)

    def test_cli_handles_mixed_schemas_and_preserves_originals(self):
        cells = [self.cell(), self.cell("basic5_scheduled_v1")]
        for cell in cells:
            cell["expected_seeds"] = [0]
            cell["runs"] = [self.record(cell)]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"schema_version": 1, "cells": cells}))
        before = {p: p.read_bytes() for p in self.root.iterdir()}
        out = self.root / "report.json"
        self.assertEqual(api.main(["--config", str(plan), "--out", str(out)]), 0)
        report = json.loads(out.read_text())
        self.assertTrue(
            all(c["complete"] and c["summary"]["std"] is None for c in report["cells"])
        )
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaises(FileExistsError):
            api.main(["--config", str(plan), "--out", str(out)])

    def test_documented_empty_plan_reports_all_missing_without_scores(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/basic5_reference_accounting.json"
        )
        report = api.audit(api.read_json(path.read_bytes()))
        self.assertEqual(
            [c["track"] for c in report["cells"]], ["LINEAR", "ATTENTIVE", "FINETUNE"]
        )
        self.assertEqual(len(report["cells"]), 3)
        for cell in report["cells"]:
            self.assertEqual(cell["missing_seeds"], [0, 1, 2])
            self.assertIsNone(cell["summary"])
            self.assertFalse(cell["complete"])
