#!/usr/bin/env python3
"""Specification for contract-test.py.

**This is how "the port is finished" gets decided by a machine rather than by
opinion.** The contract is defined in the Capture repository's
`docs/CONTRACT.md`, which is the single source of truth.

What this tool holds:

- **Success needs two signals to agree:** exit status 0 and a well-formed
  manifest saying `status: "ok"`. Neither is trusted alone; on the Capture
  side a gate once returned exit 0 while reporting detected secrets
- **No file may be absent from the manifest.** An output nobody knows about
  is a hole in reproducibility, the same reasoning as the capture index
- **Nothing passes quietly.** If something is missing, it is named
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BIN = Path(__file__).resolve().parent.parent / "bin"


def load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = load("contract_test", "contract-test.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cttest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.config = self.tmp / "resolved.json"
        self.config.write_text('{"seed":0}\n', encoding="utf-8")

    def write_out(self, rel: str, data: bytes) -> dict:
        p = self.out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return {"path": rel, "sha256": sha256(data), "bytes": len(data)}

    def make_run(self, *, status: str = "ok", extra_files: dict | None = None,
                 drop: tuple[str, ...] = (), **over) -> None:
        """Build a conforming output; `over` breaks exactly one thing."""
        arts = []
        if "encoder.pt" not in drop:
            a = self.write_out("encoder.pt", b"weights")
            a["role"] = "encoder"
            arts.append(a)
        if "metrics.json" not in drop:
            body = json.dumps({
                "schema_version": 2,
                "metrics": {"final_linear_probe_top1_accuracy": 42.5},
                "metrics_raw": {"top1": 42.5}}).encode()
            a = self.write_out("metrics.json", body)
            a["role"] = "metrics"
            arts.append(a)
        for rel, data in (extra_files or {}).items():
            self.write_out(rel, data)
        man = {
            # A placeholder, not a real method: this file is about the
            # contract, and naming a method here is the start of machinery
            # that only works for one.
            "schema_version": 1, "method": "a_method",
            "stage": "pretrain", "status": status,
            "config_sha256": sha256(self.config.read_bytes()),
            "started_at": "2026-07-29T00:00:00Z",
            "finished_at": "2026-07-29T01:00:00Z",
            "seed": 0, "world_size": 1,
            "env": {"python": "3.10.13", "torch": "2.0.1"},
            "upstream": None, "artifacts": arts,
        }
        man.update(over)
        if "run_manifest.json" not in drop:
            (self.out / "run_manifest.json").write_text(
                json.dumps(man, ensure_ascii=False), encoding="utf-8")

    def check(self, **kw):
        kw.setdefault("out", self.out)
        kw.setdefault("config", self.config)
        return ct.check(**kw)


class TestHappyPath(Base):
    def test_conforming_run_passes(self):
        self.make_run()
        rc, rep = self.check()
        self.assertEqual(rc, 0, rep["violations"])
        self.assertEqual(rep["violations"], [])

    def test_exit_status_zero_agrees_with_ok(self):
        self.make_run()
        self.assertEqual(self.check(exit_status=0)[0], 0)


class TestRequiredFiles(Base):
    def test_missing_manifest_fails(self):
        self.make_run(drop=("run_manifest.json",))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("manifest-missing", [v["kind"] for v in rep["violations"]])

    def test_missing_encoder_fails(self):
        self.make_run(drop=("encoder.pt",))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("encoder-missing", [v["kind"] for v in rep["violations"]])

    def test_missing_metrics_fails(self):
        self.make_run(drop=("metrics.json",))
        self.assertEqual(self.check()[0], 1)

    def test_unparsable_manifest_fails(self):
        self.make_run()
        (self.out / "run_manifest.json").write_text("{ not json")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("manifest-unparsable",
                      [v["kind"] for v in rep["violations"]])


class TestManifestFields(Base):
    def test_every_required_field_is_checked(self):
        """Drop each field in turn and confirm each one is caught.

        A single test for "some required field is missing" would pass even
        if only one field were ever inspected.
        """
        required = ("schema_version", "method", "stage", "status",
                    "config_sha256", "started_at", "finished_at",
                    "seed", "env", "artifacts")
        for field in required:
            with self.subTest(field=field):
                shutil.rmtree(self.out); self.out.mkdir()
                self.make_run()
                man = json.loads((self.out / "run_manifest.json").read_text())
                del man[field]
                (self.out / "run_manifest.json").write_text(json.dumps(man))
                rc, rep = self.check()
                self.assertEqual(rc, 1, f"passes with {field} removed")
                self.assertTrue(
                    any(field in v.get("detail", "") for v in rep["violations"]),
                    f"the missing {field} is not named: {rep['violations']}")

    def test_finished_before_started_fails(self):
        self.make_run(started_at="2026-07-29T02:00:00Z",
                      finished_at="2026-07-29T01:00:00Z")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("time-order", [v["kind"] for v in rep["violations"]])


class TestConfigIsTheOneThatRan(Base):
    """Show that the config handed in is the config that ran."""

    def test_mismatched_config_sha_fails(self):
        self.make_run(config_sha256="0" * 64)
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("config-mismatch", [v["kind"] for v in rep["violations"]])

    def test_config_changed_after_the_run_is_detected(self):
        self.make_run()
        self.config.write_text('{"seed":999}\n', encoding="utf-8")
        self.assertEqual(self.check()[0], 1)


class TestArtifacts(Base):
    def test_listed_artifact_that_is_absent_fails(self):
        self.make_run()
        man = json.loads((self.out / "run_manifest.json").read_text())
        man["artifacts"].append({"path": "ghost.bin", "role": "extra",
                                 "sha256": "0" * 64, "bytes": 1})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("artifact-missing", [v["kind"] for v in rep["violations"]])

    def test_tampered_artifact_fails(self):
        self.make_run()
        (self.out / "encoder.pt").write_bytes(b"tampered")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("artifact-sha256", [v["kind"] for v in rep["violations"]])

    def test_wrong_byte_count_fails(self):
        self.make_run()
        man = json.loads((self.out / "run_manifest.json").read_text())
        man["artifacts"][0]["bytes"] = 999999
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        self.assertEqual(self.check()[0], 1)

    def test_encoder_role_must_be_registered(self):
        self.make_run()
        man = json.loads((self.out / "run_manifest.json").read_text())
        for a in man["artifacts"]:
            if a["path"] == "encoder.pt":
                a["role"] = "something-else"
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("encoder-role", [v["kind"] for v in rep["violations"]])


class TestAnEncoderThatCannotBeProduced(Base):
    """CONTRACT section 3: some methods produce no encoder.

    Saying so is allowed. Not producing one quietly is not. Before this, the
    tool refused every absent encoder, which contradicted the contract it is
    supposed to enforce.
    """

    def test_absent_with_a_recorded_reason_passes(self):
        self.make_run(drop=("encoder.pt",),
                      encoder_absent_reason="this stage trains no encoder")
        rc, rep = self.check()
        self.assertEqual(rc, 0, rep["violations"])

    def test_absent_with_no_reason_still_fails(self):
        self.make_run(drop=("encoder.pt",))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("encoder-missing", [v["kind"] for v in rep["violations"]])

    def test_an_empty_reason_does_not_count(self):
        """A blank string is not an explanation."""
        self.make_run(drop=("encoder.pt",), encoder_absent_reason="")
        self.assertEqual(self.check()[0], 1)

    def test_a_reason_alongside_a_real_encoder_is_refused(self):
        """The two statements contradict each other."""
        self.make_run(encoder_absent_reason="none produced")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("encoder-contradiction",
                      [v["kind"] for v in rep["violations"]])


class TestAFailedRunIsJudgedAsAFailedRun(Base):
    """CONTRACT section 4: exit != 0 with `failed` is a correctly reported
    failure, not a contract violation.

    A run that died before saving anything has no encoder and no metrics. That
    is what dying means; reporting it as a breach of contract buries the real
    reason under noise.
    """

    def test_missing_outputs_are_not_violations_when_the_run_failed(self):
        self.make_run(status="failed", drop=("encoder.pt", "metrics.json"))
        rc, rep = self.check(exit_status=1)
        self.assertEqual(rep["violations"], [])
        self.assertEqual(rc, 1)
        self.assertEqual(rep["status"], "failed")

    def test_integrity_is_still_checked_on_a_failed_run(self):
        """Whatever it *did* write must still be described correctly."""
        self.make_run(status="failed", drop=("encoder.pt",))
        (self.out / "metrics.json").write_bytes(b"tampered")
        rc, rep = self.check(exit_status=1)
        self.assertEqual(rc, 1)
        self.assertIn("artifact-sha256", [v["kind"] for v in rep["violations"]])

    def test_unlisted_files_are_still_caught_on_a_failed_run(self):
        self.make_run(status="failed", drop=("encoder.pt",),
                      extra_files={"stray.log": b"x"})
        rc, rep = self.check(exit_status=1)
        self.assertIn("unlisted-file", [v["kind"] for v in rep["violations"]])

    def test_a_successful_run_still_needs_its_outputs(self):
        """The relaxation must not leak into the success path."""
        self.make_run(status="ok", drop=("encoder.pt", "metrics.json"))
        rc, rep = self.check(exit_status=0)
        self.assertEqual(rc, 1)
        kinds = [v["kind"] for v in rep["violations"]]
        self.assertIn("encoder-missing", kinds)
        self.assertIn("metrics-missing", kinds)


class TestNoUnlistedFiles(Base):
    """An output nobody knows about is a hole. Same idea as the capture index."""

    def test_unlisted_file_fails(self):
        self.make_run(extra_files={"stray.log": b"who wrote me"})
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("unlisted-file", [v["kind"] for v in rep["violations"]])
        self.assertIn("stray.log", " ".join(
            v.get("detail", "") for v in rep["violations"]))

    def test_unlisted_file_in_a_subdirectory_fails(self):
        self.make_run(extra_files={"logs/train.log": b"x"})
        self.assertEqual(self.check()[0], 1)

    def test_the_manifest_itself_is_not_unlisted(self):
        """The manifest cannot contain its own hash; it is the one exception."""
        self.make_run()
        rc, rep = self.check()
        self.assertEqual(rc, 0, rep["violations"])


class TestMetrics(Base):
    def test_unparsable_metrics_fails(self):
        self.make_run()
        body = b"{ not json"
        a = self.write_out("metrics.json", body)
        man = json.loads((self.out / "run_manifest.json").read_text())
        for x in man["artifacts"]:
            if x["path"] == "metrics.json":
                x.update({"sha256": a["sha256"], "bytes": a["bytes"]})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("metrics-unparsable",
                      [v["kind"] for v in rep["violations"]])

    def test_non_numeric_metric_fails(self):
        """The string "42.3" will not do; nothing can compare it."""
        self.make_run()
        body = json.dumps({"schema_version": 1,
                           "metrics": {"top1": "42.3"}}).encode()
        a = self.write_out("metrics.json", body)
        man = json.loads((self.out / "run_manifest.json").read_text())
        for x in man["artifacts"]:
            if x["path"] == "metrics.json":
                x.update({"sha256": a["sha256"], "bytes": a["bytes"]})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("metrics-not-numeric",
                      [v["kind"] for v in rep["violations"]])

    def test_boolean_is_not_accepted_as_a_number(self):
        """In Python bool subclasses int. Do not let it through."""
        self.make_run()
        body = json.dumps({"schema_version": 1,
                           "metrics": {"converged": True}}).encode()
        a = self.write_out("metrics.json", body)
        man = json.loads((self.out / "run_manifest.json").read_text())
        for x in man["artifacts"]:
            if x["path"] == "metrics.json":
                x.update({"sha256": a["sha256"], "bytes": a["bytes"]})
        (self.out / "run_manifest.json").write_text(json.dumps(man))
        self.assertEqual(self.check()[0], 1)


class TestTwoSignalsMustAgree(Base):
    """Success is exit status 0 *and* status ok. Neither alone."""

    def test_exit_nonzero_with_ok_status_fails(self):
        self.make_run(status="ok")
        rc, rep = self.check(exit_status=1)
        self.assertEqual(rc, 1)
        self.assertIn("status-disagreement",
                      [v["kind"] for v in rep["violations"]])

    def test_exit_zero_with_failed_status_fails(self):
        self.make_run(status="failed")
        rc, rep = self.check(exit_status=0)
        self.assertEqual(rc, 1)
        self.assertIn("status-disagreement",
                      [v["kind"] for v in rep["violations"]])

    def test_failed_status_with_nonzero_exit_is_a_reported_failure(self):
        """A failure, correctly reported. Not a violation, and not a success."""
        self.make_run(status="failed")
        rc, rep = self.check(exit_status=1)
        self.assertEqual(rc, 1)
        self.assertEqual([v["kind"] for v in rep["violations"]], [])
        self.assertEqual(rep["status"], "failed")

    def test_unknown_status_is_refused(self):
        self.make_run(status="probably-fine")
        rc, rep = self.check()
        self.assertEqual(rc, 1)
        self.assertIn("status-unknown", [v["kind"] for v in rep["violations"]])


class TestTheMetricVocabularyIsEnforced(Base):
    """The contract fixes the names, so the checker has to know them.

    A vocabulary that only the writing side enforces is a convention. A run
    arriving from somewhere else -- an older adapter, a hand-written output,
    a port that skipped `adapterlib` -- would sail through with whatever
    names it liked, and the comparison built on top would be wrong in a way
    no file records. This project has already shipped a contract clause that
    was never implemented because the field name was left unsaid.

    The checker reads the vocabulary from `adapterlib` rather than keeping a
    copy. Two copies of one rule is the divergence that broke the container
    jobs.
    """

    def metrics(self, doc: dict) -> None:
        """Replace metrics.json with `doc`, keeping the manifest honest."""
        self.make_run()
        body = json.dumps(doc).encode()
        a = self.write_out("metrics.json", body)
        a["role"] = "metrics"
        man = json.loads((self.out / "run_manifest.json").read_text())
        man["artifacts"] = [x for x in man["artifacts"]
                            if x["path"] != "metrics.json"] + [a]
        (self.out / "run_manifest.json").write_text(json.dumps(man),
                                                    encoding="utf-8")

    def kinds(self) -> list[str]:
        return [x["kind"] for x in self.check()[1]["violations"]]

    def test_a_conforming_metrics_file_passes(self):
        """The negative control. Without it every check below could be
        passing because the checker rejects everything."""
        self.metrics({"schema_version": 2,
                      "metrics": {"final_pretext_loss": 1.5},
                      "metrics_raw": {"val_loss": 1.5}})
        rc, rep = self.check()
        self.assertEqual(rc, 0, rep["violations"])

    def test_a_name_outside_the_vocabulary_is_rejected(self):
        self.metrics({"schema_version": 2,
                      "metrics": {"top1": 42.5},
                      "metrics_raw": {"top1": 42.5}})
        rc, rep = self.check()
        self.assertNotEqual(rc, 0)
        self.assertIn("metrics-unknown-name", self.kinds())
        self.assertIn("top1", json.dumps(rep["violations"]))

    def test_the_original_names_must_be_there(self):
        """Required by the contract: dropping them loses what the original
        called its own numbers, and nothing would say so."""
        self.metrics({"schema_version": 2,
                      "metrics": {"final_pretext_loss": 1.5}})
        self.assertNotEqual(self.check()[0], 0)
        self.assertIn("metrics-raw-missing", self.kinds())

    def test_the_original_names_must_be_an_object(self):
        self.metrics({"schema_version": 2,
                      "metrics": {"final_pretext_loss": 1.5},
                      "metrics_raw": []})
        self.assertIn("metrics-raw-missing", self.kinds())

    def test_the_original_values_must_be_numbers_too(self):
        """Both blocks, or the unchecked one becomes the place to hide."""
        self.metrics({"schema_version": 2,
                      "metrics": {"final_pretext_loss": 1.5},
                      "metrics_raw": {"val_loss": "1.5"}})
        self.assertIn("metrics-not-numeric", self.kinds())

    def test_the_old_shape_is_rejected(self):
        """Schema 1 has no vocabulary and no original names. Accepting it
        would leave both new rules optional in practice."""
        self.metrics({"schema_version": 1, "metrics": {"top1": 42.5}})
        self.assertNotEqual(self.check()[0], 0)
        self.assertIn("metrics-schema", self.kinds())

    def test_the_checker_reads_the_one_vocabulary(self):
        """Not a copy of it. A copy diverges, and the two agree right up to
        the case that matters."""
        src = (BIN / "contract-test.py").read_text(encoding="utf-8")
        self.assertIn("METRIC_VOCABULARY", src)
        self.assertNotIn("linear_probe_top1", src,
                         "the vocabulary is spelled out here as well as in "
                         "adapterlib, so the two can drift apart")


if __name__ == "__main__":
    unittest.main()
