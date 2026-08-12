#!/usr/bin/env python3
"""Specification for adapterlib: the one place a run_manifest is written.

Ten years of methods means ten years of incompatible environments, so each
adapter is its own process. What must **not** be repeated per method is the
manifest logic: CLAUDE.md forbids implementing one rule twice, and scanners
that disagreed about classification are the common root of past defects here.
So every adapter writes its manifest through this module.

That puts three obligations on it:

- **Standard library only, and importable on every generation.** Measurement
  of 64 environment definitions found Python 3.10 in 62 and 3.12 in 2, so the
  module must run on 3.10 without depending on anything installed
- **Both success signals come from one place.** Exit status and
  `status:` are written together, so they cannot disagree
- **Every file under `--out` is listed.** An output nobody knows about is a
  hole in reproducibility, and the adapter is the only thing that can see them
  all at the moment the run ends
"""

from __future__ import annotations

import hashlib
import os
import json
import subprocess
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import adapterlib                                     # noqa: E402


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="altest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.config = self.tmp / "resolved.json"
        self.config.write_text('{"seed":7}\n', encoding="utf-8")

    def manifest(self) -> dict:
        return json.loads((self.out / "run_manifest.json").read_text())

    def run_ok(self, body=None, **kw):
        """The ordinary case: a run that produces an encoder.

        Unless the caller declares why there is no encoder, one is written
        after the body -- otherwise adapterlib refuses the run, which is the
        behaviour TestMissingEncoder exists to check.
        """
        kw.setdefault("method", "m")
        kw.setdefault("stage", "s")
        inner = body or (lambda ctx: None)

        def wrapped(ctx):
            inner(ctx)
            if not kw.get("encoder_absent_reason"):
                p = ctx.out / "encoder.pt"
                if not p.exists():
                    p.write_bytes(b"weights")
        return adapterlib.run(config=self.config, out=self.out,
                              body=wrapped, **kw)


class TestSuccessPath(Base):
    def test_it_writes_a_manifest(self):
        self.run_ok()
        self.assertTrue((self.out / "run_manifest.json").is_file())

    def test_the_status_is_ok_and_the_return_is_zero(self):
        self.assertEqual(self.run_ok(), 0)
        self.assertEqual(self.manifest()["status"], "ok")

    def test_every_required_field_is_present(self):
        self.run_ok()
        man = self.manifest()
        for f in ("schema_version", "method", "stage", "status",
                  "config_sha256", "started_at", "finished_at", "seed",
                  "world_size", "env", "upstream", "artifacts"):
            self.assertIn(f, man)

    def test_the_config_hash_is_of_the_file_that_was_handed_in(self):
        self.run_ok()
        self.assertEqual(
            self.manifest()["config_sha256"],
            hashlib.sha256(self.config.read_bytes()).hexdigest())

    def test_the_seed_comes_from_the_config(self):
        self.run_ok()
        self.assertEqual(self.manifest()["seed"], 7)

    def test_a_config_without_a_seed_is_refused(self):
        """An unrecorded seed makes the run unreproducible by definition."""
        self.config.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.run_ok()
        self.assertIn("seed", str(e.exception))

    def test_times_are_utc_and_in_order(self):
        self.run_ok()
        man = self.manifest()
        self.assertTrue(man["started_at"].endswith("Z"))
        self.assertTrue(man["finished_at"].endswith("Z"))
        self.assertLessEqual(man["started_at"], man["finished_at"])

    def test_the_environment_is_recorded(self):
        self.run_ok()
        self.assertEqual(self.manifest()["env"]["python"],
                         ".".join(str(x) for x in sys.version_info[:3]))

    def test_the_installed_packages_are_recorded(self):
        """**A run that cannot say which torch produced it is not a record.**

        `env` held only python and hostname, so nothing in the manifest
        identified the environment -- while CONTRACT section 3 shows the
        library version in the example. Bitwise agreement across machines is
        not achievable for floating-point work; knowing the environments
        differed is, and it is the whole of what makes a difference
        explainable.
        """
        self.run_ok()
        pkgs = self.manifest()["env"]["packages"]
        self.assertIsInstance(pkgs, dict)
        self.assertTrue(pkgs, "no packages were recorded")
        # unittest is standard library, so something installed must appear:
        # the interpreter always has at least pip or setuptools available in
        # the environments this runs in.
        for name, version in pkgs.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(version, str)

    def test_the_package_set_has_a_fingerprint(self):
        """One value to compare, so two runs can be told apart at a glance."""
        self.run_ok()
        env = self.manifest()["env"]
        want = adapterlib.fingerprint(env["packages"])
        self.assertEqual(env["packages_sha256"], want)
        self.assertRegex(env["packages_sha256"], r"^[0-9a-f]{64}$")

    def test_the_fingerprint_changes_with_the_packages(self):
        """A constant would satisfy the test above."""
        a = adapterlib.fingerprint({"torch": "2.0.1"})
        b = adapterlib.fingerprint({"torch": "2.1.0"})
        self.assertNotEqual(a, b)

    def test_the_fingerprint_ignores_ordering(self):
        """Two identical environments must not look different."""
        self.assertEqual(adapterlib.fingerprint({"a": "1", "b": "2"}),
                         adapterlib.fingerprint({"b": "2", "a": "1"}))

    def test_the_platform_is_recorded(self):
        """x86_64 and arm64 give different floating-point results. The
        manifest has to say which one ran."""
        self.run_ok()
        env = self.manifest()["env"]
        for key in ("system", "machine"):
            self.assertTrue(env.get(key), f"{key} is not recorded")

    def test_world_size_defaults_to_one(self):
        self.run_ok(env={})
        self.assertEqual(self.manifest()["world_size"], 1)

    def test_world_size_comes_from_the_launcher(self):
        """The launcher owns fan-out; the adapter only records what it got."""
        self.run_ok(env={"WORLD_SIZE": "8"})
        self.assertEqual(self.manifest()["world_size"], 8)

    def test_a_non_numeric_world_size_is_refused(self):
        with self.assertRaises(adapterlib.AdapterError):
            self.run_ok(env={"WORLD_SIZE": "lots"})


class TestArtifactListing(Base):
    def body_writing(self, files: dict):
        def body(ctx):
            for rel, data in files.items():
                p = ctx.out / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        return body

    def test_written_files_are_listed_with_hash_and_size(self):
        self.run_ok(self.body_writing({"encoder.pt": b"w"}))
        art = {a["path"]: a for a in self.manifest()["artifacts"]}
        self.assertEqual(art["encoder.pt"]["sha256"],
                         hashlib.sha256(b"w").hexdigest())
        self.assertEqual(art["encoder.pt"]["bytes"], 1)

    def test_files_in_subdirectories_are_listed(self):
        self.run_ok(self.body_writing({"logs/train.log": b"x"}))
        self.assertIn("logs/train.log",
                      [a["path"] for a in self.manifest()["artifacts"]])

    def test_nothing_the_body_wrote_is_left_out(self):
        """**The adapter is the only thing that can see them all.**"""
        files = {f"f{i}.bin": bytes([i]) for i in range(5)}
        self.run_ok(self.body_writing(files),
                    encoder_absent_reason="not part of this check")
        listed = {a["path"] for a in self.manifest()["artifacts"]}
        self.assertEqual(listed, set(files))

    def test_the_manifest_does_not_list_itself(self):
        """It cannot contain its own hash. That is the one exception."""
        self.run_ok()
        self.assertNotIn("run_manifest.json",
                         [a["path"] for a in self.manifest()["artifacts"]])

    def test_a_manifest_left_by_an_earlier_run_is_not_listed(self):
        """Mutation testing found the exclusion untested.

        On a first run the manifest does not exist yet when the outputs are
        collected, so removing the guard changes nothing. It only bites when
        `--out` already holds one -- a re-run into the same directory -- and
        then the manifest would claim a hash of its own previous self.
        """
        self.run_ok()
        self.assertTrue((self.out / "run_manifest.json").is_file())
        self.run_ok()
        self.assertNotIn("run_manifest.json",
                         [a["path"] for a in self.manifest()["artifacts"]])

    def test_known_names_get_their_fixed_roles(self):
        self.run_ok(self.body_writing({"encoder.pt": b"w",
                                       "metrics.json": b'{"metrics":{}}'}))
        art = {a["path"]: a["role"] for a in self.manifest()["artifacts"]}
        self.assertEqual(art["encoder.pt"], "encoder")
        self.assertEqual(art["metrics.json"], "metrics")

    def test_other_files_are_listed_rather_than_dropped(self):
        self.run_ok(self.body_writing({"notes.txt": b"x"}))
        art = {a["path"]: a["role"] for a in self.manifest()["artifacts"]}
        self.assertIn("notes.txt", art)
        self.assertTrue(art["notes.txt"], "an artifact was listed with no role")

    def test_paths_are_recorded_with_forward_slashes(self):
        """The manifest must read the same on every platform."""
        self.run_ok(self.body_writing({"a/b/c.bin": b"x"}))
        self.assertIn("a/b/c.bin",
                      [a["path"] for a in self.manifest()["artifacts"]])

    def test_artifacts_are_in_a_stable_order(self):
        """An unstable order changes the manifest bytes for no reason."""
        files = {"z.bin": b"1", "a.bin": b"2", "m.bin": b"3"}
        self.run_ok(self.body_writing(files))
        paths = [a["path"] for a in self.manifest()["artifacts"]]
        self.assertEqual(paths, sorted(paths))


class TestMissingEncoder(Base):
    """CONTRACT section 3: not producing one is allowed. Doing so quietly is not."""

    def test_a_declared_reason_is_recorded(self):
        self.run_ok(encoder_absent_reason="this stage trains no encoder")
        self.assertIn("trains no encoder",
                      self.manifest()["encoder_absent_reason"])

    def test_an_absent_encoder_with_no_reason_is_refused(self):
        # Deliberately not run_ok(), which supplies an encoder for the
        # ordinary case; this is the case where nothing supplies one.
        with self.assertRaises(adapterlib.AdapterError) as e:
            adapterlib.run(config=self.config, out=self.out, method="m",
                           stage="s", body=lambda ctx: None)
        self.assertIn("encoder.pt", str(e.exception))

    def test_a_refusal_writes_no_manifest(self):
        """The contract already says an absent manifest means failure, and a
        manifest describing a run that was refused would be a false record."""
        with self.assertRaises(adapterlib.AdapterError):
            adapterlib.run(config=self.config, out=self.out, method="m",
                           stage="s", body=lambda ctx: None)
        self.assertFalse((self.out / "run_manifest.json").exists())

    def test_declaring_a_reason_while_producing_one_is_refused(self):
        """The two statements contradict each other; do not let both stand."""
        def body(ctx):
            (ctx.out / "encoder.pt").write_bytes(b"w")
        with self.assertRaises(adapterlib.AdapterError):
            self.run_ok(body, encoder_absent_reason="none produced")


class TestFailurePath(Base):
    def test_a_failing_body_is_recorded_as_failed(self):
        def body(ctx):
            raise RuntimeError("the loss went to nan")
        rc = adapterlib.run(config=self.config, out=self.out, method="m",
                            stage="s", body=body,
                            encoder_absent_reason="failed before saving")
        self.assertNotEqual(rc, 0)
        self.assertEqual(self.manifest()["status"], "failed")

    def test_the_reason_is_recorded_not_swallowed(self):
        def body(ctx):
            raise RuntimeError("the loss went to nan")
        adapterlib.run(config=self.config, out=self.out, method="m", stage="s",
                       body=body, encoder_absent_reason="failed early")
        self.assertIn("nan", self.manifest()["error"])

    def test_a_failed_run_still_lists_what_it_wrote(self):
        """Partial output is evidence. Losing it makes the failure harder to
        understand, and leaves files no manifest knows about."""
        def body(ctx):
            (ctx.out / "partial.bin").write_bytes(b"x")
            raise RuntimeError("died after writing")
        adapterlib.run(config=self.config, out=self.out, method="m", stage="s",
                       body=body, encoder_absent_reason="failed early")
        self.assertIn("partial.bin",
                      [a["path"] for a in self.manifest()["artifacts"]])

    def test_the_two_signals_agree_on_failure(self):
        """exit != 0 with status ok would be one of them lying."""
        def body(ctx):
            raise RuntimeError("x")
        rc = adapterlib.run(config=self.config, out=self.out, method="m",
                            stage="s", body=body,
                            encoder_absent_reason="failed early")
        self.assertEqual((rc != 0), (self.manifest()["status"] == "failed"))

    def test_a_missing_config_is_refused(self):
        with self.assertRaises(adapterlib.AdapterError):
            adapterlib.run(config=self.tmp / "gone.json", out=self.out,
                           method="m", stage="s", body=lambda ctx: None)


class TestMetrics(Base):
    def test_metrics_are_written_in_the_contract_shape(self):
        def body(ctx):
            ctx.write_metrics(
                {"top1": 42.5},
                names={"top1": "final_linear_probe_top1_accuracy"})
        self.run_ok(body, stage="linear_eval",
                    encoder_absent_reason="no encoder in this test")
        m = json.loads((self.out / "metrics.json").read_text())
        self.assertEqual(m, {
            "schema_version": 2,
            "metrics": {"final_linear_probe_top1_accuracy": 42.5},
            "metrics_raw": {"top1": 42.5}})

    def test_the_stage_reaches_the_context(self):
        """It has to, or the family rule has nothing to read and every port
        silently falls into the refusing branch."""
        seen = {}

        def body(ctx):
            seen["stage"] = ctx.stage
            ctx.write_metrics({"n": 1}, names={"n": "steps_completed"})
        self.run_ok(body, stage="linear_eval",
                    encoder_absent_reason="no encoder in this test")
        self.assertEqual(seen["stage"], "linear_eval")

    def test_a_non_numeric_metric_is_refused_at_the_source(self):
        """Catching it here names the metric; contract-test can only say the
        file is wrong after the run has already cost its GPU hours."""
        def body(ctx):
            ctx.write_metrics(
                {"top1": "42.5"},
                names={"top1": "final_pretext_top1_accuracy"})
        with self.assertRaises(adapterlib.AdapterError) as e:
            self.run_ok(body, encoder_absent_reason="x")
        self.assertIn("top1", str(e.exception))

    def test_a_boolean_metric_is_refused(self):
        """In Python bool subclasses int, so it would slip past isinstance."""
        def body(ctx):
            ctx.write_metrics({"converged": True},
                              names={"converged": "epochs_completed"})
        with self.assertRaises(adapterlib.AdapterError):
            self.run_ok(body, encoder_absent_reason="x")


class TestStandardLibraryOnly(unittest.TestCase):
    def test_adapterlib_imports_nothing_outside_the_standard_library(self):
        """It has to load in every method environment, including 3.10 ones
        that have their own pinned dependencies."""
        src = (ROOT / "adapterlib" / "__init__.py").read_text()
        import ast
        names = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names.add((node.module or "").split(".")[0])
        allowed = set(sys.stdlib_module_names) | {"__future__", ""}
        self.assertEqual(names - allowed, set())

    def test_it_runs_on_a_bare_interpreter(self):
        """Measured, not assumed: import it with -I so nothing local leaks in."""
        r = subprocess.run(
            [sys.executable, "-I", "-c",
             f"import sys; sys.path.insert(0, {str(ROOT)!r}); import adapterlib"],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_it_avoids_syntax_newer_than_python_3_10(self):
        """62 of 64 measured environments are 3.10. `match` and `X | Y` in a
        runtime annotation would fail there."""
        import ast
        src = (ROOT / "adapterlib" / "__init__.py").read_text()
        self.assertIn("from __future__ import annotations", src,
                      "without it, `X | Y` annotations fail on 3.10")
        ast.parse(src, feature_version=(3, 10))


class TestDatasetSplitDir(unittest.TestCase):
    """The one data-root rule, resolved in one place (CLAUDE.md: never twice).
    DATA_ROOT is the dataset root; a stage reads its split from a subdirectory."""

    def test_it_joins_the_split_onto_the_root(self):
        self.assertEqual(adapterlib.dataset_split_dir("/data/imagenet"),
                         os.path.join("/data/imagenet", "train"))

    def test_the_split_is_configurable(self):
        self.assertEqual(
            adapterlib.dataset_split_dir("/data/imagenet", "val"),
            os.path.join("/data/imagenet", "val"))

    def test_require_refuses_a_missing_split_by_name(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with self.assertRaises(FileNotFoundError) as e:
            adapterlib.dataset_split_dir(tmp, "train", require=True)
        self.assertIn("train", str(e.exception))
        self.assertIn("DATA_ROOT", str(e.exception),
                      "the error does not say what DATA_ROOT should point at")

    def test_require_accepts_a_present_split(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "train").mkdir()
        self.assertEqual(
            adapterlib.dataset_split_dir(tmp, "train", require=True),
            str(tmp / "train"))


if __name__ == "__main__":
    unittest.main()
