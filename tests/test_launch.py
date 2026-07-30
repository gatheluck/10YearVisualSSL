#!/usr/bin/env python3
"""Specification for launch.py: resolve, submit, verify, record.

The pieces have existed for a while and nothing joined them. A scheduler
backend could submit a job and **nothing called it**; `resolve-config` and
`contract-test` were run by hand, in the right order, by whoever remembered
the order.

Three things it must get right, and each is a decision rather than plumbing.

**Resources are not part of the config.** `--gpus` and `--hours` are launcher
arguments, deliberately outside `config_sha256`: how long a scheduler is asked
to allow does not change the result, and folding it in would make two
identical experiments hash differently. What *does* affect the result --
`WORLD_SIZE` -- is recorded by the run itself, in the manifest.

**A submitted job is not a finished one.** Where the backend can say how it
went, the launcher verifies immediately. Where it cannot -- a scheduler that
has only queued the work -- it says so and stops, rather than verifying an
output directory that nothing has written yet.

**The invocation is recorded too.** The manifest says what the run did; it
cannot say what was asked of it. `launch.json` holds the authoring config, the
substitutions, the platform and the resources, so a run directory explains
itself without the shell history that produced it.

All of it is exercised here with the local backend and `methods/_reference`,
which needs nothing installed -- so these tests run in the dependency-free
job too, and none of it waits on a cluster.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


launch = load("launch", "launch.py")


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.runs = self.tmp / "runs"

    def authoring(self, **over) -> Path:
        cfg = {"seed": 0, "metrics": {"top1": 42.5}}
        cfg.update(over)
        p = self.tmp / "authoring.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return p

    def run_tool(self, *args, expect=None):
        r = subprocess.run(
            [sys.executable, str(BIN / "launch.py"), *map(str, args)],
            capture_output=True, text=True)
        if expect is not None:
            self.assertEqual(r.returncode, expect,
                             r.stdout[-2000:] + r.stderr[-2000:])
        return r

    def launch_reference(self, *extra, expect=0, **over):
        return self.run_tool("--config", self.authoring(**over),
                             "--method", "_reference",
                             "--runs-dir", self.runs, *extra, expect=expect)


class TestTheRunDirectory(Base):
    def test_it_is_named_after_the_config_that_produced_it(self):
        """**Not a timestamp.** A directory named for the configuration makes
        two runs of the same experiment collide, which is information: you
        meant to change something and did not."""
        self.launch_reference()
        made = [p for p in self.runs.iterdir() if p.is_dir()]
        self.assertEqual(len(made), 1)
        digest = json.loads((made[0] / "launch.json").read_text())
        self.assertIn(digest["config_sha256"][:12], made[0].name)
        self.assertIn("_reference", made[0].name)

    def test_the_same_config_twice_is_refused(self):
        """Silently overwriting would destroy the first result; silently
        skipping would report success for a run that did not happen."""
        self.launch_reference()
        r = self.launch_reference(expect=None)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already", (r.stdout + r.stderr).lower())

    def test_it_can_be_repeated_when_asked_explicitly(self):
        self.launch_reference()
        self.launch_reference("--again")

    def test_a_different_config_gets_a_different_directory(self):
        self.launch_reference(seed=0)
        self.launch_reference(seed=1)
        self.assertEqual(len([p for p in self.runs.iterdir() if p.is_dir()]), 2)

    def test_nothing_is_written_outside_the_runs_directory(self):
        self.authoring()          # the config exists before the snapshot
        before = {p for p in self.tmp.iterdir()}
        self.launch_reference()
        after = {p for p in self.tmp.iterdir()}
        self.assertEqual(after - before, {self.runs})


class TestResolving(Base):
    def test_the_resolved_config_is_kept_beside_the_output(self):
        """`contract-test` needs it later, and a config that lives only in
        somebody's shell history cannot be checked against."""
        self.launch_reference()
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        self.assertTrue((run / "resolved.json").is_file())

    def test_substitutions_are_passed_through(self):
        p = self.tmp / "with_var.json"
        p.write_text(json.dumps({"seed": 0, "metrics": {"top1": 1.0},
                                 "note": "${WHERE}"}), encoding="utf-8")
        self.run_tool("--config", p, "--method", "_reference",
                      "--runs-dir", self.runs, "--set", "WHERE=here", expect=0)
        run = next(x for x in self.runs.iterdir() if x.is_dir())
        self.assertEqual(
            json.loads((run / "resolved.json").read_text())["note"], "here")

    def test_an_unresolvable_config_stops_before_anything_runs(self):
        p = self.tmp / "bad.json"
        p.write_text(json.dumps({"seed": 0, "note": "${MISSING}"}),
                     encoding="utf-8")
        r = self.run_tool("--config", p, "--method", "_reference",
                          "--runs-dir", self.runs)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MISSING", r.stdout + r.stderr)
        leftover = list(self.runs.iterdir()) if self.runs.exists() else []
        self.assertEqual(leftover, [],
                         "a failed resolution left something behind, so the "
                         "next run would look like a repeat")


class TestSubmitting(Base):
    def test_the_adapter_is_run_and_produces_the_contract_outputs(self):
        self.launch_reference()
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        for name in ("encoder.pt", "metrics.json", "run_manifest.json"):
            self.assertTrue((run / "out" / name).is_file(), name)

    def test_the_platform_defaults_to_the_local_machine(self):
        self.launch_reference()
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        self.assertEqual(
            json.loads((run / "launch.json").read_text())["platform"], "local")

    def test_an_unknown_platform_is_refused_and_lists_the_known_ones(self):
        r = self.launch_reference("--platform", "nowhere", expect=None)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("local", r.stdout + r.stderr)

    def test_an_unknown_method_is_refused_by_name(self):
        """**Refused, not merely failed.**

        Without the check the launcher still exits non-zero -- it crashes
        trying to run in a directory that is not there, and the path happens
        to contain the name. That satisfied an earlier version of this test
        while telling the reader nothing. What has to hold is a refusal that
        names the methods that do exist.
        """
        r = self.run_tool("--config", self.authoring(), "--method", "nope",
                          "--runs-dir", self.runs)
        self.assertEqual(r.returncode, 2, "not a refusal, just a crash")
        msg = r.stdout + r.stderr
        self.assertIn("nope", msg)
        self.assertIn("_reference", msg, "the known methods are not listed")
        self.assertFalse(self.runs.exists(),
                         "a run directory was created for an unknown method")

    def test_a_failing_run_is_reported_as_a_failure(self):
        r = self.launch_reference(expect=None, fail="deliberate")
        self.assertNotEqual(r.returncode, 0)


class TestTheJobEnvironment(Base):
    """CONTRACT section 2 makes the launcher responsible for these.

    It set only PYTHONPATH, so the adapter fell back to whatever the shell
    happened to hold. `adapterlib` reads WORLD_SIZE from the environment and
    records it in the manifest, so a developer with WORLD_SIZE left over from
    something else would have had it written into their results.

    Multi-process fan-out is a separate matter and is not implemented; what is
    fixed here is that the single-process case is *stated* rather than
    inherited.
    """

    def env_of(self, **over) -> dict:
        return launch.job_environment(**over)

    def test_the_single_process_ranks_are_stated_not_inherited(self):
        env = self.env_of(gpus=0)
        self.assertEqual(env["WORLD_SIZE"], "1")
        self.assertEqual(env["RANK"], "0")
        self.assertEqual(env["LOCAL_RANK"], "0")

    def test_they_override_whatever_the_shell_holds(self):
        """The hazard this closes: a stale WORLD_SIZE in someone's shell being
        recorded as a fact about the run."""
        os.environ["WORLD_SIZE"] = "99"
        self.addCleanup(os.environ.pop, "WORLD_SIZE", None)
        self.launch_reference()
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        man = json.loads((run / "out" / "run_manifest.json").read_text())
        self.assertEqual(man["world_size"], 1,
                         "the shell's WORLD_SIZE reached the manifest")

    def test_the_python_path_still_reaches_the_adapter(self):
        self.assertIn("PYTHONPATH", self.env_of(gpus=0))

    def test_asking_for_more_than_one_process_is_refused_for_now(self):
        """**Not silently run as one process.** Fan-out is not implemented,
        and quietly giving one process where several were asked for would
        produce a result that looks like the one requested."""
        r = self.launch_reference("--gpus", "8", "--processes", "8",
                                  expect=None)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("fan-out", (r.stdout + r.stderr).lower())

    def test_gpus_alone_does_not_imply_fan_out(self):
        """--gpus is a resource request for the scheduler. It says nothing
        about how many processes run, so it must not silently change it."""
        self.launch_reference("--gpus", "8")
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        man = json.loads((run / "out" / "run_manifest.json").read_text())
        self.assertEqual(man["world_size"], 1)


class TestVerifying(Base):
    """Running is not the same as having run correctly."""

    def test_a_conforming_run_is_verified_against_the_contract(self):
        r = self.launch_reference()
        self.assertIn("contract", r.stdout.lower())
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        self.assertTrue(json.loads(
            (run / "launch.json").read_text())["contract_ok"])

    def test_the_verdict_survives_in_the_record(self):
        r = self.launch_reference(expect=None, fail="deliberate")
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        rec = json.loads((run / "launch.json").read_text())
        self.assertFalse(rec["contract_ok"])
        self.assertNotEqual(rec["exit_status"], 0)

    def test_a_broken_output_makes_the_launch_fail(self):
        """The launcher must not report success for a run whose outputs do
        not satisfy the contract, whatever the exit status said."""
        self.launch_reference()
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        (run / "out" / "encoder.pt").write_bytes(b"tampered")
        r = self.run_tool("--verify-only", run)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("artifact-sha256", r.stdout + r.stderr)


class TestASubmittedJobIsNotAFinishedOne(Base):
    """A backend that only queues cannot be verified yet, and must not be."""

    def test_an_unknown_exit_status_is_not_treated_as_success(self):
        rec = launch.summarise(exit_status=None, contract_ok=None)
        self.assertNotEqual(rec["outcome"], "ok")
        self.assertEqual(rec["outcome"], "submitted")

    def test_a_queued_job_is_not_verified(self):
        """Verifying an output directory nothing has written yet would report
        a contract violation for a job that has not started."""
        self.assertFalse(launch.should_verify(exit_status=None))
        self.assertTrue(launch.should_verify(exit_status=0))
        self.assertTrue(launch.should_verify(exit_status=1))

    def test_the_two_signals_must_agree_for_success(self):
        self.assertEqual(launch.summarise(0, True)["outcome"], "ok")
        for status, ok in ((0, False), (1, True), (1, False)):
            with self.subTest(exit_status=status, contract_ok=ok):
                self.assertNotEqual(
                    launch.summarise(status, ok)["outcome"], "ok")


class TestTheInvocationIsRecorded(Base):
    """The manifest says what the run did. It cannot say what was asked."""

    def test_the_record_names_everything_needed_to_repeat_it(self):
        self.launch_reference("--gpus", "8", "--hours", "24")
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        rec = json.loads((run / "launch.json").read_text())
        for key in ("authoring_config", "config_sha256", "method", "platform",
                    "gpus", "hours", "set", "exit_status", "contract_ok",
                    "outcome", "launcher_version"):
            self.assertIn(key, rec)

    def test_the_substitutions_are_in_it(self):
        p = self.tmp / "v.json"
        p.write_text(json.dumps({"seed": 0, "metrics": {"top1": 1.0},
                                 "note": "${WHERE}"}), encoding="utf-8")
        self.run_tool("--config", p, "--method", "_reference", "--runs-dir",
                      self.runs, "--set", "WHERE=here", expect=0)
        run = next(x for x in self.runs.iterdir() if x.is_dir())
        self.assertEqual(
            json.loads((run / "launch.json").read_text())["set"],
            {"WHERE": "here"})

    def test_the_resources_are_recorded_but_stay_out_of_the_config(self):
        """**The decision this file exists to pin.** Walltime does not change
        a result; folding it into the config would make two identical
        experiments hash differently."""
        self.launch_reference("--gpus", "8", "--hours", "24")
        run = next(p for p in self.runs.iterdir() if p.is_dir())
        rec = json.loads((run / "launch.json").read_text())
        resolved = json.loads((run / "resolved.json").read_text())
        self.assertEqual(rec["gpus"], 8)
        self.assertEqual(rec["hours"], 24)
        self.assertNotIn("gpus", resolved)
        self.assertNotIn("hours", resolved)

    def test_resources_do_not_change_the_run_identity(self):
        self.launch_reference("--gpus", "1")
        first = next(p for p in self.runs.iterdir() if p.is_dir()).name
        shutil.rmtree(self.runs)
        self.launch_reference("--gpus", "8")
        self.assertEqual(
            next(p for p in self.runs.iterdir() if p.is_dir()).name, first)


class TestItStaysLooselyCoupled(unittest.TestCase):
    def test_the_launcher_names_no_platform_but_the_default(self):
        """Naming one would tie the core to it. `local` is the default and is
        self-contained; everything else is resolved by name at runtime.

        Checked as a substring of the whole file, not as an exact constant.
        The first version compared constants for equality, so a docstring
        naming a platform slipped past -- `tests/test_platform_isolation.py`
        caught it instead, which is the stricter check doing the weaker one's
        job.
        """
        src = (BIN / "launch.py").read_text()
        from platforms import available_backends
        for backend in available_backends():
            if backend == "local":
                continue
            with self.subTest(backend=backend):
                self.assertNotIn(backend, src,
                                 f"the launcher names {backend}")

    def test_the_backend_is_resolved_by_name_at_runtime(self):
        src = (BIN / "launch.py").read_text()
        self.assertIn("load_backend", src)


if __name__ == "__main__":
    unittest.main()
