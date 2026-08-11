#!/usr/bin/env python3
"""Behaviour of the ABCI backend.

This backend is **optional** and the core never references it; the separation
itself is held by `tests/test_platform_isolation.py`. Here we check that what
it does is correct.

**These tests run where the submit command does not exist.** Actually
submitting would not be a test, and it would scatter jobs across a shared
machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import platforms                                    # noqa: E402

abci = platforms.load_backend("abci")
JobSpec = platforms.JobSpec


def spec(**over) -> JobSpec:
    base = dict(name="job", command=["python3", "-m", "adapter"],
                env_name="py3.10_x", gpus=8, hours=24)
    base.update(over)
    return JobSpec(**base)


class TestResourceTranslation(unittest.TestCase):
    """Translating a need into a resource. **The table lives only here.**"""

    def test_known_amounts_translate(self):
        for gpus in (0, 1, 8):
            self.assertTrue(abci.resource_type(gpus))

    def test_unknown_amount_is_refused_not_rounded(self):
        """**Never round.** Unintended resources change the result quietly."""
        with self.assertRaises(ValueError) as e:
            abci.resource_type(3)
        self.assertIn("3", str(e.exception))

    def test_the_error_says_what_is_available(self):
        with self.assertRaises(ValueError) as e:
            abci.resource_type(99)
        self.assertIn("8", str(e.exception), "it does not say which values are usable")


class TestScriptRendering(unittest.TestCase):
    """Pure, so it can be checked on its own."""

    def test_required_directives_are_present(self):
        s = abci.render_script(spec(), group="grp")
        for frag in ("#!/bin/bash", "walltime=24:00:00", "grp", "job"):
            self.assertIn(frag, s, f"{frag} is missing")

    def test_command_is_included(self):
        self.assertIn("python3 -m adapter",
                      abci.render_script(spec(), group="g"))

    def test_environment_is_exported_deterministically(self):
        """An unstable order makes the script differ every run, hiding real changes."""
        s1 = abci.render_script(spec(env={"B": "2", "A": "1"}), group="g")
        s2 = abci.render_script(spec(env={"A": "1", "B": "2"}), group="g")
        self.assertEqual(s1, s2, "the environment ordering is not stable")
        self.assertLess(s1.index('export A='), s1.index('export B='))

    def test_workdir_is_used_when_given(self):
        self.assertIn('cd "/w"',
                      abci.render_script(spec(workdir="/w"), group="g"))

    def test_unknown_gpu_count_propagates_as_an_error(self):
        with self.assertRaises(ValueError):
            abci.render_script(spec(gpus=3), group="g")

    def test_the_logs_are_merged_into_one_stream(self):
        """A single stdout+stderr log is the easiest to read when a job fails."""
        self.assertIn("-j oe", abci.render_script(spec(), group="g"))

    def test_a_failing_command_reports_where_it_failed(self):
        """`set -e` alone stops silently. The trap names the line, the command
        and the exit code, so a failure is diagnosable from the log alone."""
        s = abci.render_script(spec(), group="g")
        self.assertIn("trap", s)
        self.assertIn("ERR", s)
        for frag in ("$LINENO", "$BASH_COMMAND"):
            self.assertIn(frag, s, f"the trap does not report {frag}")

    def test_it_probes_the_environment_before_running(self):
        """The ABCI-specific silent failures -- wrong interpreter, no GPU
        visible, a submodule not checked out -- must be visible in the log
        before the command runs, so the cause is found in one look."""
        s = abci.render_script(spec(), group="g")
        for probe in ("hostname", "nvidia-smi", "python", "import torch",
                      "git submodule status"):
            self.assertIn(probe, s, f"no {probe!r} in the diagnostics")

    def test_diagnostics_never_abort_the_job(self):
        """Each probe is guarded, so a missing tool (no nvidia-smi on a CPU
        node) does not fail the job before the real command runs."""
        s = abci.render_script(spec(), group="g")
        diag = s[:s.index("import torch")]
        self.assertIn("|| true", diag)

    def test_setup_lines_are_emitted_before_the_command(self):
        """Environment activation is injected at run time (so nothing
        machine-specific lives in the repo) and must run before the command."""
        s = abci.render_script(
            spec(setup=["module load cuda/12.6",
                        "source .venvs/m/bin/activate"]),
            group="g")
        self.assertIn("module load cuda/12.6", s)
        self.assertIn("source .venvs/m/bin/activate", s)
        self.assertLess(s.index("module load cuda/12.6"),
                        s.index("python3 -m adapter"))
        # order among setup lines is preserved
        self.assertLess(s.index("module load cuda/12.6"),
                        s.index("source .venvs/m/bin/activate"))

    def test_setup_runs_with_unset_variables_tolerated(self):
        """venv/conda activation scripts reference unset variables; nounset
        would abort them. `-u` is relaxed around the injected setup only, then
        restored, so the rest of the job keeps the strict setting."""
        s = abci.render_script(
            spec(setup=["source .venvs/m/bin/activate"]), group="g")
        i_relax = s.index("set +u")
        i_setup = s.index("source .venvs/m/bin/activate")
        i_restore = s.index("set -u", i_relax + 1)
        self.assertLess(i_relax, i_setup)
        self.assertLess(i_setup, i_restore)


class TestGroupInjection(unittest.TestCase):
    """The group id is injected, never baked in: the repo is public."""

    def test_group_is_read_from_the_environment_when_not_passed(self):
        with mock.patch.dict(os.environ, {"ABCI_GROUP": "from-env"}):
            self.assertEqual(abci.Backend().group, "from-env")

    def test_an_explicit_group_still_wins(self):
        with mock.patch.dict(os.environ, {"ABCI_GROUP": "from-env"}):
            self.assertEqual(abci.Backend("explicit").group, "explicit")

    def test_submit_refuses_when_no_group_is_set_and_says_how(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            b = abci.Backend()
            with mock.patch.object(abci.shutil, "which",
                                   return_value="/x/qsub"):
                with self.assertRaises(RuntimeError) as e:
                    b.submit(spec())
        self.assertIn("ABCI_GROUP", str(e.exception),
                      "it does not say which variable to set")


class TestAvailability(unittest.TestCase):
    def test_available_when_the_submit_command_exists(self):
        with mock.patch.object(abci.shutil, "which", return_value="/x/qsub"):
            self.assertTrue(abci.Backend("g").is_available())

    def test_unavailable_when_it_does_not(self):
        """**Check, do not assume.**"""
        with mock.patch.object(abci.shutil, "which", return_value=None):
            self.assertFalse(abci.Backend("g").is_available())


class TestSubmit(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="abcitest-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))

    def _submit(self, returncode=0, stdout="12345.pbs\n", stderr=""):
        b = abci.Backend("grp", script_dir=self.tmp)
        fake = types.SimpleNamespace(returncode=returncode, stdout=stdout,
                                     stderr=stderr)
        with mock.patch.object(abci.shutil, "which", return_value="/x/qsub"), \
             mock.patch.object(abci.subprocess, "run", return_value=fake):
            return b.submit(spec())

    def test_exit_status_is_unknown_not_zero(self):
        """**Enqueuing tells you nothing about the outcome.**

        0 means "it succeeded". Passing off an unknown outcome as a success
        makes the caller treat failed jobs as finished ones. Mutation testing
        showed this path had no test at all, so it was added.
        """
        r = self._submit()
        self.assertIsNone(r.exit_status,
                          "claims an exit status although the job was only enqueued")

    def test_job_id_comes_from_the_submitter(self):
        self.assertEqual(self._submit().job_id, "12345.pbs")

    def test_script_is_written(self):
        self._submit()
        self.assertTrue((self.tmp / "job.sh").is_file())

    def test_submission_failure_is_loud(self):
        """The scheduler's own words must survive; do not paraphrase them.

        Written with escapes because a scheduler may answer in any language
        and this file has to stay ASCII (see tests/test_language.py).
        """
        msg = "\u62d2\u5426: quota exceeded"
        with self.assertRaises(RuntimeError) as e:
            self._submit(returncode=1, stderr=msg)
        self.assertIn(msg, str(e.exception))

    def test_refuses_when_unavailable_and_says_what_to_do(self):
        b = abci.Backend("grp", script_dir=self.tmp)
        with mock.patch.object(abci.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError) as e:
                b.submit(spec())
        self.assertIn("local", str(e.exception),
                      "does not say what to use instead")

    def test_nothing_is_submitted_when_unavailable(self):
        b = abci.Backend("grp", script_dir=self.tmp)
        with mock.patch.object(abci.shutil, "which", return_value=None), \
             mock.patch.object(abci.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                b.submit(spec())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
