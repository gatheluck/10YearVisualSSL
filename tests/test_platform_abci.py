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
