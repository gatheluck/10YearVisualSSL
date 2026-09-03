#!/usr/bin/env python3
"""Specification for the pre-commit hook.

**A written rule did not hold.** On 2026-07-29 a commit went out, and was
pushed, with the test suite red. The rule was stated in three separate places
at the time. A policy that is only written down does not survive; this is
enforced mechanically instead.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _checkout import needs_checkout        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / ".githooks" / "pre-commit"
PREPUSH = ROOT / ".githooks" / "pre-push"


@needs_checkout
class TestHookExists(unittest.TestCase):
    def test_hook_is_present(self):
        self.assertTrue(HOOK.is_file(), "the pre-commit hook is missing")

    def test_hook_is_executable(self):
        """Without the executable bit git ignores the hook silently."""
        self.assertTrue(HOOK.stat().st_mode & stat.S_IXUSR,
                        "not executable; git would ignore it without a word")

    def test_hook_runs_the_suite(self):
        self.assertIn("tests/run-tests.sh", HOOK.read_text(encoding="utf-8"))


@needs_checkout
class TestHookBehaviour(unittest.TestCase):
    """Drive the hook's branches for real, with a stand-in test script."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hooktest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        subprocess.run(["git", "init", "-q", str(self.tmp)],
                       check=True, capture_output=True)
        (self.tmp / "tests").mkdir()
        shutil.copy(HOOK, self.tmp / "pre-commit")
        os.chmod(self.tmp / "pre-commit", 0o755)

    def fake_suite(self, exit_code: int) -> None:
        p = self.tmp / "tests" / "run-tests.sh"
        p.write_text(f"#!/usr/bin/env bash\necho stand-in suite\n"
                     f"exit {exit_code}\n")
        os.chmod(p, 0o755)

    def run_hook(self, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.tmp / "pre-commit")],
                              cwd=self.tmp, env=env or os.environ.copy(),
                              capture_output=True, text=True)

    def test_green_suite_allows_the_commit(self):
        self.fake_suite(0)
        self.assertEqual(self.run_hook().returncode, 0)

    def test_red_suite_blocks_the_commit(self):
        self.fake_suite(1)
        r = self.run_hook()
        self.assertNotEqual(r.returncode, 0, "red suite, yet the commit passed")
        self.assertIn("aborted", r.stderr)

    def test_git_environment_is_cleared_before_running_the_suite(self):
        """**git hands GIT_DIR and friends to hooks.**

        Left in place, the git processes started by the tests operate on this
        repository instead of their own fixtures and `git add` fails with
        exit 128. That produced 120 errors in one run before this was fixed.
        """
        p = self.tmp / "tests" / "run-tests.sh"
        p.write_text(
            "#!/usr/bin/env bash\n"
            'for v in GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE; do\n'
            '  if [ -n "${!v:-}" ]; then echo "$v still set"; exit 1; fi\n'
            "done\nexit 0\n")
        os.chmod(p, 0o755)
        env = {**os.environ, "GIT_DIR": "/somewhere/.git",
               "GIT_WORK_TREE": "/somewhere", "GIT_INDEX_FILE": "/tmp/idx"}
        r = self.run_hook(env)
        self.assertEqual(r.returncode, 0,
                         f"git's environment was not cleared: {r.stderr}")

    def test_failure_output_is_shown(self):
        """Hide the reason and people reach for --no-verify."""
        self.fake_suite(1)
        self.assertIn("stand-in suite", self.run_hook().stderr)


@needs_checkout
class TestPrePushHookExists(unittest.TestCase):
    """The pre-commit hook runs the base interpreter, which has no torch: every
    test that builds a real model or backbone SKIPS there. Cross-method global
    contamination -- one method's leaked global state breaking a later method
    built in the same process -- is therefore invisible to pre-commit and to the
    always-on `core` CI job, and surfaces only in the PR-only `locked` matrix.
    That gap is why regressions keep reaching CI. The pre-push hook closes it by
    reproducing the `locked` condition (a torch venv, the whole suite in one
    process) locally, before the push.
    """

    def test_hook_is_present(self):
        self.assertTrue(PREPUSH.is_file(), "the pre-push hook is missing")

    def test_hook_is_executable(self):
        """Without the executable bit git ignores the hook silently."""
        self.assertTrue(PREPUSH.stat().st_mode & stat.S_IXUSR,
                        "not executable; git would ignore it without a word")

    def test_hook_runs_the_whole_suite_in_one_process(self):
        text = PREPUSH.read_text(encoding="utf-8")
        self.assertIn("unittest discover", text)
        self.assertIn("-s tests", text)


@needs_checkout
class TestPrePushHookBehaviour(unittest.TestCase):
    """Drive the hook's branches for real, with a stand-in venv interpreter.

    The hook auto-detects a venv whose python imports torch and runs the suite
    with it. The stand-in below answers the two invocations the hook makes --
    `python -c 'import torch'` (detection) and `python -m unittest discover`
    (the run) -- so every branch is exercised without a real torch install.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="prepushtest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        subprocess.run(["git", "init", "-q", str(self.tmp)],
                       check=True, capture_output=True)
        shutil.copy(PREPUSH, self.tmp / "pre-push")
        os.chmod(self.tmp / "pre-push", 0o755)

    def fake_venv(self) -> None:
        """A venv whose python imports torch (exit 0 on `-c`) and, on `-m`,
        emits a marker and exits with $FAKE_SUITE_EXIT."""
        py = self.tmp / ".venvs" / "fake" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text(
            "#!/usr/bin/env bash\n"
            "case \"$1\" in\n"
            "  -c) exit \"${FAKE_TORCH_IMPORT:-0}\" ;;\n"
            "  -m) echo 'stand-in torch suite'; exit \"${FAKE_SUITE_EXIT:-0}\" ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n")
        os.chmod(py, 0o755)

    def run_hook(self, overrides: dict | None = None
                 ) -> subprocess.CompletedProcess:
        """Run the hook with a hermetic environment.

        The hook honours two ambient variables -- TORCH_GATE_PYTHON and
        SKIP_TORCH_GATE. If the suite is itself run through the pre-push hook
        (or anyone exports either), that value would leak in and override the
        stand-in venv this test installs: the hook would pick a *real*
        interpreter and run `discover -s tests` in this temp dir, which has no
        `tests/`, failing with an unrelated ImportError. This test failed
        exactly that way in-suite before the scrub. Strip both from the base,
        then apply only what the test asks for.
        """
        base = os.environ.copy()
        base.pop("TORCH_GATE_PYTHON", None)
        base.pop("SKIP_TORCH_GATE", None)
        if overrides:
            base.update(overrides)
        return subprocess.run(["bash", str(self.tmp / "pre-push")],
                              cwd=self.tmp, env=base,
                              capture_output=True, text=True)

    def test_green_suite_allows_the_push(self):
        self.fake_venv()
        r = self.run_hook({"FAKE_SUITE_EXIT": "0"})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_red_suite_blocks_the_push(self):
        self.fake_venv()
        r = self.run_hook({"FAKE_SUITE_EXIT": "1"})
        self.assertNotEqual(r.returncode, 0, "red suite, yet the push passed")
        self.assertIn("aborted", r.stderr)

    def test_failure_output_is_shown(self):
        """Hide the reason and people reach for --no-verify."""
        self.fake_venv()
        r = self.run_hook({"FAKE_SUITE_EXIT": "1"})
        self.assertIn("stand-in torch suite", r.stderr)

    def test_no_torch_venv_is_announced_not_silent_and_allows(self):
        """A clone with no torch venv cannot run the gate. The skip must be
        announced (DESIGN 2.4: never a silent skip), and the push allowed rather
        than blocking work the hook cannot check."""
        self.fake_venv()  # present, but its python fails to import torch
        r = self.run_hook({"FAKE_TORCH_IMPORT": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no torch venv", r.stderr)

    def test_skip_env_is_announced_and_allows(self):
        self.fake_venv()
        r = self.run_hook({"SKIP_TORCH_GATE": "1", "FAKE_SUITE_EXIT": "1"})
        self.assertEqual(r.returncode, 0,
                         "SKIP_TORCH_GATE=1 should let the push through")
        self.assertIn("skip", r.stderr.lower())

    def test_git_environment_is_cleared_before_running_the_suite(self):
        """git hands GIT_DIR and friends to hooks; left in place the git the
        suite runs would operate on this repository, not its fixtures."""
        py = self.tmp / ".venvs" / "fake" / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text(
            "#!/usr/bin/env bash\n"
            "case \"$1\" in\n"
            "  -c) exit 0 ;;\n"
            "  -m)\n"
            "    for v in GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE; do\n"
            "      if [ -n \"${!v:-}\" ]; then echo \"$v still set\"; exit 1; fi\n"
            "    done\n"
            "    exit 0 ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n")
        os.chmod(py, 0o755)
        r = self.run_hook({"GIT_DIR": "/somewhere/.git",
                           "GIT_WORK_TREE": "/somewhere",
                           "GIT_INDEX_FILE": "/tmp/idx"})
        self.assertEqual(r.returncode, 0,
                         f"git's environment was not cleared: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
