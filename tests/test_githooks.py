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
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / ".githooks" / "pre-commit"


class TestHookExists(unittest.TestCase):
    def test_hook_is_present(self):
        self.assertTrue(HOOK.is_file(), "the pre-commit hook is missing")

    def test_hook_is_executable(self):
        """Without the executable bit git ignores the hook silently."""
        self.assertTrue(HOOK.stat().st_mode & stat.S_IXUSR,
                        "not executable; git would ignore it without a word")

    def test_hook_runs_the_suite(self):
        self.assertIn("tests/run-tests.sh", HOOK.read_text(encoding="utf-8"))


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


if __name__ == "__main__":
    unittest.main()
