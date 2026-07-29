#!/usr/bin/env python3
"""Nothing generated may be tracked.

Compiled bytecode was found committed to this repository. That is not a
cosmetic problem:

- **A `.pyc` can shadow the source it was built from.** While mutation-testing
  the language guard, a stale `__pycache__` entry made one run execute the
  *previous* mutation's bytecode. The report that came out of it was wrong,
  and it looked exactly like a real result
- A tracked `.pyc` differs per interpreter version, so every contributor
  produces a diff for a file nobody edited
- It is not reproducible from the source, which is the property this
  repository exists to hold

`.gitignore` alone would not have caught it: these files were already tracked,
and `.gitignore` has no effect on tracked files. So the check is on what git
actually tracks.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GENERATED = (".pyc", ".pyo", ".so", ".egg-info")
GENERATED_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")


def tracked_files() -> list[str]:
    r = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {r.stderr.strip()}")
    return [x for x in r.stdout.split("\0") if x]


class TestNothingGeneratedIsTracked(unittest.TestCase):
    def test_no_generated_file_is_tracked(self):
        files = tracked_files()
        bad = [f for f in files
               if f.endswith(GENERATED)
               or any(part in GENERATED_DIRS for part in Path(f).parts)]
        self.assertEqual(
            bad, [],
            "generated files are tracked; a stale one can shadow its source\n"
            + "\n".join(f"  - {x}" for x in bad)
            + "\nremove with: git rm -r --cached <path>")

    def test_the_check_actually_sees_the_tree(self):
        """Against an empty listing this would pass vacuously."""
        files = tracked_files()
        self.assertGreater(len(files), 5)
        self.assertIn("README.md", files)

    def test_the_detector_recognises_a_generated_path(self):
        """Guard against predicates that never match."""
        for sample in ("platforms/__pycache__/base.cpython-312.pyc",
                       "a/b/c.pyc", "__pycache__/x.py", "pkg.egg-info"):
            with self.subTest(sample=sample):
                self.assertTrue(
                    sample.endswith(GENERATED)
                    or any(p in GENERATED_DIRS for p in Path(sample).parts),
                    f"{sample} is not recognised as generated")

    def test_the_detector_leaves_real_sources_alone(self):
        for sample in ("platforms/base.py", "README.md", "tests/run-tests.sh",
                       "bin/contract-test.py"):
            with self.subTest(sample=sample):
                self.assertFalse(
                    sample.endswith(GENERATED)
                    or any(p in GENERATED_DIRS for p in Path(sample).parts),
                    f"{sample} is wrongly treated as generated")


class TestTheyAreAlsoIgnored(unittest.TestCase):
    """Untracking is not enough; they come straight back otherwise."""

    def test_gitignore_exists(self):
        self.assertTrue((ROOT / ".gitignore").is_file())

    def test_generated_paths_are_ignored_in_practice(self):
        """Ask git, rather than reading the patterns and guessing.

        The whole cache directory has to be ignored, not merely the `.pyc`
        inside it: a mutation that deleted the `__pycache__/` pattern survived
        when only a `.pyc` path was checked, because `*.py[cod]` still covered
        that one sample.
        """
        for sample in ("platforms/__pycache__/base.cpython-312.pyc",
                       "platforms/__pycache__/",
                       "platforms/__pycache__/whatever-else",
                       "bin/x.pyc", ".mypy_cache/x", "pkg.egg-info/PKG-INFO"):
            with self.subTest(sample=sample):
                r = subprocess.run(["git", "check-ignore", "-q", sample],
                                   cwd=ROOT)
                self.assertEqual(r.returncode, 0, f"{sample} is not ignored")

    def test_real_sources_are_not_ignored(self):
        """An ignore rule wide enough to swallow the source would pass above."""
        for sample in ("README.md", "platforms/base.py",
                       "tests/test_repo_hygiene.py"):
            with self.subTest(sample=sample):
                r = subprocess.run(["git", "check-ignore", "-q", sample],
                                   cwd=ROOT)
                self.assertEqual(r.returncode, 1, f"{sample} is ignored")


if __name__ == "__main__":
    unittest.main()
