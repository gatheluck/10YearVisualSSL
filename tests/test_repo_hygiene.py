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
import sys
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _checkout import has_git, needs_checkout   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

GENERATED = (".pyc", ".pyo", ".so", ".egg-info")
GENERATED_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")


def tracked_files() -> list[str]:
    r = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {r.stderr.strip()}")
    return [x for x in r.stdout.split("\0") if x]


class TestTheCheckoutGateIsNotAlwaysOn(unittest.TestCase):
    """The positive control for the skip that was just introduced.

    Several classes here assert properties of the *repository* and are
    skipped where there is no repository -- inside the container image, which
    carries neither `.git` nor `git` by design. A gate like that is one wrong
    condition away from skipping everywhere and reporting success, so this
    asserts that **in a checkout it is off**.

    Deliberately not itself gated.
    """

    @unittest.skipUnless((ROOT / ".git").exists() and has_git(),
                         "no .git or no git here, so the gate is on for a "
                         "reason and there is nothing to be wrong about")
    def test_where_there_is_a_git_directory_and_git_the_gate_is_off(self):
        """Premised on a usable checkout, checked without `is_checkout`.

        Asserting "this is a checkout" unconditionally would fail inside the
        image, where it is simply not applicable. Asserting it *where a .git
        and a git both exist* is the real claim: the gate must not be on in a
        checkout, or every repository test is skipping and nothing says so.

        **`and has_git()` was missing and the premise was therefore false.**
        A checkout on a machine with no git installed -- a minimal container
        with the tree mounted into it -- has a `.git`, cannot run git, and is
        correctly gated; this then asserted that it was not, and failed. The
        container job cannot see that combination, because it has no `.git`
        either. Running the whole suite with git removed from `PATH` did.
        """
        from _checkout import is_checkout
        self.assertTrue(is_checkout(ROOT),
                        "there is a .git and a git here but the gate is on, "
                        "so the repository tests are all skipping silently")

    def test_without_git_the_gate_is_on_whatever_else_is_true(self):
        """The case just carved out of the premise above, asserted rather
        than merely excluded. A `.git` that cannot be read is not a checkout,
        and claiming otherwise would send every repository test into a git
        call that raises instead of skipping."""
        from _checkout import is_checkout
        import _checkout
        real, _checkout.has_git = _checkout.has_git, lambda: False
        try:
            self.assertFalse(is_checkout(ROOT))
        finally:
            _checkout.has_git = real
        self.assertEqual(is_checkout(ROOT), (ROOT / ".git").exists()
                         and has_git(), "restoring has_git did not restore "
                         "the answer, so the test above leaked")

    def test_a_directory_that_is_not_a_repository_is_recognised(self):
        """The other half: the condition must actually be able to be false."""
        import shutil as _shutil, tempfile
        from _checkout import is_checkout
        d = Path(tempfile.mkdtemp(prefix="notarepo-"))
        self.addCleanup(_shutil.rmtree, d, ignore_errors=True)
        self.assertFalse(is_checkout(d))


@needs_checkout
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


@needs_checkout
class TestTheReadmeLayoutIsComplete(unittest.TestCase):
    """Everything that exists is listed, not merely everything listed exists.

    An audit checked the layout one way round -- every entry marked `exists`
    was really there -- and passed while **two tools were missing from it
    entirely**. Both had been added by an edit whose anchor silently failed to
    match, so nothing was written and nothing complained.

    This checks the other direction, which is the one that rots.
    """

    def layout(self) -> str:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        start = text.index("## Repository layout")
        return text[start:text.index("```", text.index("```", start) + 3)]

    def entries(self) -> set[str]:
        """The names the tree draws, matched whole.

        Not a substring search over the block: `test_launch.py` contains
        `launch.py`, so deleting the tool's own line left the check passing.
        """
        import re
        out = set()
        for line in self.layout().splitlines():
            m = re.search(r"(?:├──|└──)\s+(\S+)", line)
            if m:
                out.add(m.group(1).rstrip("/"))
        return out

    def test_every_tool_is_in_the_layout(self):
        tools = sorted(p.name for p in (ROOT / "bin").glob("*.py"))
        self.assertTrue(tools, "no tools found to check")
        drawn = self.entries()
        for name in tools:
            with self.subTest(tool=name):
                self.assertIn(name, drawn, f"bin/{name} is not in the layout")

    def test_the_entry_reader_finds_the_tree(self):
        """Against an empty set every check above passes vacuously."""
        self.assertGreater(len(self.entries()), 20)
        self.assertIn("bin", self.entries())

    def test_every_top_level_directory_is_in_the_layout(self):
        drawn = self.entries()
        tracked = {f.split("/")[0] for f in tracked_files() if "/" in f}
        for name in sorted(tracked):
            if name.startswith("."):
                continue          # dot-directories are covered where relevant
            with self.subTest(directory=name):
                self.assertIn(name, drawn, f"{name}/ is not in the layout")

    def test_the_layout_block_was_actually_found(self):
        """Against an empty block every check above passes vacuously."""
        self.assertGreater(len(self.layout().splitlines()), 20)


@needs_checkout
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
                       "bin/x.pyc", ".mypy_cache/x", "pkg.egg-info/PKG-INFO",
                       # The README tells people to create this here, and
                       # nothing ignored it: `git status` offered a whole
                       # virtual environment as untracked, ready to commit.
                       ".venv/", ".venv/lib/python3.12/site-packages/x.py"):
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
