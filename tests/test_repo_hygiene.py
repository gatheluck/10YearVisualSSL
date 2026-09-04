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

import shutil
import subprocess
import sys
import tempfile
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


# -- submodules must not leak generated content into the superproject --------
#
# A pinned submodule whose upstream ships no `.gitignore` (franca and mar both
# shipped none) turns every import into `git status` noise -- `modified:
# third_party/x (untracked content)` that never clears, because the
# superproject's own .gitignore and `git clean` do not reach inside a
# submodule. Either the submodule's git ignores its bytecode, or .gitmodules
# must tell the superproject to (`ignore = untracked`). Asked of git, not
# guessed from the patterns.

SILENCING_IGNORE_VALUES = ("untracked", "dirty", "all")


def submodules(gitmodules: Path | None = None) -> list[tuple[str, str]]:
    """(name, path) for every submodule declared in .gitmodules."""
    gm = gitmodules or (ROOT / ".gitmodules")
    if not gm.is_file():
        return []
    r = subprocess.run(["git", "config", "-f", str(gm),
                        "--get-regexp", r"submodule\..*\.path"],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        key, _, path = line.partition(" ")
        name = key[len("submodule."):-len(".path")]
        out.append((name, path))
    return out


def upstream_ignores_pyc(subdir: Path) -> bool:
    """Whether the submodule's own git would ignore built bytecode.

    Ask git (check-ignore), rather than reading .gitignore and guessing: it
    accounts for .git/info/exclude and any core.excludesfile too.
    """
    r = subprocess.run(["git", "-C", str(subdir), "check-ignore", "-q",
                        "a/__pycache__/x.cpython-312.pyc"],
                       capture_output=True)
    return r.returncode == 0


def gitmodules_ignore(name: str, gitmodules: Path | None = None) -> str | None:
    gm = gitmodules or (ROOT / ".gitmodules")
    r = subprocess.run(["git", "config", "-f", str(gm),
                        "--get", f"submodule.{name}.ignore"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def leaks_generated_content(name: str, subdir: Path,
                            gitmodules: Path | None = None) -> bool:
    """True when this checked-out submodule would report bytecode as untracked
    content: it ignores nothing itself and .gitmodules does not silence it."""
    return (not upstream_ignores_pyc(subdir)
            and gitmodules_ignore(name, gitmodules) not in
            SILENCING_IGNORE_VALUES)


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


@needs_checkout
class TestSubmodulesDoNotLeakGeneratedContent(unittest.TestCase):
    """A pinned submodule must not report generated bytecode as untracked
    content forever.

    franca and mar shipped no `.gitignore`, so the first import that writes a
    `__pycache__/*.pyc` inside their working tree makes `git status` show
    `modified: third_party/<x> (untracked content)` -- a change that can never
    be committed away, because the superproject's `.gitignore` and `git clean`
    do not reach inside a submodule. Every one of the other eight submodules
    ignores its own bytecode. The fix for the two that do not is
    `ignore = untracked` in .gitmodules, and this guard stops a newly added
    submodule from reintroducing the noise.
    """

    def checked_out(self) -> list[tuple[str, str]]:
        return [(n, p) for n, p in submodules()
                if (ROOT / p / ".git").exists()]

    def test_no_checked_out_submodule_leaks_generated_content(self):
        offenders = [p for n, p in self.checked_out()
                     if leaks_generated_content(n, ROOT / p)]
        self.assertEqual(
            offenders, [],
            "these submodules would report generated bytecode as untracked "
            "content forever:\n"
            + "\n".join(f"  - {x}" for x in offenders)
            + "\nadd `ignore = untracked` under their .gitmodules entry")

    def test_it_actually_sees_the_submodules(self):
        """Against an empty listing the check above passes vacuously."""
        declared = submodules()
        self.assertGreater(len(declared), 5, "no submodules were discovered")
        self.assertIn("third_party/var", [p for _, p in declared],
                      "a known submodule is missing from the discovery")
        self.assertTrue(self.checked_out(),
                        "no submodule is checked out, so the guard is vacuous")

    def _init_repo(self, path: Path, gitignore: str | None = None) -> None:
        path.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(path)],
                       check=True, capture_output=True)
        if gitignore is not None:
            (path / ".gitignore").write_text(gitignore, encoding="utf-8")

    def test_the_upstream_ignore_detector_fires_and_stays_quiet(self):
        """`upstream_ignores_pyc` must distinguish a submodule that ignores its
        own bytecode from one that does not -- the property franca/mar lack."""
        base = Path(tempfile.mkdtemp(prefix="submodleak-"))
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)

        quiet = base / "quiet"
        self._init_repo(quiet, "__pycache__/\n*.pyc\n")
        self.assertTrue(upstream_ignores_pyc(quiet),
                        "a submodule that ignores pyc was read as not ignoring")

        noisy = base / "noisy"
        self._init_repo(noisy)               # no ignore rules at all, as franca
        self.assertFalse(upstream_ignores_pyc(noisy),
                         "a submodule with no ignore rules was read as ignoring")

    def test_the_gitmodules_override_silences_the_detector(self):
        """The second condition: `ignore = untracked` on an otherwise-leaking
        submodule clears it, and nothing else (e.g. a url line) does."""
        base = Path(tempfile.mkdtemp(prefix="submodcfg-"))
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        noisy = base / "noisy"
        self._init_repo(noisy)               # would leak on its own

        bare = base / "bare.gitmodules"
        bare.write_text('[submodule "x"]\n\tpath = x\n\turl = ./x\n',
                        encoding="utf-8")
        self.assertTrue(leaks_generated_content("x", noisy, gitmodules=bare),
                        "a leaking submodule with no ignore config was cleared")

        silenced = base / "silenced.gitmodules"
        silenced.write_text('[submodule "x"]\n\tpath = x\n\turl = ./x\n'
                            '\tignore = untracked\n', encoding="utf-8")
        self.assertFalse(leaks_generated_content("x", noisy,
                                                 gitmodules=silenced),
                         "`ignore = untracked` did not silence the detector")


if __name__ == "__main__":
    unittest.main()
