#!/usr/bin/env python3
"""No shared machinery may name an individual method.

**This is a mechanism against a mistake made three times**, each time the
same shape: a rule written for the things that existed, which then governed
only some of the things it was supposed to.

- the CI workflow installed the first method's lock by name, so when a second
  method arrived twelve of its tests skipped and the job still reported
  success
- the lock checks ran over `methods/*/` and missed `requirements-tools.lock.txt`
- `tests/test_ci.py` held a `LOCKS` tuple naming one method

Every one was found by adding the *second* thing, which is far too late: a
list looks correct while there is only one item, and the whole point of this
repository is that there will eventually be thirty-seven.

So shared machinery -- the workflow, the tools, the tests that are about all
methods -- must **discover** methods rather than name them. A method's own
directory and its own test file may name it, because those are not shared.

Prose is exempt: the README has to be able to say which methods are ported.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _repo_files import repository_files   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"

# Where naming a method is legitimate.
#
# `_reference` is exempt everywhere: it is not a method under study but the
# known-good adapter the contract tests run against, so the shared tests that
# exercise the chain necessarily name it.
EXEMPT_METHODS = {"_reference"}
PROSE_SUFFIXES = (".md", ".rst", ".txt", ".json", ".lock")


def method_names() -> list[str]:
    if not METHODS.is_dir():
        return []
    return sorted(p.name for p in METHODS.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def scanned_files() -> list[Path]:
    """Tracked, plus untracked and not ignored -- `_repo_files` decides.

    **This was a second implementation of that rule and it diverged.** It
    raised where git was unavailable, so three tests here errored inside the
    container image -- which ships with no `.git` and no git binary by design
    -- while the language guard, asking the identical question through its
    own copy of the code, degraded to a filesystem walk and passed.

    The answer was not to skip this scan where git is missing. A guard that
    stops running in the one environment that most resembles what we publish
    is worth less than no guard, and answering a red CI by testing less is
    how a suite rots. The answer was for there to be one implementation.
    """
    return repository_files(ROOT)


def may_name(path: Path, method: str) -> bool:
    """Whether `path` is allowed to mention `method`."""
    rel = path.relative_to(ROOT)
    if rel.suffix in PROSE_SUFFIXES:
        return True                       # documentation may say anything
    if rel.name == ".gitmodules":
        return True                       # git config declaring submodule paths
    if rel.parts[:2] == ("methods", method):
        return True                       # the method's own files
    if rel.parts[0] == "tests" and rel.name.startswith(f"test_method_"):
        return True                       # a method's own test file
    return False


def names_method(text: str, method: str) -> bool:
    """Whether `text` hard-codes `method` as a name.

    A method is hard-coded in exactly two shapes, and they are the shapes the
    recorded mistakes took: a **directory path** `methods/<name>` (CI once
    installed `methods/<m>/requirements.lock.txt` by name) or a **quoted
    identifier** `"<name>"`/`'<name>'` (a `LOCKS` tuple naming one method).
    Those are matched; nothing else is.

    **Why not a bare word-boundary match.** A three-letter name -- `mar`, `var`
    -- is not only a substring of ordinary words (`primary`, `variance`, caught
    by a boundary) but also a whole token in text that has nothing to do with
    the method: `/var/lib/apt`, a docstring saying "unlike mar", the submodule
    path `third_party/var`. Matching every bare token flagged all of those --
    the too-wide-scope mistake this project keeps a list of. A path or a quoted
    literal is how a method is actually wired in for discovery; a mention in
    prose or an unrelated path is not.
    """
    m = re.escape(method)
    return bool(re.search(r"methods/" + m + r"(?![\w-])", text)
                or re.search(r"['\"]" + m + r"['\"]", text))


class TestSharedMachineryDiscoversMethods(unittest.TestCase):
    def test_there_is_more_than_one_method(self):
        """With one method this file cannot fail, and proves nothing."""
        self.assertGreater(len(method_names()), 1)

    def test_no_shared_file_names_a_method(self):
        offenders = []
        for path in scanned_files():
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for method in method_names():
                if method in EXEMPT_METHODS or may_name(path, method):
                    continue
                if names_method(text, method):
                    offenders.append(
                        f"{path.relative_to(ROOT)} names {method}")
        self.assertEqual(
            offenders, [],
            "shared machinery must discover methods, not name them -- a list "
            "looks correct until the next method arrives:\n"
            + "\n".join(f"  - {x}" for x in offenders))

    def test_an_untracked_file_is_scanned_too(self):
        """The gap the commit hook found: this guard passed while the
        offending file sat untracked, because it listed tracked files only."""
        probe = ROOT / "tests" / "_scan_probe_tmp.py"
        probe.write_text("# temporary\n", encoding="utf-8")
        self.addCleanup(probe.unlink, missing_ok=True)
        self.assertIn(probe, scanned_files())

    def test_the_scan_reads_something(self):
        self.assertGreater(len(scanned_files()), 20)

    def test_a_shared_file_naming_a_method_would_be_caught(self):
        """The detector, checked against a case it must reject."""
        real = method_names()[0]
        self.assertFalse(may_name(ROOT / ".github" / "workflows" / "x.yml",
                                  real))
        self.assertFalse(may_name(ROOT / "bin" / "launch.py", real))

    def test_a_methods_own_files_may_name_it(self):
        real = method_names()[0]
        self.assertTrue(may_name(METHODS / real / "adapter" / "__init__.py",
                                 real))
        self.assertTrue(may_name(ROOT / "tests" / f"test_method_{real}.py",
                                 real))

    def test_prose_may_name_a_method(self):
        """The README has to be able to say which methods are ported."""
        self.assertTrue(may_name(ROOT / "README.md", method_names()[0]))

    def test_the_matcher_catches_paths_and_quoted_names(self):
        """The detector against the shapes a hard-coding takes -- a fake name is
        used so this file does not itself name a real method."""
        self.assertTrue(names_method('LOCKS = ("cat",)', "cat"))       # quoted
        self.assertTrue(names_method("-r methods/cat/lock.txt", "cat"))  # path
        self.assertTrue(names_method("if m == 'cat':", "cat"))          # quoted

    def test_the_matcher_ignores_substrings_prose_and_unrelated_paths(self):
        """The false positives a three-letter name invites: a substring, a bare
        mention in prose, an unrelated system path, and the submodule path --
        none of which wires the method in for discovery."""
        self.assertFalse(names_method("concatenate scatter category", "cat"))
        self.assertFalse(names_method("# faster than cat, but simpler", "cat"))
        self.assertFalse(names_method("rm -rf /cat/lib/apt/lists", "cat"))
        self.assertFalse(names_method("path = third_party/cat", "cat"))


if __name__ == "__main__":
    unittest.main()
