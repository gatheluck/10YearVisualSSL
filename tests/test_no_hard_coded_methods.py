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

    **Whole-token, not substring.** A three-letter name like `cat` is a
    substring of `concatenate`, `scatter` and `category`, and the first version
    --
    `method in text` -- flagged all three. That is the too-wide-scope mistake
    this project keeps a list of; the name is matched exactly.

    A reference to the pinned submodule `third_party/<method>` does not count:
    the submodule mechanism is itself shared machinery, so shared files name the
    upstream legitimately (a docstring explaining it, `run-ci-locally`
    materialising it). That is naming the *upstream*, not a method wired in for
    discovery, which is what this guard is about.
    """
    hunted = text.replace(f"third_party/{method}", "")
    return re.search(r"\b" + re.escape(method) + r"\b", hunted) is not None


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

    def test_the_matcher_is_whole_token_not_substring(self):
        """The detector against the cases that decide it: a bare name is
        caught, and the substrings that a short name hides inside are not. A
        fake name is used so this file does not itself name a real method."""
        self.assertTrue(names_method('m = "cat"', "cat"))
        self.assertTrue(names_method("open('methods/cat/x')", "cat"))
        self.assertFalse(names_method("concatenate scatter category", "cat"))

    def test_a_same_named_submodule_reference_is_not_naming_a_method(self):
        """`third_party/<name>` is the pinned upstream; naming it is not wiring
        the method in for discovery. The exemption is narrow, though: a bare
        token elsewhere in the same file is still caught."""
        self.assertFalse(names_method("path = 'third_party/cat'", "cat"))
        self.assertTrue(
            names_method("third_party/cat\nif m == 'cat':", "cat"),
            "the exemption must not blanket a file that also hard-codes it")


if __name__ == "__main__":
    unittest.main()
