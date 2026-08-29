#!/usr/bin/env python3
"""Specification for mutate.py.

**The tool exists because the ad-hoc versions of it lied twice**, and a
mutation report that lies is worse than no mutation report: it converts "the
tests might be vacuous" into "the tests were checked and are fine".

Both lies are pinned here, because they are the reason for the tool:

- **an anchor that does not match must be a hard error**, never a surviving
  mutant. Reporting "SURVIVED" for a mutation that was never applied reads as
  "the tests missed this" when nothing was changed
- **bytecode must never be reused.** One run executed the previous mutation's
  `__pycache__` and produced a plausible, wrong report

And a third that neither ad-hoc version handled: an anchor appearing more than
once mutates whichever comes first, which is a coin toss dressed as a result.
"""

from __future__ import annotations

import importlib.util
import json
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


mutate = load("mutate", "mutate.py")


class TestApplyingAMutation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="mutspec-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "sample.py").write_text("x = 1\ny = 2\n", encoding="utf-8")

    def spec(self, **over) -> dict:
        s = {"label": "t", "file": "sample.py", "old": "x = 1", "new": "x = 9",
             "tests": ["t"]}
        s.update(over)
        return s

    def test_a_matching_anchor_is_applied(self):
        mutate.apply_one(self.tmp, self.spec())
        self.assertIn("x = 9", (self.tmp / "sample.py").read_text())

    def test_an_absent_anchor_is_an_error_not_a_survivor(self):
        """**The first lie.** A mutation that was never applied must never be
        reported as one the tests failed to catch."""
        with self.assertRaises(mutate.MutationError) as e:
            mutate.apply_one(self.tmp, self.spec(old="not here"))
        self.assertIn("anchor", str(e.exception))
        self.assertEqual((self.tmp / "sample.py").read_text(), "x = 1\ny = 2\n")

    def test_an_ambiguous_anchor_is_refused(self):
        """Mutating whichever comes first is a coin toss dressed as a result.

        It actually happened: an anchor matched both metric filters and the
        wrong one was mutated, so the report described a test that was never
        exercised.
        """
        (self.tmp / "sample.py").write_text("a = 0\na = 0\n", encoding="utf-8")
        with self.assertRaises(mutate.MutationError) as e:
            mutate.apply_one(self.tmp, self.spec(old="a = 0", new="a = 1"))
        self.assertIn("twice", str(e.exception).replace("2 times", "twice"))

    def test_an_ambiguous_anchor_may_be_taken_deliberately(self):
        (self.tmp / "sample.py").write_text("a = 0\na = 0\n", encoding="utf-8")
        mutate.apply_one(self.tmp, self.spec(old="a = 0", new="a = 1",
                                             all=True))
        self.assertEqual((self.tmp / "sample.py").read_text(), "a = 1\na = 1\n")

    def test_a_target_outside_the_work_tree_is_refused(self):
        """**Found by running the tool on itself.**

        Pointing the mutation at the repository and restoring it afterwards
        leaves the final state correct, so "the tree is unchanged when it
        finishes" cannot see it -- while the repository is broken for the
        duration, and a crash makes that permanent.
        """
        with self.assertRaises(mutate.MutationError) as e:
            mutate.apply_one(self.tmp, self.spec(file="../escape.py"))
        self.assertIn("outside the work tree", str(e.exception))

    def test_the_ordinary_case_is_still_allowed(self):
        """The guard must not refuse everything."""
        self.assertEqual(mutate.target_in(self.tmp, self.spec()),
                         (self.tmp / "sample.py").resolve())

    def test_a_missing_file_is_an_error(self):
        with self.assertRaises(mutate.MutationError) as e:
            mutate.apply_one(self.tmp, self.spec(file="absent.py"))
        self.assertIn("absent.py", str(e.exception))


class TestBytecodeIsNeverReused(unittest.TestCase):
    """**The second lie.** A stale `__pycache__` ran the previous mutation."""

    def test_the_tests_are_run_with_bytecode_writing_off(self):
        src = (BIN / "mutate.py").read_text()
        self.assertIn("PYTHONDONTWRITEBYTECODE", src)
        self.assertIn('"-B"', src)

    def test_the_copied_tree_carries_no_bytecode(self):
        """Copying `__pycache__` into the work tree would reintroduce it."""
        self.assertIn("__pycache__", mutate._ignore("x", ["__pycache__"]))


class TestRunningTheWholeThing(unittest.TestCase):
    """End to end, against this repository."""

    def run_spec(self, spec: dict):
        p = Path(tempfile.mkdtemp(prefix="mutrun-"))
        self.addCleanup(shutil.rmtree, p, ignore_errors=True)
        f = p / "spec.json"
        f.write_text(json.dumps(spec), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(BIN / "mutate.py"), "--spec", str(f)],
            capture_output=True, text=True)

    def test_a_mutation_the_tests_catch_is_reported_killed(self):
        r = self.run_spec({"targets": [{
            "label": "break the canonical form",
            "file": "bin/resolve-config.py",
            "old": "sort_keys=True", "new": "sort_keys=False",
            "tests": ["tests.test_resolve_config.TestCanonicalForm"]}]})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("killed by", r.stdout)

    def test_a_mutation_nothing_catches_is_reported_survived(self):
        """Mutating something the named tests cannot see."""
        r = self.run_spec({"targets": [{
            "label": "change a message nobody reads",
            "file": "bin/resolve-config.py",
            "old": "  wrote ", "new": "  written ",
            "tests": ["tests.test_resolve_config.TestCanonicalForm"]}]})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SURVIVED", r.stdout)

    def test_a_bad_anchor_fails_the_run_rather_than_surviving(self):
        r = self.run_spec({"targets": [{
            "label": "anchor that is not there",
            "file": "bin/resolve-config.py",
            "old": "this string does not appear", "new": "x",
            "tests": ["tests.test_resolve_config.TestCanonicalForm"]}]})
        self.assertEqual(r.returncode, 2, "a bad anchor was not an error")
        self.assertNotIn("SURVIVED", r.stdout + r.stderr)
        self.assertIn("anchor", r.stdout + r.stderr)

    def test_an_empty_spec_is_refused(self):
        r = self.run_spec({"targets": []})
        self.assertEqual(r.returncode, 2)

    def test_the_tree_is_left_alone(self):
        """It mutates a copy. Editing the real tree would be catastrophic in
        a way no test could undo."""
        before = (ROOT / "bin" / "resolve-config.py").read_text()
        self.run_spec({"targets": [{
            "label": "x", "file": "bin/resolve-config.py",
            "old": "sort_keys=True", "new": "sort_keys=False",
            "tests": ["tests.test_resolve_config.TestCanonicalForm"]}]})
        self.assertEqual((ROOT / "bin" / "resolve-config.py").read_text(),
                         before)


class TestABrokenBaselineStopsEverything(unittest.TestCase):
    def test_it_refuses_when_the_tests_already_fail(self):
        """Mutation results mean nothing over a red suite, and reporting
        every mutant as killed would be the most flattering possible lie."""
        p = Path(tempfile.mkdtemp(prefix="mutbase-"))
        self.addCleanup(shutil.rmtree, p, ignore_errors=True)
        f = p / "spec.json"
        f.write_text(json.dumps({"targets": [{
            "label": "x", "file": "bin/resolve-config.py",
            "old": "sort_keys=True", "new": "sort_keys=False",
            "tests": ["tests.does_not_exist"]}]}), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(BIN / "mutate.py"), "--spec", str(f)],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("before anything is mutated", r.stdout + r.stderr)


class TestTheCopyLeavesEnvironmentsOut(unittest.TestCase):
    """The tree is copied for every mutation, so what it copies matters.

    A per-method environment can be several gigabytes. Copied once per mutation
    it turned each mutate test into twelve seconds of I/O. It is left out **by
    its PEP 405 marker, not its name**: a list of names is the listing mistake
    this repository keeps making, and `tests/_repo_files.py` already excludes
    environments the same way, by `pyvenv.cfg`.
    """

    def _env(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="mutate-ign-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def test_a_directory_that_declares_itself_an_environment_is_skipped(self):
        tmp = self._env()
        (tmp / ".venvs" / "env").mkdir(parents=True)
        (tmp / ".venvs" / "env" / "pyvenv.cfg").write_text("home = /x\n")
        self.assertIn("env", mutate._ignore(str(tmp / ".venvs"), ["env"]))

    def test_ordinary_source_is_not_skipped(self):
        """The negative control: without the marker, a directory is copied."""
        tmp = self._env()
        (tmp / "adapterlib").mkdir()
        self.assertNotIn("adapterlib",
                         mutate._ignore(str(tmp), ["adapterlib"]))

    def test_the_named_patterns_still_apply(self):
        """The marker is added to the old exclusions, it does not replace
        them: .git and __pycache__ are still left out."""
        skipped = mutate._ignore(str(ROOT), [".git", "__pycache__", "bin"])
        self.assertIn(".git", skipped)
        self.assertIn("__pycache__", skipped)
        self.assertNotIn("bin", skipped)


class TestTheCopySurvivesADanglingSymlink(unittest.TestCase):
    """The tree copy carries symlinks across as symlinks.

    A vendored submodule can hold a dangling symlink: fairseq's kaldi example
    under `third_party/unilm` points `st/utils` at a kaldi target that is not
    checked out. copytree follows symlinks by default and dies on the missing
    target, and the whole mutation run then prints that OSError as if it were a
    result. This is the positive control that the copy does not.
    """

    def _dir(self, prefix: str) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def test_a_dangling_symlink_does_not_break_the_copy(self):
        src = self._dir("mutate-sym-src-")
        (src / "real.py").write_text("x = 1\n", encoding="utf-8")
        (src / "dangling").symlink_to("does-not-exist-anywhere")
        self.assertFalse((src / "dangling").exists(),  # the target is missing
                         "the fixture must be a genuinely dangling symlink")
        dst = self._dir("mutate-sym-dst-")
        mutate._copy_tree(src, dst)                     # must not raise
        self.assertEqual((dst / "real.py").read_text(encoding="utf-8"),
                         "x = 1\n")
        self.assertTrue((dst / "dangling").is_symlink(),
                        "the symlink is carried across as a symlink")


if __name__ == "__main__":
    unittest.main()
