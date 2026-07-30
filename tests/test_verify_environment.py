#!/usr/bin/env python3
"""Specification for verify-environment.py.

**Installing from a lock and having installed the lock are different claims.**
Only the second is worth anything, and nothing checked it. The documented
check was a shell one-liner comparing `pip freeze` against a single lock file
while the install used two, so it reported a difference that was not one --
which is how a correct environment came back looking wrong.

A one-liner people have to assemble correctly is not a mechanism. This is.

It answers two questions with the same comparison:

- **Is this environment the locked one?** Run it in the environment
- **Did that run use the locked environment?** Point it at the
  `run_manifest.json`, which records every installed package

The second is the one that matters after the fact. A result whose environment
was never checked against the lock is a result nobody can rebuild.
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


ve = load("verify_environment", "verify-environment.py")


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vetest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def lock(self, name: str, entries: dict) -> Path:
        p = self.tmp / name
        body = []
        for pkg, ver in entries.items():
            body.append(f"# {pkg}-{ver}-py3-none-any.whl")
            body.append(f"{pkg}=={ver} \\")
            body.append(f"    --hash=sha256:{'0' * 64}")
            body.append("")
        p.write_text("\n".join(body), encoding="utf-8")
        return p

    def manifest(self, packages: dict) -> Path:
        p = self.tmp / "run_manifest.json"
        p.write_text(json.dumps({"env": {"packages": packages}}),
                     encoding="utf-8")
        return p


class TestReadingALock(Base):
    def test_versions_are_read_through_hash_continuations(self):
        got = ve.read_locks([self.lock("a.txt", {"torch": "2.0.1"})])
        self.assertEqual(got, {"torch": "2.0.1"})

    def test_a_version_on_a_continuation_line_still_parses(self):
        """Folding continuations is not redundant, though it looks it.

        Hash lines begin with `-` and are dropped by that filter alone, so
        removing the fold broke nothing -- until a requirement puts its
        specifier on the next line. pip joins continuations before parsing,
        and without the fold this reads as `torch` with no version and is
        refused: a correct lock reported as unpinned.
        """
        p = self.tmp / "wrapped.txt"
        p.write_text("torch \\\n    ==2.0.1 \\\n"
                     f"    --hash=sha256:{'0' * 64}\n", encoding="utf-8")
        self.assertEqual(ve.read_locks([p]), {"torch": "2.0.1"})

    def test_several_locks_are_merged(self):
        """The install takes more than one file, so the check must too.

        Comparing against one of them was the original mistake: PyYAML came
        from the tooling lock and looked like an unexplained extra.
        """
        got = ve.read_locks([self.lock("a.txt", {"torch": "2.0.1"}),
                             self.lock("b.txt", {"PyYAML": "6.0.3"})])
        self.assertEqual(got, {"torch": "2.0.1", "pyyaml": "6.0.3"})

    def test_names_compare_without_regard_to_case_or_separators(self):
        """`PyYAML`, `pyyaml` and `py-yaml` are one distribution."""
        got = ve.read_locks([self.lock("a.txt", {"typing_extensions": "4.1"})])
        self.assertIn("typing-extensions", got)

    def test_two_locks_disagreeing_on_a_version_is_refused(self):
        """Whichever won would be silent, and one of them would be wrong."""
        with self.assertRaises(ve.EnvironmentMismatch) as e:
            ve.read_locks([self.lock("a.txt", {"torch": "2.0.1"}),
                           self.lock("b.txt", {"torch": "2.1.0"})])
        self.assertIn("torch", str(e.exception))

    def test_an_unpinned_entry_is_refused(self):
        p = self.tmp / "loose.txt"
        p.write_text("torch>=2.0.1\n", encoding="utf-8")
        with self.assertRaises(ve.EnvironmentMismatch) as e:
            ve.read_locks([p])
        self.assertIn("torch", str(e.exception))

    def test_a_missing_lock_is_refused_by_name(self):
        with self.assertRaises(ve.EnvironmentMismatch) as e:
            ve.read_locks([self.tmp / "absent.txt"])
        self.assertIn("absent.txt", str(e.exception))


class TestComparing(Base):
    def test_an_exact_match_passes(self):
        rc, rep = ve.compare({"torch": "2.0.1"}, {"torch": "2.0.1"})
        self.assertEqual(rc, 0)
        self.assertEqual(rep["differences"], [])

    def test_a_missing_package_is_named(self):
        rc, rep = ve.compare({"torch": "2.0.1", "numpy": "1.0"},
                             {"torch": "2.0.1"})
        self.assertEqual(rc, 1)
        self.assertIn("missing", [d["kind"] for d in rep["differences"]])
        self.assertIn("numpy", str(rep["differences"]))

    def test_an_extra_package_is_named(self):
        """**Not a warning.** Something is installed that the lock does not
        describe, so the environment is not the locked one and cannot be
        rebuilt from it."""
        rc, rep = ve.compare({"torch": "2.0.1"},
                             {"torch": "2.0.1", "requests": "2.0"})
        self.assertEqual(rc, 1)
        self.assertIn("unexpected", [d["kind"] for d in rep["differences"]])
        self.assertIn("requests", str(rep["differences"]))

    def test_a_version_difference_is_named_with_both_versions(self):
        rc, rep = ve.compare({"torch": "2.0.1"}, {"torch": "2.1.0"})
        self.assertEqual(rc, 1)
        detail = str(rep["differences"])
        self.assertIn("2.0.1", detail)
        self.assertIn("2.1.0", detail)

    def test_case_and_separators_do_not_count_as_a_difference(self):
        rc, _ = ve.compare({"typing-extensions": "4.1"},
                           {"typing_extensions": "4.1"})
        self.assertEqual(rc, 0)

    def test_differences_are_reported_in_a_stable_order(self):
        rc, rep = ve.compare({"a": "1", "b": "1", "c": "1"}, {})
        names = [d["package"] for d in rep["differences"]]
        self.assertEqual(names, sorted(names))


class TestWhatVenvSeeds(Base):
    """`pip` is put there by `python -m venv`, not by any lock.

    Measured: a bare venv on this interpreter contains exactly `pip` and
    nothing else. `setuptools` is *not* seeded -- it is a real torch
    requirement (`setuptools>=77.0.3`) and belongs in the lock, so it is
    compared like anything else.

    **Ignoring it is not the same as hiding it.** A skip that leaves no trace
    is a silent failure (DESIGN 2.4), so it is reported as ignored, with the
    reason, every time.
    """

    def test_pip_alone_is_not_counted_as_a_difference(self):
        rc, rep = ve.compare({"torch": "2.0.1"},
                             {"torch": "2.0.1", "pip": "25.0.1"})
        self.assertEqual(rc, 0, rep["differences"])

    def test_but_it_is_reported_as_ignored_with_a_reason(self):
        _, rep = ve.compare({"torch": "2.0.1"},
                            {"torch": "2.0.1", "pip": "25.0.1"})
        ignored = rep["ignored"]
        self.assertEqual([i["package"] for i in ignored], ["pip"])
        self.assertTrue(ignored[0]["reason"], "ignored without a reason")
        self.assertEqual(ignored[0]["version"], "25.0.1")

    def test_nothing_is_ignored_when_nothing_needs_to_be(self):
        _, rep = ve.compare({"torch": "2.0.1"}, {"torch": "2.0.1"})
        self.assertEqual(rep["ignored"], [])

    def test_a_lock_that_names_pip_is_honoured(self):
        """The exemption is for a package no lock mentions. Once a lock takes
        responsibility for it, it is checked like anything else."""
        rc, rep = ve.compare({"pip": "25.0.1"}, {"pip": "99.0.0"})
        self.assertEqual(rc, 1)
        self.assertEqual(rep["ignored"], [])
        self.assertIn("pip", str(rep["differences"]))

    def test_the_exemption_covers_only_what_venv_seeds(self):
        """Widening this would let real packages through unnoticed."""
        self.assertEqual(ve.SEEDED_BY_VENV, frozenset({"pip"}))

    def test_the_command_line_shows_what_it_ignored(self):
        lock = self.lock("a.txt", {"torch": "2.0.1"})
        man = self.manifest({"torch": "2.0.1", "pip": "25.0.1"})
        r = self.run_tool_here("--lock", lock, "--manifest", man)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("pip", r.stdout, "the skip left no trace in the output")

    def run_tool_here(self, *args):
        return subprocess.run(
            [sys.executable, str(BIN / "verify-environment.py"),
             *map(str, args)], capture_output=True, text=True)


class TestReadingAManifest(Base):
    def test_the_recorded_environment_is_read(self):
        got = ve.read_manifest(self.manifest({"torch": "2.0.1"}))
        self.assertEqual(got, {"torch": "2.0.1"})

    def test_a_manifest_without_a_package_record_is_refused(self):
        """Older manifests recorded only python and hostname. Treating that as
        an empty environment would report every package as missing and bury
        the real problem, which is that the run cannot be checked at all."""
        p = self.tmp / "old.json"
        p.write_text(json.dumps({"env": {"python": "3.10.13"}}),
                     encoding="utf-8")
        with self.assertRaises(ve.EnvironmentMismatch) as e:
            ve.read_manifest(p)
        self.assertIn("packages", str(e.exception))

    def test_an_unparsable_manifest_is_refused(self):
        p = self.tmp / "bad.json"
        p.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(ve.EnvironmentMismatch):
            ve.read_manifest(p)


class TestTheCommandLine(Base):
    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, str(BIN / "verify-environment.py"),
             *map(str, args)], capture_output=True, text=True)

    def test_a_matching_manifest_exits_zero(self):
        lock = self.lock("a.txt", {"torch": "2.0.1"})
        man = self.manifest({"torch": "2.0.1"})
        r = self.run_tool("--lock", lock, "--manifest", man)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_mismatching_manifest_exits_nonzero_and_says_what(self):
        lock = self.lock("a.txt", {"torch": "2.0.1"})
        man = self.manifest({"torch": "9.9.9"})
        r = self.run_tool("--lock", lock, "--manifest", man)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("torch", r.stdout + r.stderr)

    def test_two_locks_together_match_what_the_install_produced(self):
        """The case the wrong one-liner got wrong."""
        a = self.lock("a.txt", {"torch": "2.0.1"})
        b = self.lock("b.txt", {"PyYAML": "6.0.3"})
        man = self.manifest({"torch": "2.0.1", "PyYAML": "6.0.3"})
        r = self.run_tool("--lock", a, "--lock", b, "--manifest", man)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_one_lock_alone_reports_the_other_half_as_unexpected(self):
        """Confirms the tool would have caught the mistake, rather than
        quietly agreeing with it."""
        a = self.lock("a.txt", {"torch": "2.0.1"})
        man = self.manifest({"torch": "2.0.1", "PyYAML": "6.0.3"})
        r = self.run_tool("--lock", a, "--manifest", man)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("PyYAML", r.stdout + r.stderr)

    def test_without_a_manifest_it_checks_this_interpreter(self):
        """Run inside the environment, it answers the other question."""
        r = self.run_tool("--lock", self.lock("a.txt", {"torch": "0.0.0"}))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("torch", r.stdout + r.stderr)

    def test_a_refusal_is_reported_rather_than_crashing(self):
        r = self.run_tool("--lock", self.tmp / "absent.txt")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("absent.txt", r.stdout + r.stderr)

    def test_json_output_can_be_written(self):
        lock = self.lock("a.txt", {"torch": "2.0.1"})
        man = self.manifest({"torch": "9.9.9"})
        out = self.tmp / "report.json"
        self.run_tool("--lock", lock, "--manifest", man, "--json", out)
        rep = json.loads(out.read_text())
        self.assertTrue(rep["differences"])


class TestAgainstTheRealLocks(unittest.TestCase):
    """The repository's own lock files must be readable by this tool."""

    LOCKS = [ROOT / "methods" / "1_context_prediction"
             / "requirements.lock.txt",
             ROOT / "requirements-tools.lock.txt"]

    def test_they_parse(self):
        got = ve.read_locks(self.LOCKS)
        self.assertIn("torch", got)
        self.assertIn("pyyaml", got)

    def test_the_closure_is_larger_than_the_direct_requirements(self):
        """Guards against a parser that silently reads almost nothing."""
        self.assertGreater(len(ve.read_locks(self.LOCKS)), 10)


if __name__ == "__main__":
    unittest.main()
