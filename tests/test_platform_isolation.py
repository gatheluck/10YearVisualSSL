#!/usr/bin/env python3
"""Keep any dependence on a specific machine inside its own module.

**Running on a particular cluster is optional.** The core assumes none of
them. Support for one lives in `platforms/<name>/` as a loosely coupled
module and is reached only from there.

Writing that down is not enough. On the Capture side a strict rule stated in
three places was broken the same day, so this is enforced by a test.

What is held:

- **Machine-specific vocabulary appears only under `platforms/<name>/`**
- **A path that needs no cluster exists.** `platforms/local/` is always there
- **Nothing outside `platforms/` imports a specific backend.** Importing it
  means the core knows about it
- **The shared interface stays free of that vocabulary.** Once the interface
  is polluted, the separation is decorative
"""

from __future__ import annotations

import re
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLATFORMS = ROOT / "platforms"
ABCI_DIR = PLATFORMS / "abci"

# Vocabulary specific to one cluster, taken from real submission scripts
# recorded in the capture. Not guessed at. Extend it when more shows up.
ABCI_MARKERS = (
    "#PBS", "qsub", "rt_HF", "rt_HG", "rt_HC", "gag51492",
    "/groups/", "module load", "abci",
)

# Files to scan. The tests themselves and the prose describing this are out.
def _scan_targets() -> list[Path]:
    out: list[Path] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        parts = rel.parts
        if parts[0] in (".git", "docs", "tests", ".githooks"):
            continue
        if p.suffix not in (".py", ".sh", ".yaml", ".yml", ".toml", ".cfg"):
            continue
        out.append(p)
    return out


class TestAbciVocabularyIsContained(unittest.TestCase):
    def test_markers_appear_only_under_platforms_abci(self):
        leaks: list[str] = []
        for p in _scan_targets():
            rel = p.relative_to(ROOT)
            if rel.parts[:2] == ("platforms", "abci"):
                continue
            text = p.read_text(encoding="utf-8", errors="replace").lower()
            for m in ABCI_MARKERS:
                if m.lower() in text:
                    leaks.append(f"{rel}: {m}")
        self.assertEqual(leaks, [],
                         "machine-specific vocabulary has leaked outside platforms/abci/:\n"
                         + "\n".join(f"  - {x}" for x in leaks))

    def test_the_scan_actually_looks_at_something(self):
        """With nothing to scan, this check would pass vacuously."""
        self.assertTrue(_scan_targets(), "nothing to scan")


class TestAbciIsOptional(unittest.TestCase):
    """A path that works without any cluster has to exist."""

    def test_local_platform_exists(self):
        self.assertTrue((PLATFORMS / "local").is_dir(),
                        "platforms/local/ is missing; a cluster has become a prerequisite")

    def test_base_interface_exists(self):
        self.assertTrue((PLATFORMS / "base.py").is_file(),
                        "the shared interface platforms/base.py is missing")

    def test_base_interface_is_free_of_abci_vocabulary(self):
        """Once the interface is polluted, the separation is decorative."""
        text = (PLATFORMS / "base.py").read_text(encoding="utf-8").lower()
        for m in ABCI_MARKERS:
            self.assertNotIn(m.lower(), text, f"the interface mentions {m}")

    def test_nothing_outside_platforms_imports_the_abci_module(self):
        """Importing it means the core knows about it."""
        pat = re.compile(r"(?:from|import)\s+[\w.]*platforms\.abci")
        offenders: list[str] = []
        for p in _scan_targets():
            rel = p.relative_to(ROOT)
            if rel.parts[0] == "platforms":
                continue
            if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(rel))
        self.assertEqual(offenders, [],
                         f"imported from outside platforms/: {offenders}")

    def test_abci_module_exists_but_is_not_required(self):
        """It exists, but nothing breaks without it."""
        self.assertTrue(ABCI_DIR.is_dir(), "platforms/abci/ is missing")
        base = (PLATFORMS / "base.py").read_text(encoding="utf-8")
        self.assertNotIn("abci", base.lower())


class TestBackendsShareOneInterface(unittest.TestCase):
    """Load both for real and confirm they satisfy one interface.

    Imported as a package. Loading the files individually loads the same
    module twice, so the two `Backend` classes are different objects and
    `issubclass` comes out false. The Capture side hit the same shape of bug,
    where `except` stopped catching.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

    def test_local_backend_implements_the_interface(self):
        import platforms
        self.assertTrue(issubclass(
            platforms.load_backend("local").Backend, platforms.Backend))

    def test_abci_backend_implements_the_interface(self):
        import platforms
        self.assertTrue(issubclass(
            platforms.load_backend("abci").Backend, platforms.Backend))

    def test_job_spec_is_expressed_in_generic_terms(self):
        """The core states needs generically; naming resources is the backend's job."""
        import platforms
        fields = set(platforms.JobSpec.__dataclass_fields__)
        for f in ("command", "env_name", "gpus", "hours", "name"):
            self.assertIn(f, fields, f"JobSpec has no {f}")


class TestBackendResolutionHasNoHardcodedTable(unittest.TestCase):
    """**The core holds no backend name at all.** The caller supplies it."""

    @classmethod
    def setUpClass(cls) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

    def test_backends_are_discovered_not_listed(self):
        import platforms
        self.assertIn("local", platforms.available_backends())

    def test_unknown_backend_says_what_is_available(self):
        """"Not found" alone leaves the caller with nothing to act on."""
        import platforms
        with self.assertRaises(ValueError) as e:
            platforms.load_backend("nonexistent")
        self.assertIn("local", str(e.exception))

    def test_a_newly_added_backend_is_discovered_without_code_changes(self):
        """**Checked by behaviour.** Matching on text flags the usage example in
        a docstring as a hard-coded name; that mistake was made here already.

        With a table, dropping in a directory would not be noticed.
        """
        import platforms
        dummy = PLATFORMS / "_dummy_for_test"
        (dummy).mkdir(exist_ok=True)
        try:
            (dummy / "__init__.py").write_text("", encoding="utf-8")
            (dummy / "backend.py").write_text("", encoding="utf-8")
            self.assertIn("_dummy_for_test", platforms.available_backends(),
                          "dropping it in was not enough; a table is being kept")
        finally:
            shutil.rmtree(dummy, ignore_errors=True)
        self.assertNotIn("_dummy_for_test", platforms.available_backends(),
                         "still listed after removal; the filesystem is not being consulted")


if __name__ == "__main__":
    unittest.main()
