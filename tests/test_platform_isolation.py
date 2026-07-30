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

# What the guard is about: **code and configuration must not be tied to a
# platform.** Prose cannot create that tie -- the README has to be able to say
# "support for this platform is optional" in order to explain the separation
# at all -- so documentation is out, and that is a definition of the rule
# rather than a hole in it. `tests/` and `.githooks/` are out for the same
# reason: they describe the mechanism.
#
# **No suffix list.** One used to sit here holding six extensions, and a
# `Dockerfile` has none of them, so platform vocabulary could have been
# written into one and this guard would not have looked. The same narrowing
# defeated the language guard until it was removed there too. "Which files are
# text" now has one implementation, imported.
PROSE_SUFFIXES = (".md", ".rst", ".txt")
SKIP_DIRS = (".git", "docs", "tests", ".githooks")


def _scan_targets(root: Path = ROOT) -> list[Path]:
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    from test_language import classify           # the one implementation
    found, _ = classify(root)
    return [p for p in found
            if p.relative_to(root).parts[0] not in SKIP_DIRS
            and p.suffix not in PROSE_SUFFIXES]


def find_leaks(root: Path = ROOT) -> list[str]:
    """Marker sightings outside platforms/abci/, as `path: marker`."""
    leaks = []
    for p in _scan_targets(root):
        rel = p.relative_to(root)
        if rel.parts[:2] == ("platforms", "abci"):
            continue
        text = p.read_text(encoding="utf-8", errors="replace").lower()
        for marker in ABCI_MARKERS:
            if marker.lower() in text:
                leaks.append(f"{rel}: {marker}")
    return leaks


class TestAbciVocabularyIsContained(unittest.TestCase):
    def test_markers_appear_only_under_platforms_abci(self):
        self.assertEqual(
            find_leaks(), [],
            "machine-specific vocabulary has leaked outside platforms/abci/:\n"
            + "\n".join(f"  - {x}" for x in find_leaks()))

    def test_the_scan_actually_looks_at_something(self):
        """With nothing to scan, this check would pass vacuously."""
        self.assertGreater(len(_scan_targets()), 5)

    def test_a_marker_in_a_file_with_no_suffix_is_caught(self):
        """The case the old suffix list could not see. A Dockerfile is the
        one that prompted this, and it has no extension."""
        import shutil, tempfile
        d = Path(tempfile.mkdtemp(prefix="isolation-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "Dockerfile").write_text("RUN qsub something\n", encoding="utf-8")
        self.assertIn("Dockerfile: qsub", find_leaks(d))

    def test_prose_may_name_the_platform(self):
        """**Not a loophole: it is the rule.** The README cannot explain that
        the platform is optional without naming it, and prose creates no
        dependency."""
        import shutil, tempfile
        d = Path(tempfile.mkdtemp(prefix="isolation-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "README.md").write_text("Support for abci is optional.\n",
                                     encoding="utf-8")
        self.assertEqual(find_leaks(d), [])

    def test_but_code_beside_that_prose_is_still_scanned(self):
        """Excluding prose must not excuse the directory it sits in."""
        import shutil, tempfile
        d = Path(tempfile.mkdtemp(prefix="isolation-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "README.md").write_text("abci is optional\n", encoding="utf-8")
        (d / "launch.sh").write_text("qsub job.sh\n", encoding="utf-8")
        self.assertIn("launch.sh: qsub", find_leaks(d))


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
