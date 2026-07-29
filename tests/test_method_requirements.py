#!/usr/bin/env python3
"""A method must declare exactly the packages it imports.

This exists because the first port got it wrong. `requirements.txt` came
across from the capture unchanged, and it was the *legacy* track's list:
it named `timm`, `PyYAML`, `tensorboard` and `tqdm`, none of which the ported
code imports, and it was read by nobody.

Both directions of that are harmful:

- **A package that is imported but not declared** turns into an ImportError on
  a fresh machine, after the environment was built and believed correct
- **A package that is declared but not imported** is installed for nothing and,
  worse, misleads the next reader about what the method actually needs

So the requirement is checked against the imports rather than trusted. The
same check covers every method added later, which is the point: this is one
rule in one place, not a habit each port has to remember.

Versions are a separate matter, handled by `requirements.lock.txt`. This file
only settles *which* packages, not which versions.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"

# Modules that come from the repository rather than from an index.
LOCAL = {"adapterlib", "adapter", "data", "models",
         "train_step1_alexnet_official"}

# Import name -> distribution name, where they differ. Anything imported and
# not listed here is assumed to install under its own name; the completeness
# check below fails rather than let an unknown one slide.
DISTRIBUTION = {"PIL": "Pillow", "yaml": "PyYAML", "cv2": "opencv-python",
                "sklearn": "scikit-learn", "skimage": "scikit-image"}


def method_dirs() -> list[Path]:
    if not METHODS.is_dir():
        return []
    return sorted(p for p in METHODS.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def imported_modules(method: Path) -> set[str]:
    """Top-level third-party modules imported anywhere under the method."""
    found: set[str] = set()
    for py in sorted(method.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found.add((node.module or "").split(".")[0])
    return {m for m in found
            if m and m not in sys.stdlib_module_names and m not in LOCAL}


def declared_packages(req: Path) -> set[str]:
    """Distribution names in a requirements file, without version specifiers."""
    out = set()
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        out.add(re.split(r"[<>=!~\[]", line, 1)[0].strip().lower())
    return out


class TestEveryMethodDeclaresWhatItImports(unittest.TestCase):
    def test_there_is_a_method_to_check(self):
        """With no methods this file would pass without looking at anything."""
        self.assertTrue(method_dirs(), "no methods found to check")

    def test_each_method_has_a_requirements_file(self):
        for m in method_dirs():
            with self.subTest(method=m.name):
                if not imported_modules(m):
                    continue          # a method needing nothing needs no file
                self.assertTrue((m / "requirements.txt").is_file(),
                                f"{m.name} imports packages but declares none")

    def test_nothing_imported_is_undeclared(self):
        for m in method_dirs():
            req = m / "requirements.txt"
            if not req.is_file():
                continue
            declared = declared_packages(req)
            with self.subTest(method=m.name):
                for mod in sorted(imported_modules(m)):
                    dist = DISTRIBUTION.get(mod, mod).lower()
                    self.assertIn(
                        dist, declared,
                        f"{m.name} imports {mod} but {dist} is not in "
                        "requirements.txt; it would fail on a fresh machine")

    def test_nothing_declared_is_unimported(self):
        """The failure that prompted this: four packages declared, none used."""
        for m in method_dirs():
            req = m / "requirements.txt"
            if not req.is_file():
                continue
            needed = {DISTRIBUTION.get(x, x).lower()
                      for x in imported_modules(m)}
            with self.subTest(method=m.name):
                extra = declared_packages(req) - needed
                self.assertEqual(
                    extra, set(),
                    f"{m.name} declares {sorted(extra)} but imports none of "
                    "them; the list misleads whoever builds the environment")

    def test_the_scan_finds_real_imports(self):
        """Against an empty scan both checks above would pass vacuously."""
        found = set()
        for m in method_dirs():
            found |= imported_modules(m)
        self.assertTrue(found, "no third-party imports were found at all")

    def test_the_distribution_table_is_used_where_it_must_be(self):
        """Guard the mapping itself: PIL does not install as `pil`."""
        self.assertEqual(DISTRIBUTION["PIL"], "Pillow")


class TestTheOptionalToolingDependency(unittest.TestCase):
    """PyYAML belongs to the tooling, not to any method.

    Found by running the README's own commands in order: the environment its
    first step builds could not carry out its second, because
    `resolve-config.py` needs PyYAML to read a YAML authoring config and no
    method declares it -- correctly, since no method imports it.
    """

    TOOLS = ROOT / "requirements-tools.txt"
    LOCK = ROOT / "requirements-tools.lock.txt"

    def test_the_tooling_requirement_is_declared_and_locked(self):
        self.assertTrue(self.TOOLS.is_file())
        self.assertTrue(self.LOCK.is_file())

    def test_it_is_pyyaml_and_nothing_else(self):
        """The core stays standard-library only. This is the one optional
        extra, and it buys exactly one thing: YAML authoring."""
        self.assertEqual(declared_packages(self.TOOLS), {"pyyaml"})

    def test_the_tooling_lock_is_exact(self):
        for line in self.LOCK.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                self.assertIn("==", line, f"{line!r} is not pinned")

    def test_no_method_declares_it(self):
        """It is the launcher's dependency. A method that declared it would
        be claiming to import something it does not."""
        for m in method_dirs():
            req = m / "requirements.txt"
            if req.is_file():
                with self.subTest(method=m.name):
                    self.assertNotIn("pyyaml", declared_packages(req))


class TestVersionsArePinned(unittest.TestCase):
    """`torch>=2.0.0` is not a reproducible environment.

    A floor says what will import, not what ran. A run recorded against a
    floor cannot be rebuilt: any version above it satisfies the file.
    """

    def test_each_method_has_a_lock_file(self):
        for m in method_dirs():
            if not (m / "requirements.txt").is_file():
                continue
            with self.subTest(method=m.name):
                self.assertTrue(
                    (m / "requirements.lock.txt").is_file(),
                    f"{m.name} has no requirements.lock.txt, so the "
                    "environment it ran in cannot be rebuilt")

    def test_every_line_in_a_lock_file_is_an_exact_version(self):
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            for line in lock.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                with self.subTest(method=m.name, line=line):
                    self.assertRegex(
                        line, r"==",
                        f"{line!r} is not pinned; a lock file states exactly "
                        "one version")

    def test_the_lock_covers_everything_declared(self):
        for m in method_dirs():
            req, lock = m / "requirements.txt", m / "requirements.lock.txt"
            if not (req.is_file() and lock.is_file()):
                continue
            with self.subTest(method=m.name):
                missing = declared_packages(req) - declared_packages(lock)
                self.assertEqual(missing, set(),
                                 f"{m.name}: declared but not locked")


if __name__ == "__main__":
    unittest.main()
