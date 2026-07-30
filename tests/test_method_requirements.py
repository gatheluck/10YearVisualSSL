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


def _canon(name: str) -> str:
    """Distribution names compare case- and separator-insensitively."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def requirement_lines(path: Path) -> list[str]:
    """Requirement lines, with hash continuations folded back in.

    A hashed lock spreads one requirement over several lines with trailing
    backslashes. Reading it line by line makes a hash look like a requirement
    that is not pinned -- which is exactly how one of these checks first
    failed. The rule lives here once so both callers agree.
    """
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(line)
    return out


def declared_packages(req: Path) -> set[str]:
    """Distribution names in a requirements file, without version specifiers."""
    return {re.split(r"[<>=!~\[]", line, 1)[0].strip().lower()
            for line in requirement_lines(req)}


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

    def test_the_tooling_lock_is_exact_and_hashed(self):
        lines = requirement_lines(self.LOCK)
        self.assertTrue(lines, "the tooling lock is empty")
        for line in lines:
            self.assertIn("==", line, f"{line!r} is not pinned")
            self.assertIn("--hash=sha256:", line, f"{line!r} has no hash")

    def test_no_method_declares_it(self):
        """It is the launcher's dependency. A method that declared it would
        be claiming to import something it does not."""
        for m in method_dirs():
            req = m / "requirements.txt"
            if req.is_file():
                with self.subTest(method=m.name):
                    self.assertNotIn("pyyaml", declared_packages(req))


class TestTheInterpreterIsPinnedToo(unittest.TestCase):
    """A lock over packages says nothing about the interpreter running them.

    Nothing in this repository constrained the Python version: no
    `.python-version`, no `requires-python`, nothing. The same pinned torch
    behaves differently on a different interpreter, and the lock would have
    looked satisfied either way.
    """

    VERSION_FILE = ROOT / ".python-version"

    def test_the_interpreter_version_is_declared(self):
        self.assertTrue(self.VERSION_FILE.is_file(),
                        "no .python-version: the interpreter is unconstrained")

    def test_it_is_an_exact_version(self):
        text = self.VERSION_FILE.read_text(encoding="utf-8").strip()
        self.assertRegex(text, r"^\d+\.\d+\.\d+$",
                         f"{text!r} is not an exact version")

    def test_the_declared_minor_matches_what_the_locks_were_built_for(self):
        """A lock holds wheels built for one ABI. Declaring 3.11 while the
        wheels are cp312 would install nothing."""
        declared = self.VERSION_FILE.read_text(encoding="utf-8").strip()
        major_minor = ".".join(declared.split(".")[:2])
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            text = lock.read_text(encoding="utf-8")
            if "cp3" in text:
                with self.subTest(method=m.name):
                    tag = "cp" + major_minor.replace(".", "")
                    self.assertIn(tag, text,
                                  f"the lock holds no {tag} wheels")


class TestTheLockIsAClosure(unittest.TestCase):
    """A lock that names only the direct requirements is not a lock.

    Measured before this: the file pinned 3 packages while 12 were installed.
    The nine that floated included torch's own dependencies.
    """

    def test_the_lock_names_more_than_the_direct_requirements(self):
        for m in method_dirs():
            req, lock = m / "requirements.txt", m / "requirements.lock.txt"
            if not (req.is_file() and lock.is_file()):
                continue
            with self.subTest(method=m.name):
                direct, locked = declared_packages(req), declared_packages(lock)
                self.assertGreater(
                    len(locked), len(direct),
                    f"{m.name}: the lock has {len(locked)} packages and the "
                    f"direct requirements have {len(direct)}; transitive "
                    "dependencies are not pinned")

    def test_the_lock_contains_the_dependencies_of_what_it_locks(self):
        """Closure, checked against the packages' own metadata.

        Counting entries was not enough: deleting one transitive dependency
        left the count above the direct requirements and nothing failed. This
        reads each locked package's `Requires-Dist` and demands the result be
        present too.

        Only unconditional requirements are followed. Anything behind an
        environment marker or an extra may legitimately be absent, and
        evaluating markers needs a package we do not have.
        """
        import importlib.metadata as md
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            locked = {_canon(x) for x in declared_packages(lock)}
            checked = 0
            for name in sorted(locked):
                try:
                    reqs = md.requires(name) or []
                except md.PackageNotFoundError:
                    continue          # not installed here; nothing to read
                checked += 1
                for raw in reqs:
                    if ";" in raw:    # marker or extra: may not apply
                        continue
                    dep = _canon(re.split(r"[<>=!~\[ (]", raw, 1)[0])
                    with self.subTest(method=m.name, package=name, needs=dep):
                        self.assertIn(
                            dep, locked,
                            f"{name} requires {dep}, which the lock omits; "
                            "the lock is not a closure")
            if checked == 0:
                self.skipTest(f"{m.name}: none of its packages are installed "
                              "here, so the closure cannot be read")

    def test_every_locked_package_carries_a_hash(self):
        """Without hashes a version can be replaced under the same name."""
        for m in method_dirs():
            lock = m / "requirements.lock.txt"
            if not lock.is_file():
                continue
            for line in requirement_lines(lock):
                with self.subTest(method=m.name, package=line.split("==")[0]):
                    self.assertIn("--hash=sha256:", line,
                                  f"{line.split('==')[0]} has no hash")


# The platforms a lock is expected to be installable on, as substrings that
# must all appear in one wheel filename. Declared here so that "we support
# linux arm64" cannot quietly stop being true.
#
# Found by building: Docker on Apple silicon builds linux/arm64, whose wheels
# are `aarch64`. The lock held x86_64 and macOS arm64 only, so pip refused --
# correctly, and loudly, but the platform should have been covered.
TARGET_PLATFORMS = {
    "linux x86_64": ("x86_64",),
    "linux aarch64": ("aarch64",),
    "macOS arm64": ("macosx", "arm64"),
}
UNIVERSAL = "py3-none-any"


def all_locks() -> list[Path]:
    """Every lock in the repository, not only the methods'.

    The first version of the coverage check looked at method locks alone. The
    container build then failed on the *tooling* lock, which had the same gap
    -- a rule applied to some of the things it governs is a rule with a hole
    in it.
    """
    found = [m / "requirements.lock.txt" for m in method_dirs()]
    found.append(ROOT / "requirements-tools.lock.txt")
    return [p for p in found if p.is_file()]


def wheels_per_package(lock: Path) -> dict:
    """Package name -> the wheel filenames recorded in its comments.

    The generator writes one `# <filename>` line per distinct wheel directly
    above the requirement it belongs to.
    """
    out, pending = {}, []
    for raw in lock.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            name = line[1:].strip()
            if name.endswith((".whl", ".tar.gz")):
                pending.append(name)
        elif "==" in line and not line.startswith("-"):
            out[_canon(line.split("==")[0])] = pending
            pending = []
        elif not line:
            continue
    return out


class TestTheLockCoversEveryTargetPlatform(unittest.TestCase):
    """A lock that omits a platform is not wrong, but it is not usable there.

    pip refuses an unlisted wheel rather than installing something unverified,
    which is the behaviour we want -- but the failure arrives at build time,
    far from the file that caused it. This brings it forward.
    """

    def test_there_is_more_than_one_lock_to_check(self):
        """Guards the widening: checking a single file would pass vacuously
        if the others quietly stopped being found."""
        self.assertGreater(len(all_locks()), 1)

    def test_every_package_has_a_wheel_for_every_target(self):
        for lock in all_locks():
            name = str(lock.relative_to(ROOT))
            per_package = wheels_per_package(lock)
            self.assertTrue(per_package, f"{name}: no wheels recorded")
            for pkg, wheels in sorted(per_package.items()):
                if any(UNIVERSAL in w for w in wheels):
                    continue          # pure python: one wheel serves all
                for label, needles in sorted(TARGET_PLATFORMS.items()):
                    with self.subTest(lock=name, package=pkg,
                                      platform=label):
                        self.assertTrue(
                            any(all(n in w for n in needles) for w in wheels),
                            f"{pkg} has no wheel for {label}; installing "
                            "there would be refused for want of a hash")

    def test_the_number_of_hashes_matches_the_number_of_wheels(self):
        """The comments and the hashes are written together; if they drift,
        the check above is reading a filename that no hash belongs to."""
        for lock in all_locks():
            per_package = wheels_per_package(lock)
            for line in requirement_lines(lock):
                pkg = _canon(line.split("==")[0])
                with self.subTest(lock=str(lock.relative_to(ROOT)),
                                  package=pkg):
                    self.assertEqual(line.count("--hash=sha256:"),
                                     len(per_package.get(pkg, [])),
                                     "hashes and recorded wheels disagree")

    def test_the_declared_targets_are_the_ones_we_claim_to_support(self):
        """Shrinking this silently would make the check above pass by
        covering less."""
        self.assertEqual(set(TARGET_PLATFORMS),
                         {"linux x86_64", "linux aarch64", "macOS arm64"})


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
            lines = requirement_lines(lock)
            self.assertTrue(lines, f"{m.name}: the lock is empty")
            for line in lines:
                with self.subTest(method=m.name, line=line.split()[0]):
                    self.assertIn(
                        "==", line,
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
