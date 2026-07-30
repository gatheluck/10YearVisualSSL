#!/usr/bin/env python3
"""Specification for the container image.

**I could not build or run this.** There is no Docker, Podman or Apptainer on
the machine it was written on, so every claim below is about the *definition*,
checked by reading it, and none is about a built image. That limit is stated
in the README too, and it is the reason these checks are as strict as they
are: reading is all the verification there is until someone builds it.

What the definition has to hold:

- **The base image is pinned by digest.** A tag is a moving target; `python:3.12`
  means something different next month, and an image built from it cannot be
  rebuilt
- **The environment is a fresh venv, not the base image's site-packages.**
  Measured on the host: a bare venv seeds exactly `pip`. Installing into the
  base image's own site-packages would mix in whatever it ships, and the
  result would not be the locked environment
- **The build verifies itself.** `verify-environment.py` runs as a build step,
  so an image whose environment is not the lock **fails to build** rather than
  existing and lying about what is in it
- **Nothing unpinned is installed.** No `apt-get upgrade`, no floating tags
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"


def method_dirs() -> list[Path]:
    if not METHODS.is_dir():
        return []
    return sorted(p for p in METHODS.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def dockerfiles() -> list[Path]:
    return [m / "Dockerfile" for m in method_dirs()
            if (m / "Dockerfile").is_file()]


def instructions(text: str) -> list[str]:
    """Logical instructions, with line continuations folded in."""
    out = []
    for line in text.replace("\\\n", " ").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(re.sub(r"\s+", " ", line))
    return out


class TestThereIsOne(unittest.TestCase):
    def test_a_method_that_can_be_locked_can_be_containerised(self):
        for m in method_dirs():
            if not (m / "requirements.lock.txt").is_file():
                continue
            with self.subTest(method=m.name):
                self.assertTrue((m / "Dockerfile").is_file(),
                                f"{m.name} has a lock but no Dockerfile")

    def test_the_scan_finds_something(self):
        """With no Dockerfile every check below would pass vacuously."""
        self.assertTrue(dockerfiles())


class TestTheBaseImageIsFixed(unittest.TestCase):
    def test_it_is_pinned_by_digest(self):
        """**A tag is a moving target.** `python:3.12-slim` means something
        different next month, and an image built from it cannot be rebuilt."""
        for df in dockerfiles():
            froms = [i for i in instructions(df.read_text())
                     if i.upper().startswith("FROM ")]
            with self.subTest(file=str(df.relative_to(ROOT))):
                self.assertTrue(froms, "no FROM instruction")
                for line in froms:
                    self.assertRegex(
                        line, r"@sha256:[0-9a-f]{64}",
                        f"{line!r} is not pinned by digest")

    def test_no_floating_tag_is_used(self):
        for df in dockerfiles():
            text = df.read_text()
            with self.subTest(file=str(df.relative_to(ROOT))):
                self.assertNotIn(":latest", text)

    def test_the_interpreter_matches_the_repository_pin(self):
        """A base image on another Python would not accept the cp312 wheels,
        and the failure would come at build time with no explanation."""
        declared = (ROOT / ".python-version").read_text().strip()
        for df in dockerfiles():
            line = next(i for i in instructions(df.read_text())
                        if i.upper().startswith("FROM "))
            with self.subTest(file=str(df.relative_to(ROOT))):
                self.assertIn(declared, line,
                              f"the base image is not Python {declared}")


class TestTheEnvironmentIsTheLockedOne(unittest.TestCase):
    def pip_installs(self, df: Path) -> list[str]:
        """The RUN instructions that install packages.

        Read from instructions rather than from the file's text: the words
        `venv`, `--require-hashes` and the lock filenames all appear in the
        comments explaining them, so a substring search passes even when the
        instruction is gone. Two mutations survived exactly that way.
        """
        return [i for i in instructions(df.read_text())
                if i.upper().startswith("RUN ") and "pip install" in i]

    def test_the_install_requires_hashes(self):
        for df in dockerfiles():
            with self.subTest(file=str(df.relative_to(ROOT))):
                installs = self.pip_installs(df)
                self.assertTrue(installs, "nothing is installed")
                for ins in installs:
                    self.assertIn("--require-hashes", ins)

    def test_it_installs_into_a_fresh_virtual_environment(self):
        """Measured on the host: a bare venv seeds exactly `pip`.

        Installing into the base image's own site-packages would mix in
        whatever that image happens to ship, and the result would not be the
        locked environment -- nor would it match how the host is built.

        Both halves are required: creating a venv and never putting it on the
        PATH installs into the base image anyway.
        """
        for df in dockerfiles():
            ins = instructions(df.read_text())
            with self.subTest(file=str(df.relative_to(ROOT))):
                created = [i for i in ins
                           if i.upper().startswith("RUN ") and "-m venv" in i]
                self.assertTrue(created, "no virtual environment is created")
                target = created[0].split("-m venv", 1)[1].strip().split()[0]
                on_path = [i for i in ins if i.upper().startswith("ENV ")
                           and f"{target}/bin:" in i]
                self.assertTrue(on_path,
                                f"{target} is created but never put on PATH, "
                                "so pip installs into the base image")

    def test_every_lock_it_copies_exists(self):
        """A COPY of a file that is not there fails the build with a message
        about paths, which says nothing about the real mistake."""
        for df in dockerfiles():
            for ins in instructions(df.read_text()):
                if not ins.upper().startswith("COPY "):
                    continue
                src = ins.split()[1]
                if not src.endswith(".txt"):
                    continue
                with self.subTest(file=str(df.relative_to(ROOT)), copies=src):
                    self.assertTrue((ROOT / src).is_file(),
                                    f"{src} does not exist")

    def test_it_installs_both_the_method_and_the_tooling_locks(self):
        """Naming one of the two is the mistake that made a correct
        environment look broken. The image installs the same pair the README
        does.

        Checked on the install instruction, not on the file's text: copying a
        lock in and never installing it satisfies a substring search while
        producing an image missing half its environment.
        """
        for df in dockerfiles():
            ins = instructions(df.read_text())
            copied = {}
            for i in ins:
                if i.upper().startswith("COPY ") and ".lock.txt" in i:
                    src, dst = i.split()[1], i.split()[2]
                    copied[dst] = src
            installed = " ".join(self.pip_installs(df))
            with self.subTest(file=str(df.relative_to(ROOT))):
                self.assertEqual(len(copied), 2,
                                 f"expected two locks, copied {sorted(copied)}")
                for dst, src in sorted(copied.items()):
                    self.assertIn(dst, installed,
                                  f"{src} is copied in but never installed")
                srcs = " ".join(copied.values())
                self.assertIn("requirements-tools.lock.txt", srcs)
                self.assertIn(
                    str((df.parent / "requirements.lock.txt")
                        .relative_to(ROOT)), srcs)

    def test_the_build_verifies_itself(self):
        """**An image that is not the locked environment must not exist.**

        Checking after the fact relies on somebody remembering. Running
        verify-environment.py as a build step makes the image impossible to
        produce when its contents disagree with the lock.
        """
        for df in dockerfiles():
            runs = " ".join(i for i in instructions(df.read_text())
                            if i.upper().startswith("RUN "))
            with self.subTest(file=str(df.relative_to(ROOT))):
                self.assertIn("verify-environment.py", runs,
                              "the build never checks its own environment")


class TestNothingUnpinnedIsInstalled(unittest.TestCase):
    def test_the_system_packages_are_left_alone(self):
        """`apt-get upgrade` pulls whatever the mirror holds that day, so two
        builds of one Dockerfile give two different images."""
        for df in dockerfiles():
            joined = " ".join(instructions(df.read_text())).lower()
            with self.subTest(file=str(df.relative_to(ROOT))):
                self.assertNotIn("apt-get upgrade", joined)
                self.assertNotIn("apt-get dist-upgrade", joined)

    def test_any_apt_install_pins_its_versions(self):
        for df in dockerfiles():
            for ins in instructions(df.read_text()):
                low = ins.lower()
                if "apt-get install" not in low:
                    continue
                with self.subTest(file=str(df.relative_to(ROOT))):
                    self.assertIn("=", ins.split("apt-get install", 1)[1],
                                  "apt packages are installed without versions")

    def test_pip_is_not_upgraded_to_whatever_is_current(self):
        """`pip install --upgrade pip` is the same moving target."""
        for df in dockerfiles():
            joined = " ".join(instructions(df.read_text()))
            with self.subTest(file=str(df.relative_to(ROOT))):
                self.assertNotRegex(joined, r"--upgrade\s+pip\b")
                self.assertNotRegex(joined, r"\bpip\s+install\s+-U\b")


class TestTheBuildContextIsBounded(unittest.TestCase):
    def test_a_dockerignore_exists(self):
        self.assertTrue((ROOT / ".dockerignore").is_file())

    def test_it_excludes_what_would_change_every_build(self):
        """`.git` and run outputs change constantly; including them would
        invalidate the cache and copy secrets-shaped things into the image.

        Compared against parsed entries, not the file's text: `".git" in text`
        is satisfied by `.github`, so deleting the `.git` line changed
        nothing.
        """
        entries = {line.split("#", 1)[0].strip()
                   for line in (ROOT / ".dockerignore").read_text().splitlines()}
        entries.discard("")
        for pattern in (".git", "runs", "__pycache__", ".venv"):
            self.assertIn(pattern, entries, f"{pattern} is not excluded")


if __name__ == "__main__":
    unittest.main()
