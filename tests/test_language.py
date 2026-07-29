#!/usr/bin/env python3
"""Everything in this repository must be written in English.

This repository will be published to the world. Documentation, comments,
docstrings, error messages and test names all have to be readable by people
who do not read Japanese.

A convention alone does not hold. We have already seen a strict rule written
in three separate places get broken on the same day, so this is enforced
mechanically rather than by agreement.

If a Japanese document is ever wanted, add it deliberately as `*.ja.md` and
extend the allowlist below in the same commit. The point is that the choice
has to be explicit.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hiragana, katakana, CJK ideographs, and fullwidth forms.
# Written with escapes so that this file is itself pure ASCII: the scanner
# must not be the one file that is exempt from the rule it enforces.
CJK = re.compile("[\u3000-\u303f\u3040-\u30ff\u4e00-\u9fff\uff00-\uffef]")

# Sample text used by the tests. Built from escapes for the same reason.
JAPANESE = "\u3053\u308c\u306f\u65e5\u672c\u8a9e"

SKIP_DIRS = (".git",)
MAX_BYTES = 8 << 20

# Deliberate exceptions. **Only `*.ja.md` may appear here** -- otherwise any
# offending file could be made to disappear by naming it, which is exactly
# how a mutation of this guard survived. Every entry must also exist.
ALLOWLIST: tuple[str, ...] = ()


def classify(root: Path, max_bytes: int = MAX_BYTES
             ) -> tuple[list[Path], list[tuple[str, str]]]:
    """Split the tree into files to read and files not read, with reasons.

    **There is no list of extensions to scan.** A suffix list is a dial that
    can be turned down without anything failing; narrowing it to hide an
    offending file was an actual surviving mutation of this guard. So every
    regular file is read, and only genuinely unreadable ones are set aside --
    each with a reason, because a silent skip is a silent failure
    (DESIGN 2.4).
    """
    found: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if p.relative_to(root).parts[0] in SKIP_DIRS:
            continue
        if rel in ALLOWLIST:
            skipped.append((rel, "allowlisted as a deliberate translation"))
            continue
        try:
            size = p.stat().st_size
            if size > max_bytes:
                skipped.append((rel, f"larger than {max_bytes} bytes"))
                continue
            data = p.read_bytes()
        except OSError as exc:
            skipped.append((rel, f"cannot be read: {exc}"))
            continue
        if b"\0" in data:
            skipped.append((rel, "binary: contains a NUL byte"))
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append((rel, "not valid UTF-8"))
            continue
        found.append(p)
    return found, skipped


class TestEverythingIsEnglish(unittest.TestCase):
    def test_no_japanese_text(self):
        found, skipped = classify(ROOT)
        offenders: list[str] = []
        for p in found:
            text = p.read_text(encoding="utf-8")
            for n, line in enumerate(text.splitlines(), 1):
                if CJK.search(line):
                    offenders.append(f"{p.relative_to(ROOT)}:{n}")
                    break            # one hit per file is enough to report
        self.assertEqual(
            offenders, [],
            "This repository is published; write in English.\n"
            + "\n".join(f"  - {x}" for x in offenders)
            + "\nnot read: "
            + (", ".join(f"{p} ({why})" for p, why in skipped) or "nothing"))

    def test_the_scan_actually_looks_at_something(self):
        """A scan over zero files would pass vacuously."""
        found, _ = classify(ROOT)
        self.assertGreater(len(found), 5)

    def test_the_documents_are_among_what_is_read(self):
        """Name them: a scan that reads only code would pass the check above."""
        found, _ = classify(ROOT)
        names = {str(p.relative_to(ROOT)) for p in found}
        for must in ("README.md", "CLAUDE.md", "docs/PLATFORMS.md",
                     "LICENSE", ".githooks/pre-commit"):
            self.assertIn(must, names, f"{must} is never read")

    def test_the_detector_actually_detects(self):
        """Every script class, checked on its own.

        A single mixed sample such as "kore wa nihongo" is hiragana *and*
        kanji, so it still matches when one of the two ranges is deleted.
        Mutation testing showed exactly that: removing hiragana, and removing
        kanji, both went unnoticed. One sample per range, therefore.
        """
        for label, sample in (
                ("hiragana", "\u3042\u3044\u3046"),
                ("katakana", "\u30a2\u30a4\u30a6"),
                ("kanji", "\u65e5\u672c\u8a9e"),
                ("ideographic punctuation", "\u3001\u3002\u300c"),
                ("ideographic space", "\u3000"),
                ("fullwidth forms", "\uff21\uff22\uff10")):
            with self.subTest(script=label):
                self.assertTrue(CJK.search(sample), f"{label} is not detected")

    def test_the_detector_does_not_fire_on_english(self):
        """A regex that matches everything would pass the test above."""
        for sample in ("This is plain English.", "naive, cafe, resume",
                       "x = 1  # ok", "", "0123456789 !?-_/"):
            with self.subTest(sample=sample):
                self.assertIsNone(CJK.search(sample))


class TestTheGuardCannotBeSilentlyDisarmed(unittest.TestCase):
    """Mutation testing showed two ways to switch this guard off.

    With Japanese actually present in README.md, both of these made the suite
    pass again:

    - dropping ``.md`` from a list of scanned suffixes
    - adding ``README.md`` to the allowlist

    Neither shows up as a failure, so neither would be noticed. A guard whose
    own configuration is unguarded is not a guard.
    """

    def _tree(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="langtest-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def test_an_unfamiliar_extension_is_still_scanned(self):
        """No suffix list to narrow: anything decodable as text is read."""
        d = self._tree()
        (d / "notes.rst").write_text(JAPANESE, encoding="utf-8")
        found, _ = classify(d)
        self.assertIn(d / "notes.rst", found)

    def test_a_file_with_no_extension_is_still_scanned(self):
        d = self._tree()
        (d / "Makefile").write_text(JAPANESE, encoding="utf-8")
        found, _ = classify(d)
        self.assertIn(d / "Makefile", found)

    def test_binary_files_are_not_scanned_but_are_reported(self):
        """Skipping is allowed. Skipping quietly is not (DESIGN 2.4)."""
        d = self._tree()
        (d / "weights.bin").write_bytes(b"\x00\x01" + JAPANESE.encode())
        found, skipped = classify(d)
        self.assertNotIn(d / "weights.bin", found)
        self.assertIn("weights.bin", " ".join(p for p, _ in skipped))
        self.assertTrue(all(reason for _, reason in skipped),
                        "a skip was recorded without a reason")

    def test_an_unreadable_file_is_reported_not_dropped(self):
        """The one case where reading fails for real. It must still be named.

        Mutation testing found this path had no test: deleting the record
        entirely went unnoticed, so an unreadable file would have vanished
        from both lists without a word.
        """
        if os.geteuid() == 0:
            self.skipTest("root can read a mode-000 file, so nothing is proven")
        d = self._tree()
        p = d / "locked.md"
        p.write_text(JAPANESE, encoding="utf-8")
        p.chmod(0o000)
        self.addCleanup(p.chmod, 0o600)
        found, skipped = classify(d)
        self.assertNotIn(p, found)
        self.assertIn("locked.md", " ".join(x for x, _ in skipped),
                      "an unreadable file disappeared without a record")

    def test_nothing_disappears_from_both_lists(self):
        """Every regular file must land in exactly one of the two lists."""
        d = self._tree()
        (d / "a.md").write_text("english", encoding="utf-8")
        (d / "b.bin").write_bytes(b"\x00\xff")
        found, skipped = classify(d)
        accounted = {p.name for p in found} | {x for x, _ in skipped}
        self.assertEqual(accounted, {"a.md", "b.bin"})

    def test_an_oversized_file_is_reported_not_dropped(self):
        """The size cap is a skip like any other, so it must be recorded."""
        d = self._tree()
        (d / "huge.md").write_text(JAPANESE * 100, encoding="utf-8")
        found, skipped = classify(d, max_bytes=8)
        self.assertNotIn(d / "huge.md", found)
        self.assertIn("huge.md", " ".join(x for x, _ in skipped),
                      "an oversized file disappeared without a record")
        self.assertIn("8", " ".join(why for _, why in skipped),
                      "the reason does not say what the limit was")

    def test_the_allowlist_can_only_exempt_translations(self):
        """The escape hatch is *.ja.md and nothing else.

        Otherwise any offending file can be made to disappear by naming it
        here, which is how M5 survived.
        """
        for entry in ALLOWLIST:
            self.assertTrue(entry.endswith(".ja.md"),
                            f"only *.ja.md may be allowlisted, not {entry}")

    def test_allowlisted_paths_must_exist(self):
        """A stale entry is an exemption nobody can see the subject of."""
        for entry in ALLOWLIST:
            self.assertTrue((ROOT / entry).is_file(),
                            f"allowlisted but absent: {entry}")


if __name__ == "__main__":
    unittest.main()
