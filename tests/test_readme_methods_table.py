#!/usr/bin/env python3
"""The README per-method table must list every ported method, and in order.

Two separate failures this guards, both of which actually happened:

  1. **Membership.** The layout *tree* and the `bin/` tools are checked against
     the filesystem (tests/test_repo_hygiene.py), but the per-method **table**
     -- the one under `## Methods` that says what is ported -- was not. A method
     could be added under `methods/` and never appear in the table, or a row
     could linger after a directory was removed, and nothing would complain.

  2. **Order.** The table is documented as sorted "so they sort in numeric
     order", but each port appended its row, and the order drifted into
     `... 16, 18, 19, 22, 23, 26, 29, 33, 32, 17, 20, 21, ...` -- numeric at
     first, then arbitrary. Care did not hold the order across many ports, so a
     mechanism does: the rows must be sorted by numeric prefix, then the
     non-numbered names (image_gpt, mar, var) alphabetically.

Filesystem-only (no git), so it runs unchanged in the container image.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"
EXEMPT = {"_reference"}


def _order_key(name: str):
    """Numeric-prefixed methods first, in numeric order; then the rest by name.
    The single source of truth for how the table must be ordered."""
    m = re.match(r"(\d+)_", name)
    return (0, int(m.group(1)), "") if m else (1, 0, name)


def _table_dirs() -> "list[str]":
    """The directory keys of the per-method table rows, in the order written."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    hdr = text.index("| Directory | Method | Stages | Notes |")
    lines = text[hdr:].splitlines()
    rows = []
    for line in lines[2:]:            # skip header + |---| separator
        if not line.startswith("| `"):
            break
        rows.append(re.match(r"\| `([^`]+)`", line).group(1))
    return rows


def _disk_methods() -> "set[str]":
    return {p.name for p in METHODS.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in EXEMPT}


class TestTheMethodsTable(unittest.TestCase):
    # Method names are DISCOVERED at runtime, never written as literals here --
    # this is a shared file, and tests/test_no_hard_coded_methods.py forbids a
    # shared file from naming a method (a list rots when the next one arrives).

    def test_the_parser_found_the_table(self):
        """A positive control: against an empty parse everything below passes
        vacuously."""
        rows = _table_dirs()
        self.assertGreater(len(rows), 20)
        self.assertTrue(set(rows) & _disk_methods(),
                        "the parsed table shares no rows with methods/ -- the "
                        "parser probably missed the table")

    def test_every_method_on_disk_has_exactly_one_row(self):
        rows = _table_dirs()
        disk = _disk_methods()
        self.assertEqual(set(rows) - disk, set(),
                         "table lists methods not on disk")
        self.assertEqual(disk - set(rows), set(),
                         "methods on disk missing from the table")
        self.assertEqual(len(rows), len(set(rows)), "a method is listed twice")

    def test_the_table_is_in_numeric_then_alphabetical_order(self):
        rows = _table_dirs()
        self.assertEqual(rows, sorted(rows, key=_order_key),
                         "the methods table is out of order; sort rows by "
                         "numeric prefix, then the non-numbered names by name")

    def test_the_order_check_is_not_vacuous(self):
        """A negative control: a genuinely mis-ordered list must not satisfy the
        order rule (otherwise the assertion above could never fire). Built from
        discovered names, so no method is named here."""
        numbered = sorted((d for d in _disk_methods() if re.match(r"\d+_", d)),
                          key=_order_key)
        self.assertGreaterEqual(len(numbered), 2)
        scrambled = [numbered[1], numbered[0]] + numbered[2:]   # swap first two
        self.assertNotEqual(scrambled, sorted(scrambled, key=_order_key))


if __name__ == "__main__":
    unittest.main()
