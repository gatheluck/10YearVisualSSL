"""The operating guide lists every public method once and preserves the historical order."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"
EXEMPT = {"_reference"}


from adapterlib.method_names import HISTORICAL, order_key as _order_key


def _table_dirs() -> "list[str]":
    """The directory keys of the per-method table rows, in the order written."""
    text = (ROOT / "OPERATING_GUIDE.md").read_text(encoding="utf-8")
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

    def test_the_table_is_in_historical_then_alphabetical_order(self):
        rows = _table_dirs()
        self.assertEqual(rows, sorted(rows, key=_order_key),
                         "the methods table is out of order; sort rows by "
                         "historical order, then additional methods by name")

    def test_the_order_check_is_not_vacuous(self):
        """A negative control: a genuinely mis-ordered list must not satisfy the
        order rule (otherwise the assertion above could never fire). Built from
        discovered names, so no method is named here."""
        historical = sorted((d for d in _disk_methods() if d in HISTORICAL),
                          key=_order_key)
        self.assertGreaterEqual(len(historical), 2)
        scrambled = [historical[1], historical[0]] + historical[2:]   # swap first two
        self.assertNotEqual(scrambled, sorted(scrambled, key=_order_key))


if __name__ == "__main__":
    unittest.main()
