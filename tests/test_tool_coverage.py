#!/usr/bin/env python3
"""Every tool has to be under test, and that is enforced here.

One tool in the Capture repository was written before the testing rule was
adopted and went into production with **zero tests**. Writing the rule down
did not catch it; a machine did.

"Under test" means some file in `tests/` names the tool's filename. That
covers both styles used here: importing the module, and driving the CLI
through a subprocess.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
TESTS = ROOT / "tests"
SELF = Path(__file__).name


def _corpus() -> dict[str, str]:
    """Exclude this file. Otherwise the list of tool names below would
    satisfy the check by itself."""
    out: dict[str, str] = {}
    for p in sorted(TESTS.glob("test_*.py")):
        if p.name != SELF:
            out[p.name] = p.read_text(encoding="utf-8")
    for extra in ("run-tests.sh",):
        p = TESTS / extra
        if p.exists():
            out[extra] = p.read_text(encoding="utf-8")
    return out


class TestEveryToolIsUnderTest(unittest.TestCase):
    def test_no_tool_is_left_untested(self):
        corpus = _corpus()
        tools = sorted(p.name for p in BIN.glob("*.py"))
        tools += sorted(p.name for p in BIN.glob("*.sh"))
        self.assertTrue(tools, "no tools found under bin/")
        missing = [t for t in tools
                   if not any(t in body for body in corpus.values())]
        self.assertEqual(
            missing, [],
            "a tool is never referenced from any test.\n"
            "  add tests/test_<name>.py, or exercise it end to end:\n"
            + "".join(f"    - bin/{m}\n" for m in missing))

    def test_guard_itself_does_not_count_as_coverage(self):
        """Forget to exclude this file and the guard always passes."""
        self.assertNotIn(SELF, _corpus(),
                         "the guard itself is inside the corpus it searches")


if __name__ == "__main__":
    unittest.main()
