#!/usr/bin/env python3
"""The BASIC5 evaluation protocol lives on `main`, and its checkable claims are
enforced here.

The implementation team defined three evaluation tracks over the same five
datasets -- FAIR (frozen backbone + minimal head, the linear probe), ATTENTIVE
(frozen backbone + one shared attention reader), and FINETUNE (full backbone).
Those definitions arrived as chat messages; a chat message is not on `main`, is
not in the working tree next session, and cannot be diffed. So the protocol is
externalised into `docs/BASIC5_PROTOCOL.md` (this repository's rule: a policy in
a document does not hold unless it is also machinery), and this test makes the
parts that *can* be checked without running a model into checked facts:

  - **the protocol document exists** -- a missing source of truth is RED, never a
    silent pass;
  - **every canonical FAIR/ImageNet rule id is documented** -- the rules are read
    as a structured table (whole rows, not a substring search), so a rule cannot
    quietly drop out of the document;
  - **the document and the code agree on the saved representation** -- the doc
    declares the default saved feature (L2-normalised, FAIR rule c), and that
    declaration must equal `bin/extract-features.py`'s actual CLI default, so the
    two cannot drift apart unnoticed.

The rule table is parsed structurally and the parser carries a positive and a
negative control (a well-formed row is read; a decoy line that merely contains a
pipe is not), because a detector that decides what gets checked needs both.

Standard-library only: it reads text and argparse defaults, so it runs in the
base environment, not behind a torch/numpy gate.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "BASIC5_PROTOCOL.md"
TOOL = ROOT / "bin" / "extract-features.py"

# The FAIR / ImageNet-1k linear-probe rules the document must enumerate. These
# ids are the stable handles the rest of the work (and its commits) refer to;
# dropping one from the document must go red here.
CANONICAL_FAIR_RULE_IDS = {"b", "c", "d", "e", "opt", "seed", "aug", "metric"}

# The marker line that precedes the machine-readable rule table in the document.
RULES_MARKER = "<!-- BASIC5-FAIR-IMAGENET-RULES -->"
# The machine-readable declaration of the default saved representation.
DEFAULT_REP_RE = re.compile(r"<!--\s*default-representation:\s*(\w+)\s*-->")


def parse_rule_ids(text: str) -> "set[str]":
    """The set of rule ids in the machine-readable table under RULES_MARKER.

    A rule row is a table row whose first cell is a short bare token (the id);
    the header row (`id`) and the `---` separator are not rules, and a prose
    line that merely contains a `|` is not a row. Whole cells are compared, so
    nothing is matched as a substring.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.strip() == RULES_MARKER)
    except StopIteration:
        return set()
    ids: "set[str]" = set()
    for ln in lines[start + 1:]:
        s = ln.strip()
        if not s.startswith("|"):
            # The table ends at the first line that is not a table row.
            if ids:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2:
            continue
        first = cells[0]
        if first in ("id", "") or set(first) <= set("-: "):
            continue                       # header or separator row
        if re.fullmatch(r"[a-z][a-z0-9_]*", first):
            ids.add(first)
    return ids


def _tool():
    name = "extract_features_tool_for_protocol"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


class TestProtocolDocumentExists(unittest.TestCase):
    def test_the_document_is_present(self):
        self.assertTrue(DOC.is_file(),
                        f"{DOC} is missing: the protocol has no source of "
                        "truth on main")

    def test_it_names_all_three_tracks(self):
        text = DOC.read_text(encoding="utf-8")
        for track in ("BASIC5_FAIR_v1", "BASIC5_FINETUNE_v1",
                      "BASIC5_ATTENTIVE_v1"):
            self.assertIn(track, text,
                          f"the document does not define {track}")


class TestFairRulesAreEnumerated(unittest.TestCase):
    def test_every_canonical_rule_id_is_documented(self):
        ids = parse_rule_ids(DOC.read_text(encoding="utf-8"))
        missing = CANONICAL_FAIR_RULE_IDS - ids
        self.assertFalse(missing,
                         f"FAIR/ImageNet rule ids missing from the document: "
                         f"{sorted(missing)}")


class TestRuleParserControls(unittest.TestCase):
    """The parser that decides what counts as a rule needs both controls."""

    def test_positive_a_well_formed_row_is_read(self):
        doc = (f"{RULES_MARKER}\n"
               "| id | scope | requirement | status |\n"
               "|----|-------|-------------|--------|\n"
               "| c  | feature | L2-normalised | conformant |\n"
               "| seed | probe | seeds 0,1,2 | deviation |\n")
        self.assertEqual(parse_rule_ids(doc), {"c", "seed"})

    def test_negative_a_prose_pipe_and_the_header_are_not_rules(self):
        doc = (f"{RULES_MARKER}\n"
               "| id | scope | requirement | status |\n"
               "|----|-------|-------------|--------|\n"
               "| c  | feature | L2-normalised | conformant |\n"
               "\n"
               "Prose about a|b pipes that is not a table row.\n")
        # Only the real row 'c' is a rule; the header, separator, blank and the
        # prose line (a decoy a|b) are not.
        self.assertEqual(parse_rule_ids(doc), {"c"})

    def test_no_marker_means_no_rules(self):
        self.assertEqual(parse_rule_ids("no marker here\n| c | x |\n"), set())


class TestDocumentAndCodeAgreeOnDefault(unittest.TestCase):
    """The doc declares the default saved representation; the CLI must match it,
    so the protocol claim and the tool cannot drift apart."""

    def _declared_default(self) -> str:
        m = DEFAULT_REP_RE.search(DOC.read_text(encoding="utf-8"))
        self.assertIsNotNone(
            m, "the document does not declare default-representation")
        return m.group(1)

    def test_declared_default_is_l2(self):
        self.assertEqual(self._declared_default(), "l2",
                         "FAIR rule c requires the default saved feature to be "
                         "L2-normalised")

    def test_the_cli_default_matches_the_document(self):
        args = _tool().build_parser().parse_args(
            ["--data-root", "/d", "--out", "/o"])
        self.assertEqual(
            args.representation, self._declared_default(),
            "bin/extract-features.py's default representation and the protocol "
            "document have drifted apart")


if __name__ == "__main__":
    unittest.main()
