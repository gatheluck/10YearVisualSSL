#!/usr/bin/env python3
"""A method that HAS a unified ViT-B/16 Step 2 must not say its Step 2 is absent.

The Step-2 fan-out added a from-scratch ViT-B/16 pretraining to ~34 methods. Some
methods' README/provenance were left claiming "the ViT step 2 is excluded" / "not
ported" / "eval-only port" even though the Step 2 was in fact added -- documentation
that contradicts the code. This guard mechanises the invariant so the drift cannot
recur (a policy in a document does not hold; make it machinery -- CLAUDE.md).

A method "has a Step 2" iff it ships `configs/pretrain_vit.yaml` OR its
`configs/pretrain.yaml` declares `save_at_epochs` (the milestone-only methods
31_dinov3 / 35_vjepa). For such a method, no sentence of its README.md or
provenance.json may assert the Step 2 is absent.

The detector is sentence-scoped (a step-2 mention and an exclusion word must land in
the *same* sentence), so a method that legitimately excludes something else in a
different sentence (e.g. an optional ResNet variant) is not flagged. It carries a
positive and a negative control, per the repo's detector rule.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"

# A step-2 mention, and an assertion of absence. Both must occur in one sentence.
# "step 2", "step2" and "step-2" all count (a hyphen must not be a way past the
# guard).
_STEP2 = re.compile(r"\bstep[\s-]?2\b", re.IGNORECASE)
_EXCLUSION = re.compile(
    r"\b(exclud\w*|not ported|not brought across|no place in this port|"
    r"has no place|is not part of this port)\b", re.IGNORECASE)
# A whole-method "eval-only port" claim (distinct from the correct phrase
# "the eval-only Step-1 path/download", which describes only the Step-1 stage).
# Singular and plural both count -- a status line summarising "X, Y and Z are
# eval-only ports" is the same whole-port claim.
_EVAL_ONLY_PORT = re.compile(r"\beval-only ports?\b", re.IGNORECASE)

# A WHOLE-PORT no-training / no-encoder claim. 28_dinov2 / 30_aim / 36_franca began
# as eval-only ports whose docs said the whole port "trains nothing" and that
# "there is no encoder.pt". They gained a from-scratch Step-2 pretrain that DOES
# train and DOES write encoder.pt, so such a whole-port claim is now false.
#
# The signal must be the WHOLE-PORT phrasing, not any no-encoder mention: a
# per-stage note ("the linear_eval stage produces a classifier, not an encoder";
# "the Step-1 as-is row writes no encoder.pt") is correct and common, and every
# method's eval stage carries one. So this detector matches only the whole-port
# forms ("this port trains nothing", "there is no encoder.pt"), and a scope word in
# the same sentence still rescues even those (e.g. "...for the Step-1 as-is row").
_NO_TRAINING = re.compile(
    r"this port trains nothing|there is no `?encoder\b", re.IGNORECASE)
_SCOPE_WORD = re.compile(
    r"step[\s-]?1|as-is|download|probe|linear[_ ]?eval|official|frozen backbone",
    re.IGNORECASE)


def _sentences(text: str) -> list:
    """Split into sentences/clauses on ';', newlines, and sentence-final '.' (one
    followed by whitespace or end-of-string) -- coarse but enough to keep an
    exclusion claim from binding to a step-2 mention in another clause. A period
    inside a token like `encoder.pt` or `1.0` is not a boundary, so those stay
    whole."""
    return [s.strip() for s in re.split(r"[;\n]|\.(?=\s|$)", text) if s.strip()]


def stale_step2_claims(text: str) -> list:
    """Sentences that wrongly assert a Step 2 is absent."""
    hits = []
    for sent in _sentences(text):
        if _EVAL_ONLY_PORT.search(sent):
            hits.append(sent)
        elif _STEP2.search(sent) and _EXCLUSION.search(sent):
            hits.append(sent)
    return hits


def blanket_no_encoder_claims(text: str) -> list:
    """Sentences claiming no training / no encoder without scoping it to Step 1."""
    return [sent for sent in _sentences(text)
            if _NO_TRAINING.search(sent) and not _SCOPE_WORD.search(sent)]


def _provenance_text(path: Path) -> str:
    """All string values of provenance.json, joined -- so scope/note/rewritten are
    all scanned regardless of shape (list or dict)."""
    parts = []

    def walk(v):
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(json.loads(path.read_text(encoding="utf-8")))
    return "\n".join(parts)


def has_step2(method_dir: Path) -> bool:
    if (method_dir / "configs" / "pretrain_vit.yaml").is_file():
        return True
    native = method_dir / "configs" / "pretrain.yaml"
    return native.is_file() and "save_at_epochs" in native.read_text(encoding="utf-8")


def step2_methods() -> list:
    return sorted(m for m in METHODS.iterdir()
                  if m.is_dir() and m.name[0].isdigit() and has_step2(m))


_README_ROW = re.compile(r"^\|\s*`(\d+_[a-z0-9_]+)`\s*\|(.*)$")


def readme_table_rows() -> dict:
    """The root README Methods table, as {method_name: row_text}. The root table
    is a separate file from the per-method READMEs, so the per-method guard above
    does not cover it -- a row can still deny a Step 2 the method now ships."""
    rows = {}
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        m = _README_ROW.match(line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


# Status/roadmap docs are prose, not the per-method files or the Methods table, so
# the guards above miss them. These carry a *current-status* claim per method (e.g.
# the README Status row, the roadmap's ported/what-is-excluded column), so a
# present-tense denial of a shipped Step 2 here is drift. Accurate *history* lives
# in STEP2_VIT_PORTING.md / STEP2_CONSISTENCY_AUDIT.md ("were first ported
# eval-only", "converting the trio to two-stage"), which are deliberately NOT
# scanned.
STATUS_DOCS = ("README.md", "docs/PORTING_ROADMAP.md")

# A Methods-table row -- `| `name` | ... |`. The row is ABOUT its first-column
# method; a mention of another method in the notes ("analogous to `36_franca`") is
# a cross-reference, not a claim about that other method. So a table row's claim is
# attributed to its subject alone, never to every method token on the line -- a
# line-wide attribution is the "substring/scope too wide" mistake this repo keeps
# re-making (a `36_franca` mention in eva02's eval-only row is not eva02 denying
# 36_franca's Step 2). `name` may be numbered (`36_franca`) or unnumbered (`eva02`).
_TABLE_ROW_SUBJECT = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|")


def _scan_status_lines(lines, names) -> list:
    """(method, sentence) for every line that denies a Step-2 method's Step 2.

    A Methods-table row is attributed to its subject alone (and flagged only when
    that subject is itself a Step-2 method); any other prose line is attributed to
    every Step-2 method it names by its `NN_name` token. Pure over its inputs so
    the detector can be driven by both a positive and a negative control."""
    tokens = {name: re.compile(r"`" + re.escape(name) + r"`") for name in names}
    out = []
    for line in lines:
        clean = line.replace("*", "").replace("`", "")
        hits = stale_step2_claims(clean)
        if not hits:
            continue
        row = _TABLE_ROW_SUBJECT.match(line)
        if row:
            subject = row.group(1)
            if subject in names:
                out += [(subject, sent) for sent in hits]
            continue
        for name in names:
            if tokens[name].search(line):
                out += [(name, sent) for sent in hits]
    return out


def status_doc_denials() -> list:
    """(doc, method, sentence) for every status-doc line that references a Step-2
    method and denies its Step 2. Discovery-based: methods are found on disk, never
    listed."""
    names = [m.name for m in step2_methods()]
    out = []
    for doc in STATUS_DOCS:
        path = ROOT / doc
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        out += [(doc, name, sent)
                for name, sent in _scan_status_lines(lines, names)]
    return out


class TestStep2DocsDoNotDenyTheStep2(unittest.TestCase):
    def test_there_are_step2_methods_to_check(self):
        """A guard over an empty set proves nothing."""
        self.assertGreater(len(step2_methods()), 20)

    def test_no_step2_method_claims_its_step2_is_absent(self):
        offences = []
        for m in step2_methods():
            for doc in ("README.md", "provenance.json"):
                p = m / doc
                if not p.is_file():
                    continue
                text = _provenance_text(p) if doc == "provenance.json" else \
                    p.read_text(encoding="utf-8")
                for sent in stale_step2_claims(text):
                    offences.append(f"{m.name}/{doc}: {sent[:120]}")
        self.assertEqual(
            offences, [],
            "these methods have a Step 2 but their docs deny it:\n"
            + "\n".join(f"  - {o}" for o in offences))

    def test_no_step2_method_makes_a_blanket_no_encoder_claim(self):
        offences = []
        for m in step2_methods():
            for doc in ("README.md", "provenance.json"):
                p = m / doc
                if not p.is_file():
                    continue
                text = _provenance_text(p) if doc == "provenance.json" else \
                    p.read_text(encoding="utf-8")
                for sent in blanket_no_encoder_claims(text):
                    offences.append(f"{m.name}/{doc}: {sent[:120]}")
        self.assertEqual(
            offences, [],
            "these methods write encoder.pt (Step 2) but their docs claim the "
            "whole port trains nothing / has no encoder.pt (scope it to Step 1):\n"
            + "\n".join(f"  - {o}" for o in offences))

    def test_the_readme_table_row_does_not_deny_the_step2(self):
        rows = readme_table_rows()
        names = {m.name for m in step2_methods()}
        offences = []
        for name in sorted(names):
            row = rows.get(name)
            if row is None:
                continue
            # Strip markdown emphasis/code marks so a bolded claim
            # ("**eval-only** port") is not a way past the detector.
            clean = row.replace("*", "").replace("`", "")
            for sent in stale_step2_claims(clean):
                offences.append(f"{name}: {sent[:120]}")
        self.assertEqual(
            offences, [],
            "these methods have a Step 2 but their root README table row denies "
            "it:\n" + "\n".join(f"  - {o}" for o in offences))

    def test_the_readme_table_was_actually_read(self):
        """A guard over an empty parse proves nothing."""
        self.assertGreater(len(readme_table_rows()), 30)

    def test_status_docs_do_not_deny_a_step2(self):
        offences = [f"{doc} [{name}]: {sent[:110]}"
                    for doc, name, sent in status_doc_denials()]
        self.assertEqual(
            offences, [],
            "these status docs (README Status/prose, PORTING_ROADMAP) deny a "
            "Step 2 that is now ported:\n" + "\n".join(f"  - {o}" for o in offences))

    def test_the_status_docs_reference_step2_methods(self):
        """A guard that references no method proves nothing: every status doc must
        actually mention some Step-2 method (else the scan is vacuous)."""
        names = [m.name for m in step2_methods()]
        for doc in STATUS_DOCS:
            path = ROOT / doc
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(doc=doc):
                self.assertTrue(any(f"`{n}`" in text for n in names),
                                f"{doc} references no Step-2 method")

    # ---- detector controls (positive + negative) ----
    def test_detector_flags_a_real_denial(self):
        self.assertTrue(stale_step2_claims(
            "The capture's ViT step 2 is EXCLUDED, as in every port."))
        self.assertTrue(stale_step2_claims("Not ported: the ViT step 2."))
        self.assertTrue(stale_step2_claims(
            "Step 2 (ViT-B) has no place in this port."))
        self.assertTrue(stale_step2_claims(
            "So this is an eval-only port, the DINOv2 sibling."))

    def test_detector_ignores_legitimate_text(self):
        # step 2 present, but it is ported -- not a denial
        self.assertFalse(stale_step2_claims(
            "The capture's unified ViT-B/16 Step 2 is ported additively (arch: vit)."))
        # an exclusion, but of something that is not the step 2
        self.assertFalse(stale_step2_claims(
            "The capture's optional ResNet variant remains EXCLUDED, as in every port."))
        # "eval-only" describing only the Step-1 stage, not the whole port
        self.assertFalse(stale_step2_claims(
            "The eval-only Step-1 path is unchanged."))
        # a milestone line
        self.assertFalse(stale_step2_claims(
            "Milestone checkpoints are written at 100/200/300."))

    def test_no_encoder_detector_flags_a_blanket_claim(self):
        self.assertTrue(blanket_no_encoder_claims(
            "This port trains nothing and produces no encoder.pt."))
        self.assertTrue(blanket_no_encoder_claims("There is no encoder.pt."))

    # ---- status-scan attribution controls (positive + negative) ----
    # Fake method names ("cat"/"dog") keep this shared file from hard-coding a
    # real method (tests/test_no_hard_coded_methods.py), exactly as that guard's
    # own controls do.
    def test_status_scan_flags_a_step2_methods_own_row_denial(self):
        # positive: a Step-2 method's OWN table row denying its Step 2 is drift.
        self.assertTrue(_scan_status_lines(
            ["| `cat` | Cat | linear eval | an eval-only port |"], ["cat"]))

    def test_status_scan_flags_a_prose_denial_by_token(self):
        # positive: a non-table prose line naming a Step-2 method and denying it.
        self.assertTrue(_scan_status_lines(
            ["- `cat`: the ViT step 2 is excluded here."], ["cat"]))

    def test_status_scan_ignores_a_cross_reference_in_another_row(self):
        # negative: a non-Step-2 method's row that is itself eval-only and merely
        # cross-references a Step-2 sibling ("cat") is not a denial of that
        # sibling's Step 2.
        self.assertEqual(_scan_status_lines(
            ["| `dog` | Dog | linear eval | a pure eval-only port, "
             "analogous to `cat` |"], ["cat"]), [])

    def test_no_encoder_detector_allows_a_scoped_claim(self):
        # a whole-port claim scoped to the Step-1 as-is/download probe -- not drift
        self.assertFalse(blanket_no_encoder_claims(
            "There is no encoder.pt for the Step-1 as-is row; it probes a download."))
        # a per-stage note (pronoun scope), not a whole-port claim
        self.assertFalse(blanket_no_encoder_claims(
            "The as-is row reuses the official backbone. This trains nothing."))
        # a classifier is not an encoder -- not a whole-port no-encoder claim
        self.assertFalse(blanket_no_encoder_claims(
            "The linear_eval stage produces a classifier, not an encoder module."))


if __name__ == "__main__":
    unittest.main()
