"""Documented `bin/fetch-weights.py` commands must actually work.

A shipped command that names a flag the tool does not have is a trap: a method
once told users to run `--section backbone_artifact`, but the flag is
`--artifact`, so the copy-pasted command failed. The fix is machinery, not care
(CLAUDE.md). Method names are discovered here, never spelled out
(tests/test_no_hard_coded_methods.py).

Two guards, both **discovering** from source rather than listing:

1. `--section` is not a flag of any tool under `bin/`, so it must appear in no
   doc/config/provenance. (Self-adjusting: if a tool ever defines `--section`
   the guard steps aside and asks to be updated.)
2. Every documented `--artifact <name>` names a real `*_artifact` section in the
   method whose provenance the command references.
"""

from __future__ import annotations

import ast
import glob
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
FETCH = "fetch-weights.py"


def tool_flags() -> set[str]:
    """Every long option defined by any tool under bin/, read from argparse."""
    flags: set[str] = set()
    for tool in glob.glob(str(BIN / "*.py")):
        tree = ast.parse(Path(tool).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                            and a.value.startswith("--"):
                        flags.add(a.value)
    return flags


def _text_files() -> list[Path]:
    pats = ["methods/*/README.md", "methods/*/configs/*.yaml",
            "methods/*/provenance.json", "docs/*.md", "README.md"]
    files: list[Path] = []
    for p in pats:
        files += [Path(x) for x in glob.glob(str(ROOT / p))]
    return files


def _provenance_sections(method: str) -> set[str]:
    p = ROOT / "methods" / method / "provenance.json"
    if not p.is_file():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k for k in d if k.endswith("_artifact")}


def _a_download_method() -> "tuple[str, str]":
    """Discover any method that declares an artifact section, and one section
    name -- so the controls never spell a method out."""
    for d in sorted((ROOT / "methods").iterdir()):
        secs = _provenance_sections(d.name)
        if secs:
            return d.name, sorted(secs)[0]
    raise AssertionError("no method declares an artifact section")


def fetch_windows(text: str) -> list[str]:
    """Each fetch-weights invocation as its line plus the next two, so a command
    wrapped across lines (a shell block, or a `--provenance ... --artifact ...`
    prose comment) is seen whole. Used only for the artifact-name check, which
    keys on `--artifact` AND a provenance path being present together."""
    lines = text.splitlines()
    return ["\n".join(lines[i:i + 3]) for i, ln in enumerate(lines)
            if FETCH in ln]


class TestNoInventedFetchFlag(unittest.TestCase):
    def test_section_is_not_used_because_no_tool_defines_it(self):
        self.assertNotIn(
            "--section", tool_flags(),
            "a bin/ tool now defines --section; update this guard to check it "
            "like a real flag rather than forbidding it")
        offenders = [str(f.relative_to(ROOT)) for f in _text_files()
                     if "--section" in f.read_text(encoding="utf-8")]
        self.assertEqual(
            offenders, [],
            f"--section is not a flag of any bin/ tool, but these files use it "
            f"(did they mean --artifact?): {offenders}")


class TestDocumentedArtifactNamesExist(unittest.TestCase):
    def test_every_documented_artifact_names_a_real_section(self):
        bad = []
        for f in _text_files():
            for win in fetch_windows(f.read_text(encoding="utf-8")):
                m_art = re.search(r"--artifact\s+([a-z_]+)", win)
                m_prov = re.search(r"methods/([^/ ]+)/provenance\.json", win)
                if m_art and m_prov:
                    name, method = m_art.group(1), m_prov.group(1)
                    if name not in _provenance_sections(method):
                        bad.append(
                            f"{f.relative_to(ROOT)}: --artifact {name} is not a "
                            f"section of methods/{method}")
        self.assertEqual(bad, [], f"artifact names that do not exist: {bad}")


class TestTheDetectorsFire(unittest.TestCase):
    """A guard that cannot fail is not a guard."""

    def test_section_detector(self):
        self.assertIn("--section", "run --section backbone")   # sanity of the token
        self.assertNotIn("--section", "run --artifact backbone")

    def test_artifact_name_check_flags_a_wrong_name(self):
        method, real_section = _a_download_method()   # discovered, not named
        win = (f"bin/fetch-weights.py --provenance methods/{method}/"
               f"provenance.json --artifact not_a_section --out d")
        m_art = re.search(r"--artifact\s+([a-z_]+)", win)
        m_prov = re.search(r"methods/([^/ ]+)/provenance\.json", win)
        self.assertTrue(m_art and m_prov)
        self.assertNotIn(m_art.group(1),
                         _provenance_sections(m_prov.group(1)))
        # and the discovered real section passes
        self.assertIn(real_section, _provenance_sections(method))


if __name__ == "__main__":
    unittest.main()
