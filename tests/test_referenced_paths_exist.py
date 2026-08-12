"""Paths named in the docs must exist. A rename that misses a doc reference
leaves a command pointing at a file that is not there -- e.g. after configs and
mutation specs were renamed step1 -> pretrain, some READMEs still cited the old
names. Discover, never list: scan the tracked prose/configs and check each
referenced repo path resolves. Capture-side citations (paths that do not live in
this repo) are matched only for our own trees (methods/*/configs, mutations/)."""

from __future__ import annotations

import glob
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUT = re.compile(r"mutations/[\w.\-]+\.json")
CFG = re.compile(r"--config\s+(methods/[\w./\-]+\.ya?ml)")


def _docs() -> list[Path]:
    # Shipped-documentation surface only. Test files are deliberately excluded:
    # they legitimately carry synthetic placeholder paths (e.g. methods/x/...)
    # that are not meant to resolve, so a path-existence check there is unsound.
    pats = ["methods/*/README.md", "methods/*/configs/*.yaml",
            "methods/*/provenance.json", "docs/*.md", "README.md"]
    return [Path(x) for p in pats for x in glob.glob(str(ROOT / p))]


def missing_refs():
    bad = []
    for f in _docs():
        text = f.read_text(encoding="utf-8")
        for m in MUT.findall(text):
            if not (ROOT / m).is_file():
                bad.append(f"{f.relative_to(ROOT)} -> {m}")
        for c in CFG.findall(text):
            if not (ROOT / c).is_file():
                bad.append(f"{f.relative_to(ROOT)} -> --config {c}")
    return sorted(set(bad))


class TestReferencedRepoPathsExist(unittest.TestCase):
    def test_documented_mutation_and_config_paths_exist(self):
        bad = missing_refs()
        self.assertEqual(bad, [], f"docs reference repo paths that do not exist "
                                  f"(stale after a rename?): {bad}")


class TestTheDetectorFires(unittest.TestCase):
    def test_it_flags_a_missing_path_and_clears_a_real_one(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(
            __import__("shutil").rmtree, tmp, ignore_errors=True)
        (tmp / "x.md").write_text("see mutations/nope-does-not-exist.json\n")
        # reuse the same logic on a controlled file
        text = (tmp / "x.md").read_text()
        self.assertTrue([m for m in MUT.findall(text) if not (ROOT / m).is_file()])
        real = "mutations/" + Path(sorted(glob.glob(str(ROOT/"mutations/*.json")))[0]).name
        self.assertTrue((ROOT / real).is_file())


if __name__ == "__main__":
    unittest.main()
