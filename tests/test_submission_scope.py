"""Readers of each portable-package entry point can reach its scope guide.

These tests validate navigation and delivery, not the truth of scientific prose.
That distinction requires a manual reference and evidence review.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scope_links(document):
    return [target for target in re.findall(r"\]\(([^)]+)\)", document)
            if Path(target).name == "SUBMISSION_SCOPE.md"]


class TestSubmissionScope(unittest.TestCase):
    def test_scope_link_detection_uses_complete_filename(self):
        self.assertEqual(scope_links("[scope](../SUBMISSION_SCOPE.md)"),
                         ["../SUBMISSION_SCOPE.md"])
        self.assertEqual(scope_links("SUBMISSION_SCOPE.md [other](OLD_SUBMISSION_SCOPE.md)"), [])

    def test_every_package_entry_point_links_to_existing_scope_guide(self):
        guide = ROOT / "docs/SUBMISSION_SCOPE.md"
        entries = [ROOT / "README.md", *sorted((ROOT / "methods").glob("*/README.md")),
                   *sorted((ROOT / "docs").glob("*.md")),
                   ROOT / "docs/submission_protocols/README.md"]
        self.assertTrue(guide.is_file(), "portable-package scope guide missing")
        for entry in entries:
            if entry == guide:
                continue
            with self.subTest(entry=entry.relative_to(ROOT)):
                links = scope_links(entry.read_text())
                self.assertTrue(links, "no scope navigation from this entry point")
                for target in links:
                    self.assertEqual((entry.parent / target).resolve(), guide.resolve())
                    self.assertEqual((entry.parent / target).read_bytes(), guide.read_bytes())


if __name__ == "__main__":
    unittest.main()
