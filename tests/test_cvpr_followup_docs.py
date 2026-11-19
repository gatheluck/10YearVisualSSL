"""Current coverage navigation resolves to versioned executable-scope guides."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CurrentNavigation(unittest.TestCase):
    def test_scope_and_historical_ledger_link_to_current_guides(self):
        targets = {'CVPR_ADAPTATION.md', 'CVPR_EXTENDED_COVERAGE.md', 'REPRESENTATION_ANALYSIS.md'}
        for name in ('SUBMISSION_SCOPE.md', 'FINAL_PAPER_COVERAGE.md'):
            path = ROOT / 'docs' / name
            links = re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)', path.read_text())
            resolved = {Path(link).name for link in links if (path.parent / link).is_file()}
            with self.subTest(document=name):
                self.assertTrue(targets <= resolved, targets - resolved)

    def test_analysis_guide_has_complete_cli_example_and_scope_boundaries(self):
        path = ROOT / 'docs' / 'REPRESENTATION_ANALYSIS.md'
        text = path.read_text()
        self.assertRegex(text, r'python -m downstream\.representation_analysis --config \S+ --out \S+')
        for filename in ('representation_analysis.py', 'representation_neighbors.py', 'representation_plane.py'):
            self.assertTrue((ROOT / 'downstream' / filename).is_file())
        for concept in ('sample_ids', 'image_reference', 'patch_reference', 'record_value', '10,000', '2,000'):
            self.assertIn(concept, text)


if __name__ == '__main__':
    unittest.main()
