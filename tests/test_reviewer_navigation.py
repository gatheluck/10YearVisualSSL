"""Check the reviewer's first action and the route from the paper to its files."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReviewerNavigation(unittest.TestCase):
    def test_overview_links_to_protocols_and_appendix_before_quick_check(self):
        for page in ('README.md', 'docs/templates/CVPR_REVIEWER_README.md'):
            with self.subTest(page=page):
                text = (ROOT / page).read_text()
                overview, *sections = re.split(r'^## ', text, flags=re.M)
                self.assertEqual(sections[0].splitlines()[0], 'Quick verification')
                for anchor in ('evaluation-protocols', 'files-cited-in-appendix-e'):
                    self.assertIn(f'](#{anchor})', overview)
                self.assertIn('python -m unittest', sections[0])

    def test_review_archive_reading_order(self):
        page = ROOT / 'docs/templates/CVPR_REVIEWER_README.md'
        self.assertEqual(re.findall(r'^## (.+)$', page.read_text(), re.M), [
            'Quick verification', 'Evaluation protocols', 'Figures and analyses',
            'Reading the results', 'Files cited in Appendix E', 'Data, dependencies and notices',
        ])
