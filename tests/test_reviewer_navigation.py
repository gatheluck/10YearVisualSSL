"""Check the reviewer's first action and the route from the paper to its files."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReviewerNavigation(unittest.TestCase):
    def test_quick_check_selects_source_only_then_full_probe_coverage(self):
        text = (ROOT / 'OPERATING_GUIDE.md').read_text()
        section = text.split('## Quick verification\n', 1)[1].split('\n## ', 1)[0]
        blocks = re.findall(r'^```sh\n(.*?)^```$', section, re.M | re.S)
        suites = [line.split()[3:] for block in blocks for line in block.splitlines()
                  if line.startswith('python -m unittest ')]
        shared = ['tests.test_figure3_comparison', 'tests.test_cvpr_extended_coverage',
                  'tests.test_transfer_comparison']
        self.assertCountEqual(suites[0], shared + [
            'tests.test_dpt_probe.DenseProbe.test_missing_submodule_reports_actionable_error'])
        self.assertEqual(len(suites), 2)
        self.assertCountEqual(suites[1], shared + ['tests.test_dpt_probe'])
        commands = [line for block in blocks for line in block.splitlines()]
        setup = 'git submodule update --init --checkout -- third_party/geosubprobe'
        self.assertIn(setup, commands)
        self.assertLess(commands.index('python -m unittest ' + ' '.join(suites[0])),
                        commands.index(setup))
        self.assertLess(commands.index(setup),
                        commands.index('python -m unittest ' + ' '.join(suites[1])))

    def test_overview_links_to_protocols_and_appendix_before_quick_check(self):
        for page in ('OPERATING_GUIDE.md', 'docs/templates/CVPR_REVIEWER_README.md'):
            with self.subTest(page=page):
                text = (ROOT / page).read_text()
                overview, *sections = re.split(r'^## ', text, flags=re.M)
                self.assertEqual(sections[0].splitlines()[0], 'Quick verification')
                for anchor in ('evaluation-protocols', 'files-cited-in-appendix-e'):
                    prefix = 'README.md' if page == 'OPERATING_GUIDE.md' else ''
                    self.assertIn(f']({prefix}#{anchor})', overview)
                self.assertIn('python -m unittest', sections[0])

    def test_review_archive_reading_order(self):
        page = ROOT / 'docs/templates/CVPR_REVIEWER_README.md'
        self.assertEqual(re.findall(r'^## (.+)$', page.read_text(), re.M), [
            'Quick verification', 'Evaluation protocols', 'Figures and analyses',
            'Reading the results', 'Files cited in Appendix E', 'Data, dependencies and notices',
        ])
