"""Follow the paper guide from the submission scope to each experiment route."""
from pathlib import Path
import re
import unittest


DOCS = Path(__file__).resolve().parents[1] / 'docs'


class PaperGuideNavigation(unittest.TestCase):
    def test_reader_can_reach_each_figure_and_its_local_instructions(self):
        links = re.compile(r'\[[^\]]+\]\(([^)]+)\)')
        target = 'PAPER_REPRODUCTION_GUIDE.md'
        self.assertIn(target, links.findall((DOCS / 'SUBMISSION_SCOPE.md').read_text()))
        guide = (DOCS / target).read_text()
        sections = re.split(r'^## ', guide, flags=re.MULTILINE)[1:]
        figures = [section for section in sections if section.startswith('Figure ')]
        self.assertEqual([section.split(' — ', 1)[0] for section in figures],
                         [f'Figure {i}' for i in range(1, 8)])
        for section in figures:
            self.assertTrue(links.findall(section), section.splitlines()[0])
        for link in links.findall(guide):
            self.assertTrue((DOCS / link.partition('#')[0]).is_file(), link)

    def test_archive_readme_links_to_guide_and_receives_its_return_links(self):
        readme = (DOCS / 'templates/CVPR_REVIEWER_README.md').read_text()
        guide = (DOCS / 'PAPER_REPRODUCTION_GUIDE.md').read_text()
        fragments = re.findall(r'\]\(\.\./README\.md#([^)]*)\)', guide)
        headings = [re.sub(r'[^\w -]', '', h.lower()).replace(' ', '-')
                    for h in re.findall(r'^#+ (.+)$', readme, re.M)]
        self.assertFalse(set(fragments) - set(headings), set(fragments) - set(headings))
        self.assertIn('docs/PAPER_REPRODUCTION_GUIDE.md', re.findall(r'\]\(([^)]+)\)', readme))

    def test_archive_quick_check_selects_only_available_probe_tests_first(self):
        readme = (DOCS / 'templates/CVPR_REVIEWER_README.md').read_text()
        quick = readme.split('## Quick verification\n', 1)[1].split('\n## ', 1)[0]
        commands = re.findall(r'^python -m unittest (.+)$', quick, re.M)
        probe = 'tests.test_dpt_probe'
        self.assertIn(probe + '.DenseProbe.test_missing_submodule_reports_actionable_error',
                      commands[0].split())
        self.assertNotIn(probe, commands[0].split())
        self.assertEqual(len(commands), 2)
        self.assertIn(probe, commands[1].split())
        self.assertIn('third_party/geosubprobe/evals/models/probes.py',
                      quick.split(commands[0], 1)[1].split(commands[1], 1)[0])

    def test_figure7_count_agrees_with_paired_reported_correlations(self):
        source = (DOCS / 'REPRESENTATION_ANALYSIS.md').read_text()
        block = source.split('### Check against the original figure tables\n', 1)[1].lstrip().split('\n\n', 1)[0]
        pairs = re.findall(r'(0\.\d{4})\s+and\s+(0\.\d{4})\s+for (\S+)', block)
        self.assertEqual(len(pairs), 5)
        guide = (DOCS / 'PAPER_REPRODUCTION_GUIDE.md').read_text()
        figure = guide.split('## Figure 7 —', 1)[1].split('\n## ', 1)[0]
        count = re.search(r'(\d+) pooled Spearman correlations', figure)
        self.assertIsNotNone(count, 'report the total across both predictors and all five tasks')
        self.assertEqual(int(count.group(1)), 2 * len(pairs))
