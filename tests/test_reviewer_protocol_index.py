"""Visitors to the repository and the submitted archive reach the same protocols."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'docs/templates/CVPR_REVIEWER_README.md'


def protocol_section(page):
    text = page.read_text()
    heading = '<summary>Evaluation protocols</summary>' if page == ROOT / 'README.md' else '## Evaluation protocols'
    _, found, rest = text.partition(heading + '\n')
    if not found:
        raise AssertionError(f'{page.name} has no evaluation protocol index')
    end = '\n</details>' if page == ROOT / 'README.md' else '\n## '
    return rest.split(end, 1)[0]


class ReviewerProtocolIndex(unittest.TestCase):
    def test_paper_metric_and_completed_repeat_notes_are_reachable(self):
        for page in (ROOT / 'README.md', TEMPLATE):
            links = set(re.findall(r'\]\(([^)]+)\)', page.read_text()))
            target = 'docs/BASIC5_V1_AUDIT.md#coco-evaluation-and-completed-repeats'
            with self.subTest(page=page):
                self.assertIn(target, links)
        guide = (ROOT / 'docs/BASIC5_V1_AUDIT.md').read_text()
        notes = guide.split('## COCO evaluation and completed repeats', 1)[-1].split('\n## ', 1)[0]
        for term in ('Table 26', 'n=1', 'Table 29', 'Micro-F1', 'bounding-box AP'):
            self.assertIn(term, notes)

    def test_readers_can_distinguish_completed_table_26_runs_from_scheduled_seeds(self):
        for page in (ROOT / 'README.md', TEMPLATE):
            with self.subTest(page=page.name):
                section = protocol_section(page)
                note = next((line for line in section.splitlines() if 'Table 26' in line), '')
                self.assertRegex(note, r'18 of 25[^.]*three runs')
                self.assertRegex(note, r'VGGT-Omega on SSv2[^.]*one run')
                self.assertRegex(note, r'six[^.]*blank')

    def test_root_and_review_copy_link_to_the_same_protocols(self):
        indexes = []
        for page in (ROOT / 'README.md', TEMPLATE):
            self.assertTrue(page.is_file(), page.name)
            section = protocol_section(page)
            rows = [line for line in section.splitlines() if line.startswith('| ')]
            self.assertEqual([r.split('|')[1].strip().split(' (')[0] for r in rows[2:]],
                             ['Linear probing', 'Attentive probing', 'Full fine-tuning',
                              'LoRA', 'DPT dense probing'])
            for row, (track, version) in zip(rows[2:], [
                ('LINEAR', 'v1'), ('ATTENTIVE', 'v1'), ('FINETUNE', 'v1'),
                ('LORA', 'v2'), ('DPT', 'v2'),
            ], strict=True):
                prefix = f'docs/current_protocols/BASIC5_{track}_{version}'
                self.assertIn(f'({prefix}.md)', row)
                self.assertIn(f'({prefix}.json)', row)
                spec = json.loads((ROOT / (prefix + '.json')).read_text())
                self.assertEqual(spec['global_rules']['seeds'], [0, 1, 2])
                tasks = {'ade20k', 'nyuv2'} if track == 'DPT' else {
                    'imagenet1k', 'ssv2', 'coco', 'ade20k', 'nyuv2'}
                self.assertEqual(set(spec['tasks']), tasks)
            links = re.findall(r'\]\(([^)]+)\)', section)
            for link in links:
                path, _, anchor = link.partition('#')
                target = ROOT / path
                self.assertTrue(target.is_file(), link)
                if anchor:
                    headings = re.findall(r'^#+ (.+)$', target.read_text(), re.M)
                    self.assertIn(anchor, [re.sub(r'[^\w -]', '', heading.lower()).replace(' ', '-')
                                           for heading in headings], link)
            indexes.append(links)
        self.assertEqual(*indexes)
