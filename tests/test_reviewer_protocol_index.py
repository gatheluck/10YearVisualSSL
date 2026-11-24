"""Visitors to the repository and the submitted archive reach the same protocols."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'docs/templates/CVPR_REVIEWER_README.md'


class ReviewerProtocolIndex(unittest.TestCase):
    def test_root_and_review_copy_link_to_the_same_protocols(self):
        indexes = []
        for page in (ROOT / 'README.md', TEMPLATE):
            self.assertTrue(page.is_file(), page.name)
            text = page.read_text()
            self.assertIn('## Evaluation protocols\n', text)
            section = text.split('## Evaluation protocols\n', 1)[1].split('\n## ', 1)[0]
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
            for link in re.findall(r'\]\(([^)]+)\)', section):
                path, _, anchor = link.partition('#')
                target = ROOT / path
                self.assertTrue(target.is_file(), link)
                if anchor:
                    headings = re.findall(r'^#+ (.+)$', target.read_text(), re.M)
                    self.assertIn(anchor, [re.sub(r'[^\w -]', '', heading.lower()).replace(' ', '-')
                                           for heading in headings], link)
            indexes.append(section.strip())
        self.assertEqual(*indexes)
