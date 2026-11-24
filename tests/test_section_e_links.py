"""Open every companion named by Appendix E directly from either landing page."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AppendixNavigation(unittest.TestCase):
    def test_every_cited_document_has_a_direct_file_link(self):
        expected = {f'docs/initial_protocols/INITIAL_{stem}_v1.md' for stem in (
            'STEP1', 'STEP2', 'STEP3_3DFM', 'STEP3_4DFM', 'STEP3_VISION_GENERATIVE',
            'STEP3_VIDEO_SSL', 'STEP3_VIDEO_WORLD_MODELS', 'STEP3_VLM')}
        for track in ('LINEAR', 'ATTENTIVE', 'FINETUNE'):
            for name in (f'BASIC5_{track}.md', f'EXTEND_{track}.md', f'EXTEND_{track}_v1.json'):
                expected.add('docs/submission_protocols/' + name)
        expected.add('docs/submission_protocols/BASIC5_FRONTIER_SYSTEM_PROMPTS.md')
        expected.update(f'docs/current_protocols/BASIC5_{track}_v2.{ext}'
                        for track in ('LORA', 'DPT') for ext in ('md', 'json'))
        for page in ('README.md', 'docs/templates/CVPR_REVIEWER_README.md'):
            with self.subTest(page=page):
                links = {target.partition('#')[0] for target in re.findall(
                    r'\]\(([^)]+)\)', (ROOT / page).read_text())}
                self.assertFalse(expected - links, sorted(expected - links))
                for target in expected:
                    self.assertTrue((ROOT / target).is_file(), target)
