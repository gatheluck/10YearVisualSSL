"""Reviewer navigation must distinguish manuscript terms from legacy labels."""
import re
import unittest
from tests._repo_files import repository_files
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
LEGACY = re.compile(r'\bstep[ _-]*[1-4]\b', re.I)
GUIDE = ROOT / 'docs/PAPER_TERMINOLOGY.md'

class PaperTerminology(unittest.TestCase):
    def test_reviewer_navigation_uses_manuscript_terms(self):
        text = (ROOT / 'docs/templates/REVIEWER_README.md').read_text()
        rows = [line for line in text.splitlines() if line.startswith('|')]
        self.assertGreater(len(rows), 5)
        for row in rows:
            self.assertFalse(LEGACY.search(row.split('|')[1]), row)
        for term in ('ASIS', 'CTRL', 'PAPER_TERMINOLOGY.md'):
            self.assertIn(term, text)

    def test_legacy_guides_offer_resolvable_mapping(self):
        paths = [p for p in repository_files(ROOT) if p.suffix == '.md' and p.relative_to(ROOT).parts[0] in ('docs', 'methods')]
        checked = 0
        for p in paths:
            if p == GUIDE or 'submission_protocols' in p.parts or 'templates' in p.parts:
                continue  # supplied protocol originals remain unchanged
            if not LEGACY.search(p.read_text()):
                continue
            checked += 1
            links = re.findall(r'\]\(([^)]+)\)', p.read_text())
            self.assertTrue(any((p.parent / unquote(x.split('#')[0])).resolve() == GUIDE.resolve()
                                and (p.parent / unquote(x.split('#')[0])).is_file()
                                for x in links), str(p))
        self.assertGreater(checked, 20)

    def test_mapping_covers_scope_without_renaming_interfaces(self):
        self.assertTrue(GUIDE.is_file(), "paper terminology guide is missing")
        text = GUIDE.read_text()
        for term in ('ASIS', 'CTRL', 'Section 3', 'Section 4', 'Section 5.1', 'Section 5.2', 'Appendix E', 'LP', 'AP', 'FT'):
            self.assertIn(term, text)
        self.assertIn('not a one-to-one', text)
        self.assertIn('identifiers', text)

    def test_initial_coco_is_detection_without_changing_unified_registry(self):
        for p in (ROOT / 'docs/initial_protocols').glob('*.md'):
            with self.subTest(path=p.name):
                self.assertNotRegex(p.read_text(), r'(?i)multi[ -]?label')
        for name in ('VIDEO_SSL', 'VIDEO_WORLD_MODELS', 'VLM'):
            text = (ROOT / f'docs/initial_protocols/INITIAL_STEP3_{name}_v1.md').read_text()
            self.assertIn('Faster R-CNN', text)
            self.assertIn('## Historical evidence and its limits', text)
            self.assertIn('## Proposed changes and unresolved choices', text)
            self.assertIn('2026-09-26 task correction', text)
            self.assertNotIn('COCO_GLOBAL_MULTILABEL_MAP_v1', text)

    def test_template_links_resolve_at_export_root(self):
        text = (ROOT / 'docs/templates/REVIEWER_README.md').read_text()
        links = re.findall(r'\]\(([^)]+)\)', text)
        self.assertGreater(len(links), 15)
        for link in links:
            with self.subTest(link=link):
                self.assertTrue((ROOT / unquote(link.split('#')[0])).exists())

if __name__ == '__main__':
    unittest.main()
