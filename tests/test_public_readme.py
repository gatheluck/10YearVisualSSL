"""Exercise the public paper entry points from a source checkout."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAGE = 'https://gatheluck.github.io/10YearVisualSSL/'
TITLE = 'Revisiting Vision Model Baselines in the Deep Learning Era'
LINK = re.compile(r'\[([^]\n]+)\]\(([^)]+)\)')


def anchors(text):
    return {re.sub(r'[^\w -]', '', h.lower()).replace(' ', '-')
            for h in re.findall(r'^#+ (.+)$', text, re.M)}


class PublicReadme(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / 'README.md').read_text()

    def test_published_identity_and_resources(self):
        self.assertEqual(self.text.splitlines()[0], '# ' + TITLE)
        links = dict(LINK.findall(self.text))
        self.assertEqual(links.get('Project Page'), PAGE)
        self.assertEqual(links.get('Citation'), '#citation')
        self.assertEqual(links.get('BibTeX'), PAGE + '#citation')
        citation = self.text.split('## Citation\n')[1].split('\n## ')[0]
        self.assertIn('Hirokatsu Kataoka, Kohsuke Ide, Yoshihiro Fukuhara, Shumpei Takezaki, '
                      'Nakamasa Inoue, Yuki M Asano and Christian Rupprecht.', citation)
        self.assertIn(f'*{TITLE}*. Manuscript, 2026.', citation)
        self.assertIn('tests.yml/badge.svg?branch=main)', self.text)
        self.assertIn('actions/workflows/tests.yml?query=branch%3Amain)', self.text)

    def test_figures_open_their_own_instructions_and_sources(self):
        rows = [line for line in self.text.splitlines() if re.match(r'^\| Figure \d+ \|', line)]
        self.assertEqual(len(rows), 7)
        sources = ['docs/initial_protocols/INITIAL_STEP1_v1.md', 'downstream/native_detection.py',
                   'downstream/figure3_lora_configs.py', 'downstream/controlled_comparison.py',
                   'downstream/transfer_comparison.py', 'downstream/frontier.py',
                   'downstream/representation_nested.py']
        for n, (row, expected) in enumerate(zip(rows, sources, strict=True), 1):
            label, subject, instructions, implementation = row.strip('|').split('|')
            self.assertEqual(label.strip(), f'Figure {n}')
            self.assertTrue(subject.strip())
            guide, = LINK.findall(instructions)
            source, = LINK.findall(implementation)
            path, fragment = guide[1].split('#')
            self.assertEqual(path, 'docs/PAPER_REPRODUCTION_GUIDE.md')
            self.assertTrue(fragment.startswith(f'figure-{n}-'))
            self.assertIn(fragment, anchors((ROOT / path).read_text()))
            self.assertEqual(source[1], expected)
            self.assertTrue((ROOT / source[1]).exists(), source)

    def test_local_links_and_full_operating_guide(self):
        links = dict(LINK.findall(self.text))
        self.assertIn('OPERATING_GUIDE.md#methods', links.values())
        for _, target in LINK.findall(self.text):
            if '://' in target:
                continue
            path, _, fragment = target.partition('#')
            file = ROOT / (path or 'README.md')
            self.assertTrue(file.exists(), target)
            if fragment:
                self.assertIn(fragment, anchors(file.read_text()), target)

    def test_first_example_writes_documented_three_seed_configs(self):
        block = re.search(r'^```sh\n(.*?)\n```', self.text, re.M | re.S)
        self.assertIsNotNone(block)
        args = shlex.split(block[1])
        self.assertEqual(args[:3], ['python3', '-m', 'downstream.figure3_lora_configs'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'configs').symlink_to(ROOT / 'configs', target_is_directory=True)
            subprocess.run([sys.executable, *args[1:]], check=True, cwd=temp,
                           env={**os.environ, 'PYTHONPATH': str(ROOT)})
            files = list((root / 'figure3-configs').glob('*.json'))
            configs = [json.loads(f.read_text()) for f in files]
            self.assertEqual(len(configs), 45)
            self.assertEqual(len({(c['task'], c['backbone']['kind'], c['seed'])
                                  for c in configs}), 45)
            self.assertEqual({c['seed'] for c in configs}, {0, 1, 2})
