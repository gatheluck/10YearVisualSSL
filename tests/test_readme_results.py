"""Check the public results against the adopted means and the rendered chart."""
import csv
from itertools import combinations
from pathlib import Path
import re
import subprocess
import sys
import unittest
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
CHART = 'docs/figures/figure3-classification.svg'
MEANS = 'docs/figures/figure3-classification.csv'
RENDERER = 'bin/render-readme-figure3.py'
SVG = '{http://www.w3.org/2000/svg}'

# Independently transcribed from Supplement Tables 22 and 23.
EXPECTED = {
    'ImageNet-1K': {
        'DINOv3-7B': ('86.54', '87.82'),
        'VGGT-Ω': ('10.68', '81.62'),
        'RAEv2': ('75.41', '86.26'),
        'V-JEPA 2.1-G': ('57.61', '81.39'),
        'SigLIP 2-G': ('84.61', '87.36'),
    },
    'SSv2': {
        'DINOv3-7B': ('14.90', '47.43'),
        'VGGT-Ω': ('4.70', '44.57'),
        'RAEv2': ('8.20', '40.69'),
        'V-JEPA 2.1-G': ('21.51', '69.21'),
        'SigLIP 2-G': ('22.96', '40.90'),
    },
}


class ReadmeResults(unittest.TestCase):
    def setUp(self):
        self.readme = (ROOT / 'README.md').read_text(encoding='utf-8')

    def test_rankings_include_every_model_through_rank_thirty(self):
        section = self.readme.partition('## Model rankings\n')[2].split('\n## ', 1)[0]
        self.assertTrue(section, 'The public README must show model rankings')
        path = 'docs/BASIC_FIVE_RANKS.md'
        self.assertIn(f']({path})', section)
        full = self.table_rows((ROOT / path).read_text(encoding='utf-8'))
        self.assertEqual(len(full), 109)
        expected = [(rank, name, *scores) for name, rank, *scores in full if int(rank) <= 30]
        self.assertEqual(self.table_rows(section), expected)
        self.assertIn(f'{len(expected)} of {len(full)} models', section)
        self.assertEqual(expected[-1][:2], ('30', 'CLIP'))

    def test_results_pair_lp_and_lora_for_each_model(self):
        section = self.readme.partition('## Results\n')[2].split('\n## ', 1)[0]
        self.assertTrue(section, 'The public README must show the five-task results')
        rows = self.table_rows(section)
        models = ['DINOv3-7B', 'VGGT-Ω', 'RAEv2', 'V-JEPA 2.1-G', 'SigLIP 2-G']
        self.assertEqual([row[:2] for row in rows],
                         [(model, method) for model in models for method in ('LP', 'LoRA')])
        self.assertIn('| Model | Method | ImageNet-1K ↑ | COCO ↑ | ADE20K ↑ | NYUv2 ↓ | SSv2 ↑ |',
                      section)
        scores = [re.sub(r'<[^>]+>', '', value) for row in rows for value in row[2:]]
        self.assertEqual(len(scores), 50)
        for value in scores:
            self.assertRegex(value, r'^\d+\.\d+†?±\d+\.\d+$')
        self.assertEqual([score for score in scores if '†' in score], ['87.90†±0.03', '56.12†±0.11'])
        self.assertIn('standard deviation over three runs', section)
        self.assertIn('50 epochs on ImageNet-1K and 24 on SSv2', section)

    @staticmethod
    def table_rows(section):
        return [tuple(cell.strip() for cell in line.strip('|').split('|'))
                for line in section.splitlines() if line.startswith('| ')][2:]

    def test_accessible_extended_results_precede_the_rankings(self):
        intro = self.readme.split('## Model rankings\n', 1)[0]
        image = re.search(r'<img src="([^"]+)" alt="([^"]+)">', intro)
        self.assertIsNotNone(image, 'The README introduction has no results chart')
        self.assertLess(intro.index('>Project Page</a>'), image.start())
        target, alt = image.groups()
        self.assertEqual(target, 'docs/figures/extended-transfer.png')
        self.assertIn(f']({target})', intro)
        self.assertTrue((ROOT / target).is_file())
        for label in ('40 evaluation settings', 'linear', 'attentive'):
            self.assertIn(label, alt.lower())

    def test_highlights_correspond_to_their_sources(self):
        highlights = self.readme.split('## Highlights\n', 1)[1].split('## Model rankings', 1)[0]
        ranks = self.table_rows((ROOT / 'docs/BASIC_FIVE_RANKS.md').read_text(encoding='utf-8'))
        rae = next(row for row in ranks if row[0] == 'RAEv2')
        self.assertEqual((rae[4], rae[6]), ('1', '100'))
        self.assertIn('RAEv2 ranks first on ADE20K segmentation and 100th on SSv2', highlights)
        flips = sum(
            (float(a_lp) > float(b_lp)) != (float(a_ap) > float(b_ap))
            for (a_lp, a_ap), (b_lp, b_ap) in combinations(EXPECTED['SSv2'].values(), 2)
        )
        self.assertIn(f'{flips} of 10 model pairs', highlights)
        for label in ('image-level', 'patch-level', 'held-out model families', 'Spearman',
                      '0.19 to 0.75 on ADE20K', '0.01 to 0.67 on NYUv2'):
            self.assertIn(label, highlights)

    def test_linked_data_has_all_twenty_reported_means(self):
        self.assertTrue((ROOT / MEANS).is_file(), 'The chart data is missing')
        with (ROOT / MEANS).open(encoding='utf-8', newline='') as source:
            rows = list(csv.DictReader(source))
        reported = {(r['dataset'], r['model']): (r['lp_top1_pct'], r['ap_top1_pct'])
                    for r in rows}
        self.assertEqual(len(rows), 10)
        self.assertEqual(reported, {(task, model): pair for task, models in EXPECTED.items()
                                    for model, pair in models.items()})

    def test_documented_command_regenerates_chart_with_correct_geometry(self):
        guide = (ROOT / 'docs/PAPER_REPRODUCTION_GUIDE.md').read_text(encoding='utf-8')
        command = re.search(r'`python3 (' + re.escape(RENDERER) + r') > ([^ `]+)`', guide)
        self.assertIsNotNone(command, 'The guide has no chart regeneration command')
        script, destination = command.groups()
        self.assertEqual(destination, CHART)
        plotted = subprocess.check_output([sys.executable, script], cwd=ROOT)
        self.assertEqual(plotted, (ROOT / destination).read_bytes())
        svg = ElementTree.fromstring(plotted)
        panels = {p.attrib['aria-label']: p for p in svg.findall(f'{SVG}g')
                  if 'aria-label' in p.attrib}
        self.assertEqual(set(panels), set(EXPECTED))
        for task, model_scores in EXPECTED.items():
            panel = panels[task]
            labels = {t.text: t for t in panel.findall(f'{SVG}text')}
            self.assertIn('Top-1 accuracy (%)', labels)
            zero, mid, full = (float(labels[t].attrib['x']) for t in ('0', '50', '100'))
            self.assertAlmostEqual(mid, (zero + full) / 2)
            models = {g.attrib['aria-label']: g for g in panel.findall(f'{SVG}g')}
            self.assertEqual(set(models), set(model_scores))
            for model, scores in model_scores.items():
                group = models[model]
                self.assertEqual([t.text for t in group.findall(f'{SVG}text')], [model, *scores])
                lp, ap = group.findall(f'{SVG}rect')
                self.assertEqual(lp.attrib['fill'], 'none')
                self.assertNotEqual(ap.attrib['fill'], 'none')
                for bar, value in zip((lp, ap), scores, strict=True):
                    self.assertEqual(float(bar.attrib['x']), zero)
                    self.assertAlmostEqual(float(bar.attrib['width']),
                                           (full - zero) * float(value) / 100, places=3)
