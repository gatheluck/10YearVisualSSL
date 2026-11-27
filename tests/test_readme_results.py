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

    def test_accessible_results_precede_figure_index(self):
        intro = self.readme.split('## Find a figure or table\n')[0]
        image = re.search(r'^!\[([^\]]+)\]\(([^)]+)\)$', intro, re.M)
        self.assertIsNotNone(image, 'The README introduction has no results chart')
        self.assertLess(intro.index('[Project Page]'), image.start())
        alt, target = image.groups()
        for label in ('ImageNet-1K', 'SSv2', 'linear', 'attentive', 'top-1'):
            self.assertIn(label.lower(), alt.lower())
        self.assertEqual(target, CHART)
        self.assertTrue((ROOT / target).is_file())
        self.assertIn(f']({MEANS})', intro)
        self.assertIn('Tables 22 and 23', intro)
        self.assertRegex(intro, r'means across three runs.*standard deviations omitted')

    def test_findings_correspond_to_the_three_figures(self):
        intro = self.readme.split('## Find a figure or table\n')[0]
        bullets = dict(re.findall(r'^- \*\*(Figure [237])\.\*\* (.+)$', intro, re.M))
        self.assertEqual(set(bullets), {'Figure 2', 'Figure 3', 'Figure 7'})
        for label in ('113', 'ASIS', 'family-specific', '2.83%', '84.47%'):
            self.assertIn(label, bullets['Figure 2'])
        for dataset, scores in EXPECTED.items():
            flips = sum(
                (float(a_lp) > float(b_lp)) != (float(a_ap) > float(b_ap))
                for (a_lp, a_ap), (b_lp, b_ap) in combinations(scores.values(), 2)
            )
            self.assertRegex(bullets['Figure 3'],
                             rf'\b{flips}/10 (?:model pairs )?on {re.escape(dataset)}\b')
        for label in ('image-level', 'patch-level', 'leave-one-family-out', 'Spearman',
                      '0.19 to 0.75 on ADE20K', '0.01 to 0.67 on NYUv2'):
            self.assertIn(label, bullets['Figure 7'])

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
