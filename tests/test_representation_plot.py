"""Plot geometry reflects recorded fits, without inventing manuscript results."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('matplotlib'), 'plot dependencies required')
class RepresentationPlot(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.representation_plot'), 'plot component required')
        from downstream import representation_plot
        return representation_plot

    def fixture(self):
        from downstream.representation_plane import rotate_directions
        tasks = {name: {'theta': 30 + i * 10, 'tmin': 20 + i * 10, 'tmax': 35 + i * 10,
                        'rho_x': .2, 'rho_xy': .5 + i * .1, 'n': 3 if name == 'coco' else 4}
                 for i, name in enumerate(('imagenet', 'ssv2', 'coco', 'ade20k', 'nyuv2'))}
        return {'schema': 'representation_analysis_v1', 'status': 'diagnostic', 'record_value': False,
                'historical_run_verified': False, 'fit_models': ['a', 'b', 'c', 'd'],
                'references': {'image': 'r1', 'patch': 'r2'},
                'coordinates': {'image_agreement': [.4, .1, .1, .8], 'patch_agreement': [.7, .3, .1, .5]},
                'tasks': tasks, 'display': rotate_directions(tasks, rotation_degrees=-30)}

    def test_scene_preserves_ranked_points_explicit_angles_lengths_and_population(self):
        import numpy as np
        report = self.fixture(); original = copy.deepcopy(report)
        scene = self.api().build_scene(report, ['f1', 'f1', 'f2', 'f2'])
        np.testing.assert_array_equal(scene['points'], [[.75, 1], [.375, .5], [.375, .25], [1, .75]])
        self.assertEqual(scene['tasks']['imagenet']['endpoint'], [.5, 0.])
        self.assertEqual(scene['tasks']['imagenet']['fold_range'], [-10., 5.])
        self.assertEqual(scene['tasks']['coco']['n'], 3)
        self.assertEqual(report, original)
        self.assertFalse(scene['record_value'])
        self.assertFalse(scene['historical_run_verified'])
        self.assertEqual(scene['display_rotation_deg'], -30)

    def test_invalid_or_inconsistent_reports_fail_before_writing(self):
        api = self.api()
        for change in ('reference', 'display', 'count', 'rho', 'nonfinite', 'family'):
            report = self.fixture(); families = ['f1'] * 4
            if change == 'reference': report['fit_models'][0] = 'r1'
            if change == 'display': report['display']['tasks']['imagenet']['theta'] += 1
            if change == 'count': report['tasks']['coco']['n'] = 5
            if change == 'rho': report['tasks']['imagenet']['rho_xy'] = -0.1
            if change == 'nonfinite': report['coordinates']['image_agreement'][0] = float('nan')
            if change == 'family': families[0] = '/private/identity'
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                out = Path(td) / 'output'
                with self.assertRaises(ValueError): api.render(report, families, out)
                self.assertFalse(out.exists())

    def test_real_cli_reads_report_and_aligned_family_json(self):
        self.api()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'report.json').write_text(json.dumps(self.fixture()))
            (root / 'families.json').write_text(json.dumps(['f1', 'f1', 'f2', 'f2']))
            result = subprocess.run([sys.executable, '-m', 'downstream.representation_plot',
                '--analysis', str(root / 'report.json'), '--families', str(root / 'families.json'),
                '--out', str(root / 'plot')], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            scene = json.loads((root / 'plot/plot.json').read_text())
            self.assertEqual(scene['tasks']['imagenet']['endpoint'], [.5, 0.])
            self.assertNotIn(td, (root / 'plot/plot.json').read_text())

    def test_renderer_creates_inspectable_svg_png_and_no_input_path_metadata(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'plot'
            report = self.fixture(); report['ignored_private_path'] = td
            scene = api.render(report, ['f1', 'f1', 'f2', 'f2'], out)
            self.assertTrue((out / 'representation.png').read_bytes().startswith(b'\x89PNG'))
            svg = (out / 'representation.svg').read_text()
            self.assertIn('diagnostic', svg)
            self.assertIn('Arrow-only rotation', svg)
            self.assertNotIn(td, svg)
            self.assertEqual(json.loads((out / 'plot.json').read_text()), scene)
            self.assertNotIn(td, (out / 'plot.json').read_text())
            with self.assertRaises(FileExistsError): api.render(report, ['f1'] * 4, out)
