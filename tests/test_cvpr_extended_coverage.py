"""The current figure population must not inherit the historical catalog scope."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'docs/CVPR_EXTENDED_COVERAGE.json'


class ExtendedCoverage(unittest.TestCase):
    def load(self):
        self.assertTrue(LEDGER.is_file(), 'current Figure 5 coverage ledger missing')
        return json.loads(LEDGER.read_text())

    def test_population_and_protocol_axes_are_explicit(self):
        data = self.load()
        self.assertEqual(data['figure'], 5)
        self.assertEqual(data['protocols'], ['LP', 'AP'])
        self.assertFalse(data['scores_reproduced'])
        rows = data['settings']
        self.assertEqual([r['figure_index'] for r in rows], list(range(1, 41)))
        self.assertEqual(len({r['dataset'] for r in rows}), 40)
        self.assertEqual({g: sum(r['paper_group'] == g for r in rows)
                          for g in {r['paper_group'] for r in rows}},
                         {'recognition': 24, 'detection_segmentation': 9,
                          'geometry': 3, 'video_pose': 4})
        self.assertTrue(all(r['result_attribution'] == 'unverified' for r in rows))

    def test_entries_resolve_to_catalog_and_existing_components(self):
        data = self.load()
        catalog = json.loads((ROOT / 'docs/submission_protocols/EXTEND_LINEAR_v1.json').read_text())['datasets']
        for row in data['settings']:
            with self.subTest(dataset=row['dataset']):
                self.assertIn(row['dataset'], catalog)
                self.assertEqual(row['catalog_metric'], catalog[row['dataset']]['primary_metric'])
                if row['component_status'] == 'available':
                    self.assertTrue((ROOT / (row['module'].replace('.', '/') + '.py')).is_file())
                else:
                    self.assertEqual(row['component_status'], 'native_input_and_saved_map_evaluation')
                    self.assertTrue((ROOT / (row['module'].replace('.', '/') + '.py')).is_file())
                    self.assertTrue(row['limitation'])

    def test_conflicting_entries_do_not_become_supported_by_diagram_group(self):
        rows = {r['dataset']: r for r in self.load()['settings']}
        for name, classes, inputs in [('spacenet2_vegas', 2, 'RGB'), ('sun_rgbd', 37, 'RGB-D')]:
            self.assertEqual(rows[name]['figure_metric'], 'mIoU')
            self.assertEqual(rows[name]['figure_classes'], classes)
            self.assertEqual(rows[name]['figure_input'], inputs)
            self.assertEqual(rows[name]['component_status'], 'native_input_and_saved_map_evaluation')
            self.assertEqual(rows[name]['module'], 'downstream.figure5_native_semantics')
        for name in ('pascal_voc2012', 'pascal_context', 'mapillary_vistas', 'bdd100k', 'cityscapes'):
            self.assertEqual(rows[name]['module'], 'downstream.extended_segmentation')
        self.assertEqual(rows['livecell']['module'], 'downstream.extended_detection')
        self.assertEqual(rows['lvis_v1']['module'], 'downstream.extended_detection')
        self.assertEqual(rows['cityscapes']['native_input'], 'train_ids_only')
        self.assertEqual(rows['mpi_sintel']['module'], 'downstream.extended_flow')

    def test_human_table_matches_machine_population(self):
        path = ROOT / 'docs/CVPR_EXTENDED_COVERAGE.md'
        self.assertTrue(path.is_file(), 'reviewable current scope table missing')
        pairs = re.findall(r'^\| (\d+) \| `([^`]+)` \|', path.read_text(), re.MULTILINE)
        self.assertEqual([(int(i), name) for i, name in pairs],
                         [(row['figure_index'], row['dataset']) for row in self.load()['settings']])


if __name__ == '__main__':
    unittest.main()
