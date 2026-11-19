"""Synthetic reported means exercise descriptive comparison, not paper scores."""
import copy
import importlib
import tempfile
import unittest
from pathlib import Path


class TransferComparisonTests(unittest.TestCase):
    def document(self):
        models = list('abcde')
        return {'schema_version': 1, 'models': models, 'settings': [
            {'id': 'recognition', 'group': 'recognition', 'metric': 'percent',
             'protocol_id': {'LP': 'lp-v1', 'AP': 'ap-v1'},
             'scores': {'LP': dict(zip(models, [10, 20, 30, 40, 50], strict=True)),
                        'AP': dict(zip(models, [20, 10, 30, 40, 50], strict=True))}},
            {'id': 'geometry', 'group': 'geometry', 'metric': 'error',
             'protocol_id': {'LP': 'raw-rmse', 'AP': 'raw-rmse'},
             'scores': {'LP': dict(zip(models, [1, 2, 3, 4, 5], strict=True)),
                        'AP': dict(zip(models, [2, 1, 3, 4, 5], strict=True))}}]}

    def api(self):
        return importlib.import_module('downstream.transfer_comparison')

    def test_pair_reversals_use_means_and_direction(self):
        report = self.api().compare(self.document())
        for result in report['settings']:
            self.assertEqual(result['pair_count'], 10)
            self.assertEqual(result['reversal_count'], 1)
            self.assertEqual(result['reversals'][0]['models'], ['a', 'b'])
            self.assertEqual(result['reversals'][0]['lp_margin'], -result['reversals'][0]['ap_margin'])
        self.assertEqual(report['settings'][1]['protocols']['LP']['leaders'], ['a'])

    def test_radial_transforms_and_per_protocol_normalization(self):
        document = self.document()
        document['settings'][0]['scores']['AP']['e'] = 100
        report = self.api().compare(document)
        self.assertEqual(report['settings'][0]['protocols']['LP']['transformed']['a'], .1)
        self.assertAlmostEqual(report['settings'][0]['protocols']['LP']['radial']['a'], .2)
        self.assertAlmostEqual(report['settings'][0]['protocols']['AP']['radial']['a'], .2)
        self.assertAlmostEqual(report['settings'][1]['protocols']['LP']['radial']['b'], 2 / 3)
        self.assertEqual(report['input'], document)
        self.assertFalse(report['statistical_significance_established'])

    def test_ties_are_explicit_not_reversals_or_arbitrary_winners(self):
        document = self.document()
        document['settings'][0]['scores']['LP']['d'] = 50
        result = self.api().compare(document)['settings'][0]
        self.assertEqual(result['protocols']['LP']['leaders'], ['d', 'e'])
        self.assertIn(['d', 'e'], result['tied_pairs'])
        self.assertEqual(result['reversal_count'], 1)

    def test_rank_counts_keep_groups_and_ties_separate(self):
        document = self.document()
        document['settings'][0]['scores']['LP']['d'] = 50
        report = self.api().compare(document)
        self.assertEqual(report['unique_rank_counts']['LP']['a']['unique_first'], 1)
        self.assertEqual(report['unique_rank_counts']['LP']['e']['unique_first'], 0)
        self.assertEqual(report['group_rank_counts']['geometry']['LP']['a']['unique_first'], 1)
        self.assertEqual(report['group_rank_counts']['recognition']['AP']['e']['unique_first'], 1)

    def test_invalid_scores_membership_and_identity_are_rejected(self):
        mutations = [
            lambda d: d.__setitem__('schema_version', True),
            lambda d: d.__setitem__('extra', 0),
            lambda d: d.__setitem__('settings', []),
            lambda d: d['settings'][0].__setitem__('extra', 0),
            lambda d: d['settings'][0].__setitem__('group', ''),
            lambda d: d['settings'][0]['scores'].pop('AP'),
            lambda d: d['settings'][0]['protocol_id'].pop('AP'),
            lambda d: d['models'].append('f'),
            lambda d: d['models'].__setitem__(1, 'a'),
            lambda d: d['settings'].append(copy.deepcopy(d['settings'][0])),
            lambda d: d['settings'][0]['scores']['LP'].pop('a'),
            lambda d: d['settings'][0]['scores']['AP'].__setitem__('f', 1),
            lambda d: d['settings'][0].__setitem__('metric', 'unknown'),
            lambda d: d['settings'][0]['protocol_id'].__setitem__('AP', ''),
        ]
        mutations += [lambda d, value=value: d['settings'][0]['scores']['LP'].__setitem__('a', value)
                      for value in (float('nan'), float('inf'), True, -1, 101)]
        for mutation in mutations:
            document = self.document()
            mutation(document)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.api().compare(document)

    def test_all_zero_percent_normalization_is_not_silently_invented(self):
        document = self.document()
        document['settings'][0]['scores']['LP'] = dict.fromkeys('abcde', 0)
        with self.assertRaisesRegex(ValueError, 'zero'):
            self.api().compare(document)

    def test_cli_writes_hash_without_input_path_and_refuses_overwrite(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / 'private.json', Path(td) / 'report.json'
            source.write_text(json.dumps(self.document()))
            self.api().main(['--input', str(source), '--out', str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(len(report['input_sha256']), 64)
            self.assertNotIn(td, output.read_text())
            with self.assertRaises(FileExistsError):
                self.api().main(['--input', str(source), '--out', str(output)])
