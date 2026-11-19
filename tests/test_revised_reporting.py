"""Revised depth paper RMSE stays distinct from the aligned legacy contract."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from downstream import contract


class RevisedReporting(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.revised_reporting'), 'revised reporting required')
        from downstream import revised_reporting
        return revised_reporting

    def fixture(self, root, seed, aligned, unaligned, **changes):
        out = root / str(seed); out.mkdir()
        cfg = {'profile': 'b5x_lora_dpt_v2', 'task': 'nyuv2', 'track': 'dpt', 'seed': seed,
               'backbone': {'kind': 'fixture'}, 'data_root': 'same-data'}
        config = root / f'{seed}.json'; config.write_text(json.dumps(cfg))
        result = {'profile': cfg['profile'], 'task': cfg['task'], 'track': cfg['track'],
                  'status': 'component_complete', 'training_complete': True,
                  'epochs_completed': 30, 'recipe': {'epochs': 30}, 'checkpoint_sha256': 'same-hash',
                  'metrics': {'rmse': aligned, 'rmse_unaligned': unaligned, 'scale_alignment': 'per-image median'}}
        result.update(self.api().reporting_fields(result)); result.update(changes)
        (out / 'result.json').write_text(json.dumps(result))
        contract.write_metrics(out, {'rmse': aligned}, {'rmse': 'nyuv2_rmse'})
        contract.write_manifest(out, task='nyuv2', method_ref='revised-component', status='ok',
            config_sha256=contract.sha256_bytes(config.read_bytes()), started_at='start', finished_at='end',
            seed=seed, backbone=cfg['backbone'])
        return {'seed': seed, 'config': str(config), 'out': str(out), 'exit_status': 0}

    def test_metadata_selects_unaligned_without_mutating_aligned_primary(self):
        api = self.api()
        result = {'task': 'nyuv2', 'track': 'lora', 'metrics': {'rmse': .2, 'rmse_unaligned': .8, 'scale_alignment': 'per-image median'}}
        original = copy.deepcopy(result)
        fields = api.reporting_fields(result)
        self.assertEqual(fields['reporting']['paper_metric']['field'], 'rmse_unaligned')
        self.assertEqual(fields['reporting']['paper_metric']['value'], .8)
        self.assertEqual(fields['reporting']['primary_in_record']['value'], .2)
        self.assertEqual(result, original)
        self.assertEqual(api.reporting_fields({'task': 'imagenet'}), {})
        for metric in ({'rmse': .2}, {'rmse': .2, 'rmse_unaligned': float('nan')},
                       {'rmse': .2, 'rmse_unaligned': -1}):
            with self.assertRaises(ValueError): api.reporting_fields(dict(result, metrics=dict(metric, scale_alignment='per-image median')))

    def test_aggregate_uses_every_unaligned_seed_and_sample_std(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records = [self.fixture(root, i, .1 + i*.01, v) for i, v in enumerate([.3, .4, .5])]
            report = api.aggregate(records, expected_seeds=[0, 1, 2])
            self.assertTrue(report['complete'])
            self.assertEqual(report['paper_metric']['field'], 'rmse_unaligned')
            self.assertEqual(report['paper_metric']['values'], [.3, .4, .5])
            self.assertAlmostEqual(report['paper_metric']['mean'], .4)
            self.assertAlmostEqual(report['paper_metric']['std'], .1)
            self.assertEqual(report['primary_in_record']['values'], [.1, .11, .12000000000000001])
            self.assertFalse(report['record_value'])
            self.assertNotIn(td, json.dumps(report))
            with self.assertRaises(ValueError): api.aggregate([records[0]]*2, expected_seeds=[0, 1])

    def test_incomplete_invalid_and_changed_records_are_not_measured_zero(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self.fixture(root, 0, .1, 0.)
            b = self.fixture(root, 1, .1, .7, training_complete=False)
            c = self.fixture(root, 2, .1, .8)
            path = Path(c['out']) / 'result.json'; path.write_text(path.read_text() + ' ')
            report = api.aggregate([a, b, c], expected_seeds=[0, 1, 2, 3])
            self.assertEqual(report['completed_seeds'], [0])
            self.assertEqual(report['missing_seeds'], [3])
            self.assertEqual([r['seed'] for r in report['invalid_runs']], [1, 2])
            self.assertEqual(report['paper_metric']['mean'], 0.)
            self.assertIsNone(report['paper_metric']['std'])
            self.assertFalse(report['complete'])

    def test_inconsistent_metadata_contract_and_checkpoint_are_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = self.fixture(root, 0, .1, .5)
            wrong_metadata = self.fixture(root, 1, .2, .6, reporting={})
            wrong_primary = self.fixture(root, 2, .3, .7,
                metrics={'rmse': .9, 'rmse_unaligned': .7, 'scale_alignment': 'per-image median'},
                **api.reporting_fields({'task': 'nyuv2', 'track': 'dpt', 'metrics':
                    {'rmse': .9, 'rmse_unaligned': .7, 'scale_alignment': 'per-image median'}}))
            wrong_weights = self.fixture(root, 3, .4, .8, checkpoint_sha256='different-hash')
            report = api.aggregate([first, wrong_metadata, wrong_primary, wrong_weights], expected_seeds=[0, 1, 2, 3])
            self.assertEqual(report['completed_seeds'], [0])
            self.assertEqual(len(report['invalid_runs']), 3)
            cfg = Path(wrong_metadata['config'])
            data = json.loads(cfg.read_text()); data['track'] = 'lora'; cfg.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'recipes'):
                api.aggregate([first, wrong_metadata], expected_seeds=[0, 1])

    def test_invalid_contract_does_not_echo_private_metric_keys(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); record = self.fixture(root, 0, .1, .5)
            out = Path(record['out'])
            path = out / contract.METRICS
            data = json.loads(path.read_text()); data['metrics']['/private/identity'] = 1.
            path.write_text(json.dumps(data))
            report = api.aggregate([record], expected_seeds=[0])
            self.assertFalse(report['complete'])
            self.assertNotIn('/private/identity', json.dumps(report))
