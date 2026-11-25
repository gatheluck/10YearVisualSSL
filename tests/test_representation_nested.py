"""Nested PCoA selection, held-out isolation and public CLI delivery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class NestedGeometry(unittest.TestCase):
    @staticmethod
    def fixture():
        import numpy as np
        rng = np.random.default_rng(928)
        inputs = {'families': np.array([f'f{i // 4}' for i in range(28)]),
                  'model_ids': np.array([f'm{i}' for i in range(28)])}
        for key in ('patch_pc', 'patch_raw'):
            features = rng.normal(size=(28, 6))
            delta = features[:, None] - features[None, :]
            inputs[key] = np.exp(-(delta**2).sum(2) / 10)
        for key in ('ade20k', 'nyuv2'):
            inputs[key] = rng.normal(size=28)
        return inputs

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.representation_nested'),
                             'nested reference-free analysis is required')
        from downstream.representation_nested import analyze
        return analyze

    def test_portable_cli_reproduces_independent_choices_and_correlations(self):
        import numpy as np
        self.api()
        # Obtained by evaluating the original nested analysis on this synthetic fixture.
        expectations = {
            'ade20k': (-.146, ['raw/rank/2', 'raw/rank/2', 'raw/rank/2', 'raw/rank/2',
                               'raw/neglog/4', 'raw/neglog/2', 'raw/rank/3']),
            'nyuv2': (-.200, ['pc/rank/4', 'pc/neglog/4', 'pc/rank/2', 'pc/rank/4',
                              'pc/neglog/2', 'raw/rank/4', 'pc/rank/2']),
        }
        with tempfile.TemporaryDirectory() as td:
            path, out = Path(td) / 'aligned.npz', Path(td) / 'report.json'
            np.savez(path, **self.fixture())
            cmd = [sys.executable, '-m', 'downstream.representation_nested',
                   '--input', str(path), '--out', str(out)]
            subprocess.run(cmd, check=True, capture_output=True)
            original = out.read_bytes()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), original)
            result = json.loads(original)
            self.assertFalse(result['historical_run_verified'])
            for task, (rho, choices) in expectations.items():
                rows = result['tasks'][task]['folds']
                self.assertEqual([r['family'] for r in rows], [f'f{i}' for i in range(7)])
                self.assertEqual([f"{r['matrix'][6:]}/{r['transform']}/{r['k']}" for r in rows], choices)
                self.assertEqual(round(result['tasks'][task]['spearman'], 3), rho)

    def test_held_out_targets_do_not_affect_own_fold_predictions_or_selection(self):
        inputs = self.fixture()
        analyze = self.api()
        a = analyze(inputs)['tasks']['ade20k']
        inputs['ade20k'][:4] += 100
        b = analyze(inputs)['tasks']['ade20k']
        self.assertEqual(a['folds'][0], b['folds'][0])
        self.assertEqual(a['predictions'][:4], b['predictions'][:4])

    def test_depth_error_has_opposite_goodness_to_iou(self):
        import numpy as np
        inputs = self.fixture()
        inputs['nyuv2'] = -inputs['ade20k']
        tasks = self.api()(inputs)['tasks']
        np.testing.assert_array_equal(tasks['ade20k']['predictions'], tasks['nyuv2']['predictions'])

    def test_tied_candidates_keep_first_matrix(self):
        inputs = self.fixture()
        inputs['patch_raw'] = inputs['patch_pc'].copy()
        for task in self.api()(inputs)['tasks'].values():
            self.assertTrue(all(row['matrix'] == 'patch_pc' for row in task['folds']))

    def test_invalid_input_rejected(self):
        import numpy as np
        analyze = self.api()
        mutations = (
            ('patch_raw', lambda a: a[:-1]), ('patch_pc', lambda a: a.astype(complex)),
            ('patch_pc', lambda a: np.triu(a)), ('patch_pc', lambda a: a * 2),
            ('ade20k', lambda a: np.full_like(a, np.nan)),
            ('families', lambda a: np.array(['one'] * len(a))),
            ('model_ids', lambda a: np.array(['repeated'] * len(a))),
        )
        for key, mutate in mutations:
            with self.subTest(key=key):
                inputs = self.fixture()
                inputs[key] = mutate(inputs[key])
                with self.assertRaises(ValueError):
                    analyze(inputs)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class PatchReferenceSelection(unittest.TestCase):
    @staticmethod
    def fixture():
        import numpy as np
        rng = np.random.default_rng(717)
        image = rng.random(21)
        patch = rng.random((26, 21))
        np.fill_diagonal(patch, 1.)
        scores = rng.normal(size=(2, 21))
        return {'model_ids': np.array([str(i) for i in range(21)]),
                'families': np.repeat(list('ABCDEFG'), 3),
                'candidate_ids': np.array([str(i) for i in range(26)]),
                'image_agreement': image, 'patch_agreement': patch,
                'ade20k': scores[0], 'nyuv2': scores[1]}

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.representation_reference'),
                             'fold-wise patch reference selection is required')
        from downstream.representation_reference import analyze
        return analyze

    def test_cli_reference_removal_and_reported_fold_selection(self):
        import numpy as np
        self.api()
        with tempfile.TemporaryDirectory() as tmp:
            p, out = Path(tmp) / 'input.npz', Path(tmp) / 'out.json'
            np.savez(p, **self.fixture())
            subprocess.run([sys.executable, '-m', 'downstream.representation_nested',
                            '--mode', 'patch-reference', '--input', str(p), '--out', str(out)],
                           capture_output=True, check=True)
            report = json.loads(out.read_text())
        for task, expected, count, rho in (
                ('ade20k', [7,18,6,18,7,18,21], 20, -.024),
                ('nyuv2', [14,3,10,10,3,4,1], 19, -.242)):
            record = report['tasks'][task]
            self.assertEqual([int(r['candidate_id']) for r in record['folds']], expected)
            self.assertEqual(sum(x is not None for x in record['predictions']), count)
            self.assertEqual(record['n_eval'], count)
            self.assertEqual(round(record['spearman'], 3), rho)

    def test_depth_prediction_uses_ranks_of_negative_rmse(self):
        inputs = self.fixture()
        inputs['nyuv2'] = -inputs['ade20k']
        tasks = self.api()(inputs)['tasks']
        self.assertEqual(tasks['ade20k']['predictions'], tasks['nyuv2']['predictions'])

    def test_invalid_alignment(self):
        analyze = self.api()
        for key in ('patch_agreement', 'candidate_ids', 'families'):
            with self.subTest(key=key):
                inputs = self.fixture()
                inputs[key] = inputs[key][:-1]
                with self.assertRaises(ValueError):
                    analyze(inputs)
