"""Real feature-array analysis, identity validation and path-free outputs."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('scipy'), 'analysis dependencies required')
class AnalysisCLI(unittest.TestCase):
    def fixture(self, root):
        import numpy as np
        rng = np.random.default_rng(88)
        models = []
        for i in range(10):
            image = root / f'image{i}.npz'
            patch = root / f'patch{i}.npz'
            np.savez(image, features=rng.normal(size=(12, 5)).astype('float32'), sample_ids=np.array([f'private_image_{j}' for j in range(12)]))
            np.savez(patch, features=rng.normal(size=(3, 4, 2, 2)).astype('float32'), sample_ids=np.array([f'private_image_{j}' for j in range(3)]))
            models.append({'id': f'm{i}', 'family': f'f{i % 4}', 'image': str(image), 'patch': str(patch),
                           'scores': {t: float(i * i + j * i + (i % 3)) for j, t in enumerate(('imagenet', 'ssv2', 'coco', 'ade20k', 'nyuv2'))}})
        models[4]['scores']['coco'] = None
        return {'schema': 'representation_analysis_v1', 'models': models, 'image_reference': 'm0',
                'patch_reference': 'm1', 'k': 2, 'image_search_k': 3, 'grid': 2, 'chunk_size': 4,
                'device': 'cpu', 'diagnostic': True}

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.representation_analysis'), 'analysis CLI API required')
        from downstream.representation_analysis import run
        return run

    def test_cli_analyzes_arrays_excludes_both_references_and_hides_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.fixture(root)
            self.api()
            path = root / 'config.json'; path.write_text(json.dumps(config))
            out = root / 'output'
            result = subprocess.run([sys.executable, '-m', 'downstream.representation_analysis', '--config', str(path), '--out', str(out)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / 'analysis.json').read_text())
            self.assertEqual(report['fit_models'], [f'm{i}' for i in range(2, 10)])
            self.assertEqual(report['tasks']['imagenet']['n'], 8)
            self.assertEqual(report['tasks']['coco']['n'], 7)
            self.assertFalse(report['record_value'])
            self.assertFalse(report['historical_run_verified'])
            self.assertEqual(report['display']['display_rotation_deg'], 0.)
            self.assertEqual(report['inputs']['m0']['image_rows'], 12)
            text = (out / 'analysis.json').read_text()
            self.assertNotIn(td, text)
            self.assertNotIn('private_image_', text)
            self.assertTrue((out / 'image_agreement.npy').is_file())
            self.assertTrue((out / 'patch_agreement.npy').is_file())
            from downstream.representation_plane import percentile_rank
            expected = percentile_rank([-model['scores']['nyuv2'] for model in config['models'][2:]])
            self.assertEqual(report['tasks']['nyuv2']['target'], expected.tolist())
            with self.assertRaises(FileExistsError): self.api()(config, out)

    def test_rejects_order_mismatch_duplicate_ids_and_unknown_controls_before_outputs(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.fixture(root); run = self.api()
            path = Path(config['models'][3]['image'])
            with np.load(path) as z:
                features, ids = z['features'].copy(), z['sample_ids'].copy()
            np.savez(path, features=features, sample_ids=ids[::-1])
            with self.assertRaisesRegex(ValueError, 'order|membership'): run(config, root / 'bad')
            self.assertFalse((root / 'bad').exists())
            np.savez(path, features=features, sample_ids=ids)
            config['models'][3]['id'] = 'm0'
            with self.assertRaisesRegex(ValueError, 'unique|duplicate'): run(config, root / 'duplicate')
            config['models'][3]['id'] = 'm3'; config['unknown'] = 1
            with self.assertRaisesRegex(ValueError, 'unknown'): run(config, root / 'unknown')

    def test_real_populations_are_required_without_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.fixture(root); config['diagnostic'] = False
            with self.assertRaisesRegex(ValueError, 'population|reference profile'): self.api()(config, root / 'full')
            self.assertFalse((root / 'full').exists())

    def test_missing_score_key_is_not_an_explicit_missing_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.fixture(root)
            del config['models'][4]['scores']['coco']
            with self.assertRaisesRegex(ValueError, 'scores|keys'):
                self.api()(config, root / 'missing')

    def test_input_arrays_do_not_accumulate_across_model_population(self):
        import weakref
        from unittest.mock import patch

        self.api()
        from downstream import representation_analysis as module
        original = module._feature_file
        live, peak = [], []

        def observe(*args):
            result = original(*args)
            live.append(weakref.ref(result[0]))
            peak.append(sum(ref() is not None for ref in live))
            return result

        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.fixture(root)
            with patch.object(module, '_feature_file', side_effect=observe):
                module.run(config, root / 'bounded')
        self.assertLessEqual(max(peak), 4, 'retain only one model pair, not every model feature array')

    def test_changed_feature_bytes_after_preflight_are_rejected(self):
        from unittest.mock import patch

        import numpy as np

        self.api()
        from downstream import representation_analysis as module
        original = module._feature_file
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.fixture(root)
            first = config['models'][0]['image']
            reads = 0

            def replace_before_second_read(path, dimensions):
                nonlocal reads
                if path == first:
                    reads += 1
                    if reads == 2:
                        with np.load(path) as archive:
                            features, ids = archive['features'].copy(), archive['sample_ids'].copy()
                        np.savez(path, features=features + .25, sample_ids=ids)
                return original(path, dimensions)

            with patch.object(module, '_feature_file', side_effect=replace_before_second_read), self.assertRaisesRegex(ValueError, 'changed'):
                module.run(config, root / 'changed')
            self.assertFalse((root / 'changed').exists())


if __name__ == '__main__':
    unittest.main()
