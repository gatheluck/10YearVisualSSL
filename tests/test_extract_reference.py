"""Behavioral tests for extraction through an unchanged reference evaluator."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'bin' / 'extract-reference.py'

try:
    import numpy as np
except ImportError:
    np = None


class TestReferenceExtraction(unittest.TestCase):
    def run_reference(self, *, wrong=False, missing=False):
        self.assertTrue(TOOL.is_file(), 'reference extraction runner is not implemented')
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        calls = []
        def extract(model, loader, device):
            calls.append((model, loader.dataset.root, device))
            return [[3., 4.]], [0]
        reference = SimpleNamespace(extract_features=extract)
        def main():
            skipped, _ = reference.extract_features('weights', SimpleNamespace(dataset=SimpleNamespace(root='/data/train')), 'cuda')
            tuple(skipped.shape)  # Reference evaluators log skipped train shape.
            if not missing:
                reference.extract_features('weights', SimpleNamespace(dataset=SimpleNamespace(root='/decoy/val' if wrong else '/data/val')), 'cuda')
            raise AssertionError('probe training must never run')
        reference.main = main
        saved = []
        try:
            mod.extract_reference(reference, '/data', lambda *args: saved.append(args))
        finally:
            self.assertIs(reference.extract_features, extract)
        return calls, saved

    def test_only_validation_uses_original_extractor_and_stops_before_training(self):
        calls, saved = self.run_reference()
        self.assertEqual(calls, [('weights', '/data/val', 'cuda')])
        self.assertEqual(saved[0][:2], ([[3., 4.]], [0]))
        self.assertEqual(saved[0][2].dataset.root, '/data/val')

    def test_same_basename_different_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unexpected dataset'):
            self.run_reference(wrong=True)

    def test_missing_validation_cannot_report_success(self):
        with self.assertRaisesRegex(AssertionError, 'probe training'):
            self.run_reference(missing=True)


@unittest.skipIf(np is None, 'numpy is required for numerical delivery tests')
class TestReferenceDelivery(unittest.TestCase):
    def test_validation_rejects_wrong_order_nonfinite_zero_and_partial(self):
        self.assertTrue(TOOL.is_file())
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, 'validate_features'), 'validated delivery is not implemented')
        import numpy as np
        good = np.array([[3., 4.], [1., 2.]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        mod.validate_features(good, labels, [0, 1], 2)
        for features, targets, expected, count in [
            (good, labels[::-1], [0, 1], 2),
            (good, labels, [0, 1], 3),
            (good[:1], labels, [0, 1], 2),
            (good * np.nan, labels, [0, 1], 2),
            (good * 0, labels, [0, 1], 2),
            (good[:, 0], labels, [0, 1], 2),
        ]:
            with self.subTest(features=features, count=count):
                with self.assertRaises(ValueError):
                    mod.validate_features(features, targets, expected, count)


@unittest.skipIf(np is None, 'numpy is required for numerical delivery tests')
class TestProfileRun(unittest.TestCase):
    def test_profile_runs_real_reference_and_refuses_overwrite_or_changed_source(self):
        import hashlib
        import json
        import tempfile
        import numpy as np
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, 'run_profile'), 'profile execution is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'reference.py'
            (root / 'fixture_local_dependency.py').write_text('VALUE = 3.\n')
            source.write_text('''from fixture_local_dependency import VALUE
from types import SimpleNamespace
from pathlib import Path
import numpy as np
def extract_features(model, loader, device):
    return np.array([[3., 4.], [4., 3.]]), np.array([0, 1])
def main():
    extract_features(None, SimpleNamespace(dataset=SimpleNamespace(root=DATA_ROOT+'/val', targets=[0,1], samples=[(Path('a'),0),(Path('b'),1)], classes=['a','b'], transform='fixture')), 'cpu')
    raise AssertionError('training started')
''')
            profile = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'data_root': str(root/'data'), 'count': 2, 'id': 'fixture', 'argv': [], 'checkpoint': str(source)}
            # The fixture provides its root without relying on private configuration.
            source.write_text('DATA_ROOT = '+repr(profile['data_root'])+'\n'+source.read_text())
            profile['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            out = root / 'output'
            mod.run_profile(profile, out)
            np.testing.assert_allclose(np.load(out/'features.npy'), [[.6,.8],[.8,.6]], rtol=1e-6)
            np.testing.assert_array_equal(np.load(out/'labels.npy'), [0,1])
            self.assertEqual(json.loads((out/'result.json').read_text())['status'], 'ok')
            meta = json.loads((out/'meta.json').read_text())
            self.assertEqual(meta['arch'], 'fixture')
            self.assertEqual(meta['preprocessing'], 'fixture')
            self.assertEqual(len(meta['sample_order_sha256']), 64)
            with self.assertRaises(FileExistsError):
                mod.run_profile(profile, out)
            from unittest.mock import patch
            profile['distributed'] = True
            with patch.dict('os.environ', {'RANK': '1'}):
                mod.run_profile(profile, root/'rank1')
            self.assertFalse((root/'rank1').exists())
            profile.pop('distributed')
            source.write_text(source.read_text()+'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                mod.run_profile(profile, root/'other')


class TestSharedProbe(unittest.TestCase):
    def test_shared_probe_preserves_precision_and_never_trains(self):
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, 'extract_probe_reference'), 'shared probe extraction is not implemented')
        calls = []
        def train(**kwargs):
            raise AssertionError('probe trained')
        def extract(backbone, loader, device, *, amp_dtype):
            calls.append((device, amp_dtype, loader.dataset.root))
            return [[3,4]], [0]
        probe = SimpleNamespace(run_linear_probe=train, extract_split=extract, _amp_dtype=lambda x: 'resolved:'+x)
        paths = SimpleNamespace(train_dir='/old/train', val_dir='/old/val', num_classes=1000)
        reference = SimpleNamespace(linear_probe=probe, resolve_paths=lambda name: paths)
        def main():
            resolved = reference.resolve_paths('imagenet1k')
            self.assertEqual(resolved.val_dir, '/data/val')
            probe.run_linear_probe(backbone=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device='cuda')])), val_loader=SimpleNamespace(dataset=SimpleNamespace(root=resolved.val_dir)), cfg=SimpleNamespace(amp_dtype='bfloat16'))
            raise AssertionError('execution continued')
        reference.main = main
        saved=[]
        mod.extract_probe_reference(reference, '/data', lambda *args:saved.append(args))
        self.assertEqual(calls, [('cuda', 'resolved:bfloat16', '/data/val')])
        self.assertEqual(saved[0][:2], ([[3,4]], [0]))
        self.assertIs(probe.run_linear_probe, train)
        self.assertEqual(paths.val_dir, '/old/val')
        paths.num_classes = 100
        with self.assertRaisesRegex(ValueError, 'ImageNet-1k'):
            mod.extract_probe_reference(reference, '/data', lambda *args: None)
        paths.num_classes = 1000
        reference.main = lambda: probe.run_linear_probe(backbone=None, val_loader=SimpleNamespace(dataset=SimpleNamespace(root='/decoy/val')), cfg=None)
        with self.assertRaisesRegex(ValueError, 'unexpected dataset'):
            mod.extract_probe_reference(reference, '/data', lambda *args: None)
        reference.main = lambda: None
        with self.assertRaisesRegex(RuntimeError, 'without validation'):
            mod.extract_probe_reference(reference, '/data', lambda *args: None)


@unittest.skipIf(np is None, 'numpy is required for gathered features')
class TestGatherOrder(unittest.TestCase):
    def test_rank_concatenation_is_restored_by_sample_ids_not_labels(self):
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, 'restore_sample_order'), 'distributed sample ordering is not implemented')
        features = np.array([[30.], [10.], [40.], [20.]])
        labels = np.array([1, 0, 1, 0])
        x,y = mod.restore_sample_order(features, labels, [2,0,3,1], 4)
        np.testing.assert_array_equal(x[:,0], [10,20,30,40])
        np.testing.assert_array_equal(y, [0,0,1,1])
        for ids in [[0,0,2,3], [0,1,2,4], [0,1,2], [0,1,2,3,4]]:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    mod.restore_sample_order(features, labels, ids, 4)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch is required for distributed wiring')
class TestDistributedProbeWiring(unittest.TestCase):
    def test_gathered_sampler_indices_control_feature_order(self):
        import torch
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sent = []
        def gather(indices):
            sent.append(indices.tolist())
            return torch.tensor([2,0,3,1])
        class Dataset:
            root = '/data/val'
            def __len__(self):
                return 4
        loader = SimpleNamespace(dataset=Dataset(), sampler=[2,0])
        probe = SimpleNamespace(
            run_linear_probe=lambda **kwargs: self.fail('training executed'),
            extract_split=lambda *args, **kwargs: (torch.tensor([[30.],[10.],[40.],[20.]]), torch.tensor([1,0,1,0])),
            _amp_dtype=lambda name: None,
            dist_utils=SimpleNamespace(get_world_size=lambda:2, all_gather_tensor=gather))
        reference = SimpleNamespace(linear_probe=probe, resolve_paths=lambda name: None)
        reference.main = lambda: probe.run_linear_probe(
            backbone=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device='cpu')])),
            val_loader=loader, cfg=SimpleNamespace(amp_dtype='float32'))
        saved=[]
        mod.extract_probe_reference(reference, '/data', lambda *args:saved.append(args))
        self.assertEqual(sent, [[2,0]])
        self.assertEqual(saved[0][0][:,0].tolist(), [10,20,30,40])
        self.assertEqual(saved[0][1].tolist(), [0,0,1,1])


class TestImportedProbeBinding(unittest.TestCase):
    def test_direct_import_is_intercepted_and_restored(self):
        import sys
        from types import ModuleType
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        probe = ModuleType('fixture_shared_probe')
        exec('def run_linear_probe(**kwargs):\n    raise AssertionError("training executed")\n',probe.__dict__)
        probe._amp_dtype=lambda x:None
        probe.extract_split=lambda *args,**kwargs:([[1,2]],[0])
        original=probe.run_linear_probe
        reference=SimpleNamespace(run_linear_probe=original,resolve_paths=lambda name:None)
        reference.main=lambda:reference.run_linear_probe(backbone=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device='cpu')])),val_loader=SimpleNamespace(dataset=SimpleNamespace(root='/data/val')),cfg=SimpleNamespace(amp_dtype='float32'))
        from unittest.mock import patch
        with patch.dict(sys.modules,{'fixture_shared_probe':probe}):
            saved=[];mod.extract_probe_reference(reference,'/data',lambda *args:saved.append(args))
        self.assertEqual(saved[0][:2],([[1,2]],[0]))
        self.assertIs(reference.run_linear_probe,original)


class TestPositionalProbeBinding(unittest.TestCase):
    def test_reference_signature_binds_positional_arguments_before_extraction(self):
        spec = importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        def train(backbone, train_loader, val_loader, num_classes, save_dir, cfg, method_tag, train_sampler=None):
            self.fail('probe training executed')
        loader = SimpleNamespace(dataset=SimpleNamespace(root='/data/val'))
        model = SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device='cpu')]))
        probe = SimpleNamespace(run_linear_probe=train, _amp_dtype=lambda x:x,
                                extract_split=lambda m,l,d,amp_dtype: ([amp_dtype], [7]))
        reference = SimpleNamespace(linear_probe=probe, resolve_paths=lambda name:None)
        reference.main = lambda:probe.run_linear_probe(model, None, loader, 1000, '/unused', SimpleNamespace(amp_dtype='float32'), 'fixture')
        saved=[]
        mod.extract_probe_reference(reference, '/data', lambda *args:saved.append(args))
        self.assertEqual(saved[0][:2], (['float32'], [7]))
        self.assertIs(probe.run_linear_probe, train)


class TestLazyBackboneDevice(unittest.TestCase):
    def test_probe_device_wins_over_lazy_cpu_encoder_parameters(self):
        spec=importlib.util.spec_from_file_location('reference_runner', TOOL)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        devices=[]
        def extract(model, loader, device, **kwargs):
            devices.append(device);return [[1]], [0]
        probe=SimpleNamespace(run_linear_probe=lambda **kwargs:None,
            torch=SimpleNamespace(device=lambda x:x, cuda=SimpleNamespace(is_available=lambda:True)),
            extract_split=extract, _amp_dtype=lambda x:None)
        reference=SimpleNamespace(linear_probe=probe, resolve_paths=lambda name:None)
        reference.main=lambda:probe.run_linear_probe(
            backbone=SimpleNamespace(parameters=lambda:iter([])),
            val_loader=SimpleNamespace(dataset=SimpleNamespace(root='/data/val')),
            cfg=SimpleNamespace(amp_dtype='float32'))
        mod.extract_probe_reference(reference,'/data',lambda *args:None)
        self.assertEqual(devices,['cuda'])
