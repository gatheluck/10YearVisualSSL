"""Native profile identity must be checked at the actual provider entry point."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / 'methods' / '08_split_brain'

@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeProfile(unittest.TestCase):
    def test_rejects_unknown_option_before_loading_model(self):
        import provider_support
        spec = importlib.util.spec_from_file_location('profile_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            record = {'method': METHOD.name, 'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(),
                      'feature_options': {'config_overrides_extra': {}}}
            encoder.with_name('export.json').write_text(json.dumps(record))
            with patch.object(provider, 'importlib', SimpleNamespace(import_module=Mock(side_effect=AssertionError('invalid options reached imports')))), patch.object(provider_support, 'import_sibling', side_effect=AssertionError('invalid options reached sibling imports')), self.assertRaisesRegex(ValueError, 'unsupported'):
                provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)

    def test_profile_reaches_model_inference_and_preserves_defaults(self):
        import provider_support
        from torchvision import transforms as T
        spec = importlib.util.spec_from_file_location('positive_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        configs, calls = [], []
        class Tiny(torch.nn.Module):
            def get_encoder(self): return self
        def load(state, cfg):
            configs.append(cfg)
            return Tiny()
        def extract(*args, **kwargs):
            calls.append(kwargs)
            return torch.ones((1, 3)), torch.zeros(1, dtype=torch.long)
        dataset = SimpleNamespace(transform=T.Compose([T.Resize(224), T.CenterCrop(224)]))
        evaluator = SimpleNamespace(_build_loader=lambda *args: (dataset, None), extract_features=extract,
                                   _extract=extract, _is_mnist=lambda root: False,
                                   _imagefolder_transform=lambda size: dataset.transform,
                                   _IMAGENET_MEAN=(.485,.456,.406), _IMAGENET_STD=(.229,.224,.225))
        adapter = SimpleNamespace(load_encoder=load)
        def sibling(name): return adapter if name == 'adapter' else evaluator
        original = provider._load_config()
        options = {'lab_mode': 'numpy_float32'}
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            def run():
                with patch.object(provider, 'importlib', SimpleNamespace(import_module=Mock(side_effect=sibling))), patch.object(provider_support, 'import_sibling', side_effect=lambda directory, name: sibling(name)), patch('torchvision.datasets.ImageFolder', return_value=dataset), patch('torch.utils.data.DataLoader', return_value=None):
                    return provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)
            _, _, meta = run()
            self.assertEqual(configs[-1], original)
            self.assertEqual(meta['native_feature_options'], {})
            encoder.with_name('export.json').write_text(json.dumps({'method': METHOD.name,
                'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(), 'feature_options': options}))
            _, _, meta = run()
            self.assertEqual(configs[-1], provider_support.configure_native_config(original, options))
            self.assertEqual(meta['native_feature_options'], options)
            self.assertEqual(provider._load_config(), original)
            self.assertEqual(dataset.native_eval, True)


class TestArtifact(unittest.TestCase):
    def test_exact_checkpoint_and_protocol(self):
        r = json.loads((METHOD / "provenance.json").read_text())["step1_native_artifact"]
        self.assertEqual(r['sha256'], 'e3491f8273fa1db79cf612f456c8d26c90226052a6a256eda53d527cdba13e78')
        self.assertEqual(r['availability'], 'user_supplied')
        self.assertEqual(r['state_key'], 'state_dict')
        self.assertEqual(r['strip_prefix'], '')
        self.assertEqual(r['feature_options'], {'lab_mode': 'numpy_float32'})
        self.assertEqual(r['checkpoint_epoch_stored'], 99)
        self.assertNotIn("url", r)

@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeLab(unittest.TestCase):
    def test_float32_native_conversion_before_channel_normalization(self):
        import numpy as np
        import provider_support
        from PIL import Image
        data = provider_support.import_sibling(METHOD, 'data.split_brain_dataset')
        rgb = np.array([[[0,0,0],[255,255,255],[255,0,0],[12,123,231],[3,7,11]]], dtype=np.uint8)
        # A scalar double-precision CIE reference is independent of NumPy's
        # SIMD pow/cbrt/matmul dispatch. The float32 pipeline has rounding at
        # every stage; 2e-4 Lab units bounds that accumulated error, including
        # the final 500x subtraction. Precision is checked separately below,
        # so a float64 implementation cannot pass by being more accurate.
        def reference(pixel):
            linear = [((v / 255.0 + .055) / 1.055) ** 2.4
                      if v / 255.0 > .04045 else v / 255.0 / 12.92 for v in pixel]
            matrix = ((.4124564, .3575761, .1804375),
                      (.2126729, .7151522, .0721750),
                      (.0193339, .1191920, .9503041))
            xyz = [sum(c * v for c, v in zip(row, linear)) / white
                   for row, white in zip(matrix, (.95047, 1., 1.08883))]
            f = [v ** (1 / 3) if v > (6 / 29) ** 3
                 else v / (3 * (6 / 29) ** 2) + 4 / 29 for v in xyz]
            return (116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2]))
        # Include the dark branch, gamma boundary and all neutral intensities,
        # plus deterministic colour samples beyond the original five pixels.
        samples = np.concatenate((rgb.reshape(-1, 3),
            np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1),
            np.random.default_rng(17).integers(0, 256, (1024, 3), dtype=np.uint8)))
        rgb = samples[None, ...]
        expected = np.array([[reference(pixel) for pixel in samples]])
        proxy = SimpleNamespace(**vars(np))
        proxy.asarray = Mock(wraps=np.asarray)
        proxy.cbrt = Mock(wraps=np.cbrt)
        proxy.where = Mock(wraps=np.where)
        with patch.object(data, 'np', proxy):
            actual = data.rgb2lab(rgb, native_eval=True)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(proxy.asarray.call_args.kwargs['dtype'], np.float32)
        self.assertEqual(proxy.cbrt.call_count, 1)
        self.assertEqual(proxy.cbrt.call_args.args[0].dtype, np.float32)
        self.assertEqual(proxy.where.call_count, 2)
        for call in proxy.where.call_args_list:
            for operand in call.args[1:]:
                self.assertEqual(operand.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)
        # Normalization must be exact for this invocation's raw Lab values,
        # not for values rounded by a different CPU's instruction path.
        l, ab, raw_l, raw_ab = data._to_lab_tensors(Image.fromarray(rgb), native_eval=True)
        np.testing.assert_array_equal(raw_l, actual[..., 0])
        np.testing.assert_array_equal(raw_ab, actual[..., 1:])
        self.assertTrue(torch.equal(l, torch.from_numpy((actual[...,0]-50.)/50.).unsqueeze(0)))
        self.assertTrue(torch.equal(ab, torch.from_numpy(actual[...,1:]/128.).permute(2,0,1)))

    def test_normalization_forwards_native_mode_and_preserves_channels(self):
        import numpy as np
        import provider_support
        from PIL import Image
        data = provider_support.import_sibling(METHOD, 'data.split_brain_dataset')
        sentinel = np.array([[[25., -64., 32.], [75., 16., -8.]]], dtype=np.float32)
        for native in (False, True):
            with patch.object(data, 'rgb2lab', return_value=sentinel) as convert:
                l, ab, raw_l, raw_ab = data._to_lab_tensors(Image.new('RGB', (2, 1)), native_eval=native)
            self.assertEqual(convert.call_args.kwargs, {'native_eval': native})
            self.assertTrue(torch.equal(l, torch.tensor([[[-.5, .5]]])))
            self.assertTrue(torch.equal(ab, torch.tensor([[[-.5, .125]], [[.25, -.0625]]])))
            np.testing.assert_array_equal(raw_l, sentinel[..., 0])
            np.testing.assert_array_equal(raw_ab, sentinel[..., 1:])

    def test_unknown_lab_mode_is_refused_before_model_import(self):
        import provider_support
        spec = importlib.util.spec_from_file_location('bad_lab_provider', METHOD / 'feature_provider.py')
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        with tempfile.TemporaryDirectory() as tmp:
            encoder = Path(tmp) / 'encoder.pt'; torch.save({}, encoder)
            encoder.with_name('export.json').write_text(json.dumps({'method': METHOD.name,
                'encoder_sha256': hashlib.sha256(encoder.read_bytes()).hexdigest(),
                'feature_options': {'lab_mode': 'numpy_float32_extra'}}))
            with patch.object(provider, 'importlib', SimpleNamespace(import_module=Mock(side_effect=AssertionError('invalid Lab mode reached model imports')))), self.assertRaisesRegex(ValueError, 'Lab conversion'):
                provider.extract_val_features(encoder_path=str(encoder), data_root=tmp, split='val', device='cpu', batch_size=1, num_workers=0)


@unittest.skipUnless(HAVE_TORCH, 'torch unavailable')
class TestNativeLabPortability(unittest.TestCase):
    def test_baseline_cpu_dispatch(self):
        """Run in a fresh interpreter: NumPy reads dispatch flags at import."""
        import os
        import subprocess
        import sys
        import numpy as np
        core = np._core if hasattr(np, '_core') else np.core
        dispatch = core._multiarray_umath.__cpu_dispatch__
        env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=','.join(dispatch),
                   PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        script = """
import numpy as np
import unittest
core = np._core if hasattr(np, '_core') else np.core
m = core._multiarray_umath
assert not any(m.__cpu_features__[f] for f in m.__cpu_dispatch__), 'dispatch still enabled'
suite = unittest.defaultTestLoader.loadTestsFromName(
    'tests.test_method_08_split_brain_native_options.TestNativeLab')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
