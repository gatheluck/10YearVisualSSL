"""Source precision policy and continuation identity, without claiming GPU runs."""
import importlib
import importlib.util
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import torch
    BACKENDS = cast(Any, torch.backends)


def fixture(name, cls):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls)()


@unittest.skipUnless(HAVE, 'torch required')
class RevisedPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.modern = hasattr(BACKENDS.cuda.matmul, 'fp32_precision')
        if self.modern:
            self.switches = (BACKENDS, BACKENDS.cuda.matmul, BACKENDS.cudnn,
                             BACKENDS.cudnn.conv, BACKENDS.cudnn.rnn)
            for switch in self.switches:
                self.addCleanup(setattr, switch, 'fp32_precision', switch.fp32_precision)
        else:
            self.addCleanup(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision())
            self.addCleanup(setattr, BACKENDS.cudnn, 'allow_tf32', BACKENDS.cudnn.allow_tf32)
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.warn = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, self.deterministic, warn_only=self.warn)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('downstream.revised_precision'),
                             'source precision policy is missing')
        return importlib.import_module('downstream.revised_precision')

    def test_cuda_policy_matches_source_and_restores_all_matmul_modes(self):
        api = self.api()
        if self.modern:
            for mode in ('none', 'ieee', 'tf32'):
                BACKENDS.cuda.matmul.fp32_precision = mode
                BACKENDS.cudnn.conv.fp32_precision = 'ieee'
                BACKENDS.cudnn.rnn.fp32_precision = 'tf32'
                before = [switch.fp32_precision for switch in self.switches]
                with api.source_precision(torch.device('cuda')):
                    self.assertEqual(BACKENDS.cuda.matmul.fp32_precision, 'tf32')
                    self.assertEqual(BACKENDS.cudnn.conv.fp32_precision, 'tf32')
                    self.assertEqual(BACKENDS.cudnn.rnn.fp32_precision, 'tf32')
                    self.assertEqual(api.runtime_precision()['fp32']['cuda_matmul'], 'tf32')
                self.assertEqual([switch.fp32_precision for switch in self.switches], before)
            return
        for mode in ('highest', 'high', 'medium'):
            for cudnn in (False, True):
                torch.set_float32_matmul_precision(mode)
                BACKENDS.cudnn.allow_tf32 = cudnn
                with api.source_precision(torch.device('cuda')):
                    self.assertTrue(BACKENDS.cuda.matmul.allow_tf32)
                    self.assertTrue(BACKENDS.cudnn.allow_tf32)
                    self.assertEqual(api.runtime_precision()['fp32']['float32_matmul'], 'high')
                self.assertEqual(torch.get_float32_matmul_precision(), mode)
                self.assertEqual(BACKENDS.cudnn.allow_tf32, cudnn)

    def test_exception_nesting_and_cpu_do_not_leak_settings(self):
        api = self.api()
        if self.modern:
            BACKENDS.cuda.matmul.fp32_precision = 'ieee'
            BACKENDS.cudnn.conv.fp32_precision = 'ieee'
        else:
            torch.set_float32_matmul_precision('medium')
            BACKENDS.cudnn.allow_tf32 = False
        before = api.runtime_precision()
        with api.source_precision(torch.device('cpu')):
            self.assertEqual(api.runtime_precision(), before)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'), api.source_precision(torch.device('cuda')):
            outer = api.runtime_precision()
            with api.source_precision(torch.device('cuda')):
                self.assertEqual(api.runtime_precision(), outer)
            self.assertEqual(api.runtime_precision(), outer)
            raise RuntimeError('sentinel')
        self.assertEqual(api.runtime_precision(), before)
        self.assertEqual(json.loads(json.dumps(before)), before)

    def test_cli_applies_policy_before_execution_and_restores_on_failure(self):
        from downstream import revised_tasks
        from downstream.extended_distributed import Session
        if self.modern:
            BACKENDS.cuda.matmul.fp32_precision = 'ieee'
            BACKENDS.cudnn.conv.fp32_precision = 'ieee'
        else:
            torch.set_float32_matmul_precision('highest')
            BACKENDS.cudnn.allow_tf32 = False
        @contextmanager
        def device_session(_):
            yield Session(torch.device('cuda'))
        def execute(*_):
            if self.modern:
                self.assertEqual(BACKENDS.cuda.matmul.fp32_precision, 'tf32')
                self.assertEqual(BACKENDS.cudnn.conv.fp32_precision, 'tf32')
            else:
                self.assertTrue(BACKENDS.cuda.matmul.allow_tf32,
                                'CUDA source matmul policy was not installed')
                self.assertTrue(BACKENDS.cudnn.allow_tf32,
                                'CUDA source convolution policy was not installed')
            raise RuntimeError('execution sentinel')
        f = fixture('test_revised_task_cli', 'RevisedCLI')
        config = dict(f.config(), device='cuda')
        with tempfile.TemporaryDirectory() as directory, patch.object(revised_tasks, 'session', device_session), \
                patch.object(revised_tasks, '_execute', execute):
            out = Path(directory) / 'run'
            with self.assertRaisesRegex(RuntimeError, 'execution sentinel'):
                revised_tasks.run(config, out)
            self.assertEqual(json.loads((out / 'run_manifest.json').read_text())['status'], 'failed')
        if self.modern:
            self.assertEqual(BACKENDS.cuda.matmul.fp32_precision, 'ieee')
            self.assertEqual(BACKENDS.cudnn.conv.fp32_precision, 'ieee')
        else:
            self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
            self.assertFalse(BACKENDS.cudnn.allow_tf32)

    def test_actual_cpu_result_records_effective_runtime_without_changing_it(self):
        from downstream import revised_tasks
        f = fixture('test_revised_task_cli', 'RevisedCLI')
        f.setUp()
        if self.modern:
            BACKENDS.cuda.matmul.fp32_precision = 'ieee'
        else:
            torch.set_float32_matmul_precision('medium')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(revised_tasks, 'build_frozen_backbone', return_value=f.backbone()), \
                patch.object(revised_tasks, 'build_datasets', return_value=f.data('imagenet')):
            out = Path(directory) / 'run'
            result = revised_tasks.run(f.config(), out)
            self.assertIn('runtime_precision', result)
            self.assertEqual(result['runtime_precision'], self.api().runtime_precision())
            self.assertEqual(json.loads((out / 'result.json').read_text())['runtime_precision'], result['runtime_precision'])
        if self.modern:
            self.assertEqual(BACKENDS.cuda.matmul.fp32_precision, 'ieee')
        else:
            self.assertEqual(torch.get_float32_matmul_precision(), 'medium')

    def test_resume_rejects_changed_runtime_before_restoring_state(self):
        f = fixture('test_revised_resume', 'RevisedResumeTests')
        f.setUp()
        self.addCleanup(f.doCleanups)
        original = f.fresh(batchnorm=False)
        f.advance(original, 1)
        torch.use_deterministic_algorithms(not self.deterministic, warn_only=self.warn)
        current = f.fresh(batchnorm=False)
        before = {key: value.clone() for key, value in current.model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'identity'):
            current.load(str(original.path))
        f.equal(before, current.model.state_dict())

    def test_checkpoint_refuses_runtime_change_during_execution(self):
        f = fixture('test_revised_resume', 'RevisedResumeTests')
        f.setUp()
        self.addCleanup(f.doCleanups)
        current = f.fresh(batchnorm=False)
        f.advance(current, 1)
        saved = current.path.read_bytes()
        torch.use_deterministic_algorithms(not self.deterministic, warn_only=self.warn)
        with self.assertRaisesRegex(ValueError, 'identity'):
            current.save(1, current.step, current.losses)
        self.assertEqual(current.path.read_bytes(), saved)

    def test_analysis_policy_disables_only_matmul_and_restores(self):
        api = self.api()
        self.assertTrue(callable(getattr(api, 'precision_policy', None)), 'shared precision context is missing')
        if self.modern:
            BACKENDS.cuda.matmul.fp32_precision = 'tf32'
            BACKENDS.cudnn.conv.fp32_precision = 'ieee'
            BACKENDS.cudnn.rnn.fp32_precision = 'tf32'
        else:
            torch.set_float32_matmul_precision('medium')
            BACKENDS.cudnn.allow_tf32 = True
        before = api.runtime_precision()
        with api.precision_policy(torch.device('cuda'), matmul_tf32=False):
            current = api.runtime_precision()
            if self.modern:
                self.assertEqual(current['fp32']['cuda_matmul'], 'ieee')
                self.assertEqual(current['fp32']['cudnn_conv'], 'ieee')
                self.assertEqual(current['fp32']['cudnn_rnn'], 'tf32')
            else:
                self.assertFalse(current['fp32']['cuda_matmul_tf32'])
                self.assertTrue(current['fp32']['cudnn_tf32'])
        self.assertEqual(api.runtime_precision(), before)
        for arguments in ({'matmul_tf32': 1}, {'matmul_tf32': False, 'cudnn_tf32': 'true'}):
            with self.assertRaises(ValueError), api.precision_policy(torch.device('cuda'), **arguments):
                self.fail('invalid precision option accepted')
        self.assertEqual(api.runtime_precision(), before)

    def test_legacy_policy_restores_medium_without_operator_api(self):
        api = self.api()
        state = {'mode': 'medium'}
        class Matmul:
            @property
            def allow_tf32(self):
                return state['mode'] != 'highest'
            @allow_tf32.setter
            def allow_tf32(self, value):
                state['mode'] = 'high' if value else 'highest'
        backend = SimpleNamespace(cuda=SimpleNamespace(matmul=Matmul()),
                                  cudnn=SimpleNamespace(allow_tf32=False))
        old = SimpleNamespace(backends=backend, get_float32_matmul_precision=lambda: state['mode'],
                              set_float32_matmul_precision=lambda value: state.update(mode=value))
        self.assertFalse(hasattr(old.backends.cuda.matmul, 'fp32_precision'))
        with patch.object(api, 'torch', old):
            with api.source_precision(torch.device('cuda')):
                self.assertEqual(state['mode'], 'high')
                self.assertTrue(backend.cudnn.allow_tf32)
            self.assertEqual(state['mode'], 'medium')
            self.assertFalse(backend.cudnn.allow_tf32)
            with self.assertRaisesRegex(RuntimeError, 'legacy sentinel'), api.source_precision(torch.device('cuda')):
                raise RuntimeError('legacy sentinel')
            self.assertEqual(state['mode'], 'medium')
            self.assertFalse(backend.cudnn.allow_tf32)

    def test_inherited_numeric_switches_are_observed_and_bound(self):
        api = self.api()
        f = fixture('test_revised_resume', 'RevisedResumeTests')
        f.setUp()
        self.addCleanup(f.doCleanups)
        switches = []
        for key, obj, attr in (
            ('cudnn_enabled', BACKENDS.cudnn, 'enabled'),
            ('cudnn_benchmark', BACKENDS.cudnn, 'benchmark'),
            ('cudnn_deterministic', BACKENDS.cudnn, 'deterministic'),
            ('cuda_fp16_reduced_precision_reduction', BACKENDS.cuda.matmul, 'allow_fp16_reduced_precision_reduction'),
            ('cuda_bf16_reduced_precision_reduction', BACKENDS.cuda.matmul, 'allow_bf16_reduced_precision_reduction'),
        ):
            switches.append((key, lambda obj=obj, attr=attr: getattr(obj, attr),
                             lambda value, obj=obj, attr=attr: setattr(obj, attr, value)))
        switches.extend((
            ('sdpa_flash', BACKENDS.cuda.flash_sdp_enabled, BACKENDS.cuda.enable_flash_sdp),
            ('sdpa_memory_efficient', BACKENDS.cuda.mem_efficient_sdp_enabled, BACKENDS.cuda.enable_mem_efficient_sdp),
            ('sdpa_math', BACKENDS.cuda.math_sdp_enabled, BACKENDS.cuda.enable_math_sdp),
            ('deterministic_warn_only', torch.is_deterministic_algorithms_warn_only_enabled,
             lambda value: torch.use_deterministic_algorithms(self.deterministic, warn_only=value)),
        ))
        for key, getter, setter in switches:
            with self.subTest(key=key):
                original = getter()
                baseline = f.fresh(batchnorm=False).identity
                try:
                    setter(not original)
                    self.assertEqual(api.runtime_precision()[key], not original)
                    self.assertNotEqual(f.fresh(batchnorm=False).identity, baseline)
                finally:
                    setter(original)
