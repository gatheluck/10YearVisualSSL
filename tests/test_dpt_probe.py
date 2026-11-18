"""Four-tap dense adaptation and its pinned upstream boundary."""
import copy
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

HAVE = importlib.util.find_spec('torch') is not None
ROOT = Path(__file__).resolve().parents[1]
if HAVE:
    import torch
    from torch.nn import functional as F

    from downstream.dpt_probe import DPTProbe


@unittest.skipUnless(HAVE, 'torch required')
class DenseProbe(unittest.TestCase):
    def test_missing_submodule_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'downstream'
            package.mkdir()
            source = package / 'dpt_probe.py'
            shutil.copyfile(ROOT / 'downstream/dpt_probe.py', source)
            self.assertFalse((root / 'third_party/geosubprobe/evals/models/probes.py').exists())
            spec = importlib.util.spec_from_file_location('isolated_dpt', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.assertRaisesRegex(FileNotFoundError, 'initialize the pinned'):
                module.DPTProbe([3]*4, out_channels=2, hidden=4)

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def test_dense_outputs_and_updates_match_pinned_probe_boundary(self):
        for positive in (False, True):
            model = DPTProbe([3, 4, 5, 6], out_channels=1 if positive else 3,
                             hidden=4, positive=positive)
            self.assertGreater(sum(p.numel() for p in model.parameters()), 0)
            ref = copy.deepcopy(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            other_optimizer = torch.optim.AdamW(ref.parameters(), lr=.001)
            for _ in range(2):
                feats = [torch.randn(2, c, 2, 3, requires_grad=True) for c in (3, 4, 5, 6)]
                other = [f.detach().clone().requires_grad_() for f in feats]
                captured = []
                handle = ref.probe.out_conv.register_forward_hook(lambda m, a, out, captured=captured: captured.append(out))
                normalized = [ln(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                              for ln, f in zip(ref.norms, other)]
                ref.probe(normalized)
                handle.remove()
                expected = F.interpolate(captured[0], size=(9, 11), mode='bilinear', align_corners=False)
                if positive:
                    expected = F.softplus(expected) + .001
                actual = model(feats, (9, 11))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if positive:
                    self.assertTrue(bool((actual > .001).all()))
                optimizer.zero_grad()
                other_optimizer.zero_grad()
                actual.square().mean().backward()
                expected.square().mean().backward()
                for a, b in zip(feats, other):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                for a, b in zip(model.parameters(), ref.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                optimizer.step()
                other_optimizer.step()
                for a, b in zip(model.parameters(), ref.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertFalse(model.probe.out_conv._forward_hooks)

    def test_geometry_channel_and_constructor_refusals(self):
        for dims, channels, hidden in (([3]*3, 2, 4), ([3, 0, 3, 3], 2, 4), ([3]*4, 0, 4), ([3]*4, 2, 0)):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                DPTProbe(dims, out_channels=channels, hidden=hidden)
        model = DPTProbe([3]*4, out_channels=2, hidden=4)
        valid = [torch.ones(1, 3, 2, 2) for _ in range(4)]
        for feats, size in ((valid[:3], (9, 9)), (valid[:3]+[torch.ones(1, 4, 2, 2)], (9, 9)),
                            (valid[:3]+[torch.ones(1, 3, 3, 2)], (9, 9)), (valid, (0, 9))):
            with self.subTest(size=size), self.assertRaises(ValueError):
                model(feats, size)
            self.assertFalse(model.probe.out_conv._forward_hooks)

    def test_forward_error_removes_temporary_hook(self):
        model = DPTProbe([3]*4, out_channels=2, hidden=4)
        def fail(*args):
            raise RuntimeError('deliberate failure')
        model.probe.forward = fail
        with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
            model([torch.ones(1, 3, 2, 2) for _ in range(4)], (9, 9))
        self.assertFalse(model.probe.out_conv._forward_hooks)

    def test_roundtrip_and_no_input_list_mutation(self):
        model = DPTProbe([3]*4, out_channels=2, hidden=4)
        restored = DPTProbe([3]*4, out_channels=2, hidden=4)
        restored.load_state_dict(model.state_dict(), strict=True)
        feats = [torch.randn(1, 3, 2, 2) for _ in range(4)]
        original = [x.clone() for x in feats]
        torch.testing.assert_close(model(feats, (9, 11)), restored(feats, (9, 11)), rtol=0, atol=0)
        for before, after in zip(original, feats):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
