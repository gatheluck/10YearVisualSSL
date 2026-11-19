"""Revised depth geometry, positive-head ordering and explicitly aligned metrics."""
import importlib.util
import json
import math
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(name) is not None
           for name in ('torch', 'torchvision', 'h5py', 'scipy', 'numpy', 'PIL'))
if HAVE:
    import h5py
    import numpy as np
    import torch
    from PIL import Image
    from scipy.io import savemat
    from torch import nn
    from torchvision.transforms import functional as TF


def sparse_depth_fixture(root):
    """Use the real MATLAB layout without allocating all 1,449 frames."""
    folder = Path(root) / 'labeled'
    folder.mkdir(parents=True)
    y, x = np.indices((480, 640))
    rgb = np.stack(((3*x+7*y) % 256, (11*x+y) % 256, (x+13*y) % 256), -1).astype('uint8')
    depth = (.2 + (x + 2*y) / 250).astype('float32')
    depth[:80, :100] = .1
    depth[100:180, :100] = 10.
    depth[200:280, :100] = np.nan
    with h5py.File(folder / 'nyu_depth_v2_labeled.mat', 'w') as f:
        images = f.create_dataset('images', (1449, 3, 640, 480), dtype='uint8',
                                  chunks=(1, 3, 640, 480))
        depths = f.create_dataset('depths', (1449, 640, 480), dtype='float32',
                                  chunks=(1, 640, 480))
        for index in (0, 795):
            images[index] = rgb.transpose(2, 1, 0)
            depths[index] = depth.T
    savemat(folder / 'splits.mat', {'trainNdxs': np.arange(1, 796),
                                    'testNdxs': np.arange(796, 1450)})
    return rgb, depth


@unittest.skipUnless(HAVE, 'depth reader dependencies required')
class RevisedDepth(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def reader(self, split):
        from downstream.revised_depth import RevisedNYUv2
        reader = RevisedNYUv2(self.root, split)
        self.addCleanup(reader.close)
        return reader

    def test_reader_upright_interpolation_mask_and_paired_python_flip(self):
        rgb, depth = sparse_depth_fixture(self.root)
        reference_rgb = TF.normalize(TF.to_tensor(Image.fromarray(rgb).resize(
            (224, 224), Image.Resampling.BICUBIC)), [.485, .456, .406], [.229, .224, .225])
        reference_depth = torch.from_numpy(np.array(Image.fromarray(depth).resize(
            (224, 224), Image.Resampling.BILINEAR)))
        valid = torch.isfinite(reference_depth) & (reference_depth > .1) & (reference_depth < 10)
        for split, flip in [('val', False), ('train', False), ('train', True)]:
            reader = self.reader(split)
            self.assertEqual(len(reader), 795 if split == 'train' else 654)
            with patch('random.random', return_value=.1 if flip else .9):
                image, actual, mask = reader[0]
            expected_image = reference_rgb.flip(-1) if flip else reference_rgb
            expected_depth = reference_depth.flip(-1) if flip else reference_depth
            expected_valid = valid.flip(-1) if flip else valid
            torch.testing.assert_close(image, expected_image, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected_depth, equal_nan=True, rtol=0, atol=0)
            self.assertTrue(torch.equal(mask, expected_valid))
            self.assertEqual(tuple(actual.shape), (224, 224))
            self.assertEqual(mask.dtype, torch.bool)
            self.assertTrue((actual == .1).any() and (actual == 10).any())

    def test_reader_unit_range_pixels_and_handle_pickle_reopen(self):
        sparse_depth_fixture(self.root)
        with h5py.File(self.root / 'labeled/nyu_depth_v2_labeled.mat', 'r+') as f:
            f['images'][0] = np.ones((3, 640, 480), dtype='uint8')
        data = self.reader('train')
        with patch('random.random', return_value=.9):
            first = data[0]
            restored = pickle.loads(pickle.dumps(data))
            self.addCleanup(restored.close)
            data.close()
            for reader in (data, restored):
                actual = reader[0]
                for left, right in zip(actual, first):
                    torch.testing.assert_close(left, right, equal_nan=True, rtol=0, atol=0)
        expected = (torch.ones(3)-torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(first[0][:, 50, 50], expected)

    def test_reader_rejects_bad_split_partition_and_transposed_layout(self):
        from downstream.revised_depth import RevisedNYUv2
        sparse_depth_fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'split'):
            RevisedNYUv2(self.root, 'eval')
        savemat(self.root / 'labeled/splits.mat', {'trainNdxs': [1, 1], 'testNdxs': [2]})
        with self.assertRaisesRegex(ValueError, 'partition'):
            RevisedNYUv2(self.root, 'train')
        with h5py.File(self.root / 'labeled/nyu_depth_v2_labeled.mat', 'w') as f:
            f.create_dataset('images', (1449, 3, 480, 640), dtype='uint8')
            f.create_dataset('depths', (1449, 480, 640), dtype='float32')
        with self.assertRaisesRegex(ValueError, 'layout'):
            RevisedNYUv2(self.root, 'train')

    def test_worker_handles_are_reopened_after_probe_access(self):
        from torch.utils.data import DataLoader, Subset
        sparse_depth_fixture(self.root)
        data = self.reader('val')
        expected = data[0]
        original = data._handle
        pid = os.getpid()
        with patch('os.getpid', return_value=pid+1):
            data[0]
        self.assertFalse(original.id.valid)
        for batch in DataLoader(Subset(data, [0, 0]), batch_size=1, num_workers=2,
                                multiprocessing_context='spawn'):
            for actual, reference in zip(batch, expected):
                torch.testing.assert_close(actual[0], reference, equal_nan=True, rtol=0, atol=0)

    def test_head_positive_mapping_precedes_resize_and_preserves_gradients(self):
        from downstream.revised_depth import DepthTask
        class Body(nn.Module):
            out_channels = 1
            def forward_features(self, images):
                return images
        model = DepthTask(Body())
        with torch.no_grad():
            model.head.weight.fill_(1)
            model.head.bias.zero_()
        image = torch.tensor([[[[-4., 4.]]]], requires_grad=True)
        model.train()
        self.assertFalse(model.backbone.training)
        actual = model(image, (1, 3))
        expected = nn.functional.interpolate(nn.functional.softplus(image)+.001,
                                            (1, 3), mode='bilinear', align_corners=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        wrong = nn.functional.softplus(nn.functional.interpolate(image, (1, 3),
                                        mode='bilinear', align_corners=False))+.001
        self.assertGreater((actual-wrong).abs().max().item(), 1.)
        actual.sum().backward()
        self.assertIsNotNone(image.grad)
        self.assertGreater(model.head.weight.grad.abs().item(), 0)

    def test_silog_floor_scale_term_empty_mask_and_shape_contract(self):
        from downstream.revised_depth import silog_loss
        p = torch.tensor([[[0., 1., 4.]]], requires_grad=True)
        t = torch.tensor([[[.001, 2., 1.]]])
        valid = torch.ones_like(t, dtype=torch.bool)
        diff = torch.tensor([math.log(.0001/.001), math.log(.5), math.log(4.)])
        expected = diff.square().mean()-.5*diff.mean().square()
        actual = silog_loss(p, t, valid)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(p.grad[0, 0, 0].item(), 0)
        self.assertGreater(p.grad.abs().sum().item(), 0)
        zero = silog_loss(p, t, torch.zeros_like(valid))
        self.assertTrue(zero.requires_grad)
        self.assertEqual(zero.item(), 0)
        with self.assertRaisesRegex(ValueError, 'shape'):
            silog_loss(p.unsqueeze(1), t, valid)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            silog_loss(p, t, valid.float())

    def test_metrics_per_image_lower_median_and_global_pixel_weight(self):
        from downstream.revised_depth import evaluate_depth
        class Predictor(nn.Module):
            def forward(self, images, size):
                return images
        pred = torch.tensor([[[[1., 3., 5., 7.]]], [[[1., 3., 5., 7.]]]])
        target = torch.tensor([[[2., 3., 6., 7.]], [[2., 6., 0., 0.]]])
        mask = target > 0
        result = evaluate_depth(Predictor(), [(pred, target, mask)], torch.device('cpu'))
        # Even-sized first sample uses lower middle medians (3/3), not averaged medians (4.5/4).
        # Second sample contributes only two pixels and scale=2.
        self.assertEqual(result['scale_alignment'], 'per-image median')
        self.assertAlmostEqual(result['rmse'], math.sqrt(2/6), places=7)
        self.assertAlmostEqual(result['rmse_unaligned'], math.sqrt(12/6), places=7)
        self.assertEqual(result['median_scale_mean'], 1.5)
        self.assertEqual(result['n_pixels'], 6)
        self.assertEqual(result['n_images'], 2)
        self.assertEqual(result['n_input_images'], 2)
        split = evaluate_depth(Predictor(), [(pred[:1], target[:1], mask[:1]),
                               (pred[1:], target[1:], mask[1:])], torch.device('cpu'))
        self.assertEqual(result, split)
        json.dumps(result, allow_nan=False)

    def test_metric_clamps_thresholds_and_rejects_empty_nonfinite_or_multirank(self):
        from downstream.revised_depth import evaluate_depth
        class Predictor(nn.Module):
            def forward(self, images, size):
                return images
        pred = torch.tensor([[[[0., .8, 1., 20.]]]])
        target = torch.ones(1, 1, 4)
        mask = torch.ones_like(target, dtype=torch.bool)
        result = evaluate_depth(Predictor(), [(pred, target, mask)], torch.device('cpu'))
        self.assertEqual(result['frac_clamped_hi'], .25)
        self.assertEqual(result['frac_clamped_lo'], .25)
        self.assertEqual(result['delta1'], 25.)
        self.assertEqual(result['n_pixels'], 4)
        for batches in ([], [(pred, target, ~mask)], [(pred * float('nan'), target, mask)]):
            with self.assertRaisesRegex(ValueError, 'empty|nonfinite'):
                evaluate_depth(Predictor(), batches, torch.device('cpu'))
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=2), \
             self.assertRaisesRegex(ValueError, 'single'):
            evaluate_depth(Predictor(), [], torch.device('cpu'))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch required')
class RevisedDepthDependencies(unittest.TestCase):
    def test_loss_head_and_evaluator_do_not_require_hdf5_or_scipy(self):
        import subprocess
        import sys
        code = r"""
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('h5py', 'scipy'):
            raise ModuleNotFoundError('deliberately absent: ' + fullname)
sys.meta_path.insert(0, Block())
for dependency in ('h5py', 'scipy'):
    try:
        __import__(dependency)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('dependency blocker did not work')
import torch
from torch import nn
from downstream.revised_depth import DepthTask, RevisedNYUv2, evaluate_depth, silog_loss
class Backbone(nn.Module):
    out_channels = 3
    def forward_features(self, images):
        return images
model = DepthTask(Backbone())
images = torch.ones(1, 3, 2, 2)
target = torch.ones(1, 2, 2)
valid = target.bool()
loss = silog_loss(model(images, (2, 2)).squeeze(1), target, valid)
loss.backward()
assert model.head.weight.grad is not None
metrics = evaluate_depth(model, [(images, target, valid)], torch.device('cpu'))
assert metrics['n_pixels'] == 4 and metrics['delta1'] == 100.
try:
    RevisedNYUv2('unused-path', 'train')
except ModuleNotFoundError as exc:
    assert 'deliberately absent' in str(exc)
else:
    raise AssertionError('reader did not require its HDF5 dependency')
print('depth components worked without dataset extras')
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                text=True, capture_output=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('depth components worked without dataset extras', result.stdout)


if __name__ == '__main__':
    unittest.main()
