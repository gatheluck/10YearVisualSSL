"""Explicit revised depth reader, LoRA readout and median-aligned evaluation.

This source profile is distinct from historical unaligned NYUv2 results. Its
metric implementation does not assign a paper result to an experimental run.
"""
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from downstream.revised_data import _NORMALIZE


class RevisedNYUv2(Dataset):
    """Upright official HDF5 frames with worker-local, lazily reopened handles."""
    IMAGE_SHAPE = (1449, 3, 640, 480)
    DEPTH_SHAPE = (1449, 640, 480)

    def __init__(self, root, split):
        import h5py

        from downstream.nyuv2 import load_official_splits
        if split not in ('train', 'val'):
            raise ValueError('depth split must be train or val')
        self.mat_path = Path(root) / 'labeled/nyu_depth_v2_labeled.mat'
        with h5py.File(self.mat_path, 'r') as handle:
            if (handle['images'].shape != self.IMAGE_SHAPE
                    or handle['depths'].shape != self.DEPTH_SHAPE):
                raise ValueError('unexpected official NYUv2 HDF5 layout')
        train, val = load_official_splits(self.mat_path.parent / 'splits.mat', 1449)
        self.indices = train if split == 'train' else val
        self.train = split == 'train'
        self._handle = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._pid = None

    def __getstate__(self):
        return dict(self.__dict__, _handle=None, _pid=None)

    def _file(self):
        import h5py
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle = h5py.File(self.mat_path, 'r')
            self._pid = os.getpid()
        return self._handle

    def __getitem__(self, index):
        handle, frame = self._file(), self.indices[index]
        rgb = np.array(handle['images'][frame]).transpose(2, 1, 0)
        depth = np.array(handle['depths'][frame]).T.astype(np.float32)
        if rgb.max() <= 1.:
            rgb = rgb * 255.
        image = Image.fromarray(rgb.astype('uint8')).resize((224, 224), Image.Resampling.BICUBIC)
        depth_image = Image.fromarray(depth).resize((224, 224), Image.Resampling.BILINEAR)
        if self.train and random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth_image = depth_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target = torch.from_numpy(np.array(depth_image, dtype=np.float32))
        valid = (target > .1) & (target < 10.) & torch.isfinite(target)
        return _NORMALIZE(TF.to_tensor(image)), target, valid


class DepthTask(nn.Module):
    """Positive 1x1 readout before target resize; retains adapter gradients."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(backbone.out_channels, 1, 1)
        nn.init.normal_(self.head.weight, std=.01)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, images):
        return self.backbone.forward_features(images).float()

    def forward(self, images, out_hw):
        positive = F.softplus(self.head(self.features(images))) + .001
        return F.interpolate(positive, size=out_hw, mode='bilinear', align_corners=False)


def _validate_shapes(prediction, target, valid):
    if prediction.ndim != 3 or prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError('depth shape must match [B, H, W] without broadcasting')
    if valid.dtype != torch.bool:
        raise ValueError('depth valid mask must be boolean')


def silog_loss(prediction, target, valid):
    """Source log-MSE minus half squared log-mean, without square root."""
    _validate_shapes(prediction, target, valid)
    predicted = prediction.float().clamp_min(1e-4)
    truth = target.float().clamp_min(1e-4)
    residual = (predicted.log() - truth.log())[valid]
    if residual.numel() == 0:
        return predicted.sum() * 0.
    return residual.square().mean() - .5 * residual.mean().square()


@torch.no_grad()
def evaluate_depth(model, loader, device):
    """Per-image lower-median alignment; aggregate errors over all valid pixels.

    Empty evaluation and nonfinite predictions fail closed rather than yielding
    a numerical score. The original source returns zeros for an empty population.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError('revised depth evaluation requires a single device')
    model.eval()
    totals = torch.zeros(12, dtype=torch.float64, device=device)
    input_images = 0
    for images, target, valid in loader:
        images, target, valid = images.to(device), target.to(device), valid.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = model(images, target.shape[-2:])
        if output.ndim != 4 or output.shape[1] != 1:
            raise ValueError('depth output shape must be [B, 1, H, W]')
        prediction = output.float().squeeze(1)
        _validate_shapes(prediction, target, valid)
        if not torch.isfinite(prediction).all():
            raise ValueError('nonfinite depth prediction')
        input_images += len(target)
        masks = valid & (target > .1) & (target < 10.) & torch.isfinite(target)
        for predicted, truth, mask in zip(prediction, target.float(), masks):
            if not mask.any():
                continue
            raw, truth = predicted[mask], truth[mask].clamp(.1, 10.)
            scale = truth.median() / raw.median().clamp_min(1e-6)
            aligned = (raw * scale).clamp(.1, 10.)
            unaligned = raw.clamp(.1, 10.)
            ratio = torch.maximum(aligned / truth, truth / aligned)
            values = ((aligned-truth).square().sum(), ((aligned-truth).abs()/truth).sum(),
                      (ratio < 1.25).sum(), (ratio < 1.25**2).sum(), (ratio < 1.25**3).sum(),
                      mask.sum(), (unaligned-truth).square().sum(),
                      ((unaligned-truth).abs()/truth).sum(), scale, scale.new_tensor(1),
                      (unaligned >= 10.-1e-4).sum(), (unaligned <= .1+1e-4).sum())
            totals += torch.stack([value.to(torch.float64) for value in values])
    if not torch.isfinite(totals).all() or totals[5] <= 0:
        raise ValueError('empty or nonfinite depth evaluation')
    squared, absolute, d1, d2, d3, count, unaligned_sq, unaligned_abs, scale_sum, samples, hi, lo = totals.tolist()
    return {'rmse': math.sqrt(squared/count), 'absrel': absolute/count,
            'delta1': 100*d1/count, 'delta2': 100*d2/count, 'delta3': 100*d3/count,
            'n_pixels': int(count), 'n_images': int(samples), 'n_input_images': input_images,
            'scale_alignment': 'per-image median', 'median_scale_mean': scale_sum/samples,
            'rmse_unaligned': math.sqrt(unaligned_sq/count), 'absrel_unaligned': unaligned_abs/count,
            'frac_clamped_hi': hi/count, 'frac_clamped_lo': lo/count}
