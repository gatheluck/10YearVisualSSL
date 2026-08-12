"""Colorization dataset (Zhang et al., 2016), ported from the lab's own code.

An RGB image is converted to CIE Lab: the L channel (normalised to [0, 1]) is the
input, and the ab channels are quantised to 313 bins as the per-pixel target.
The RGB->Lab conversion is pure numpy (sRGB -> XYZ(D65) -> Lab), verified against
published CIE Lab reference values -- scikit-image / opencv are not dependencies,
despite the capture's requirements.txt naming them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import adapterlib

from .ab_quantization import get_ab_points, quantize_ab_fast

_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def rgb_to_lab(rgb_image) -> Tuple[np.ndarray, np.ndarray]:
    """RGB (PIL image or HxWx3 uint8 array) -> (L [H, W] in [0, 100],
    ab [H, W, 2])."""
    if not isinstance(rgb_image, np.ndarray):
        rgb_image = np.asarray(rgb_image)  # a PIL image -> H x W x 3 uint8
    rgb = rgb_image.astype(np.float64) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = (lin @ _RGB2XYZ.T) / _WHITE_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), (7.787 * xyz) + (16.0 / 116.0))
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return L.astype(np.float32), np.stack([a, b], axis=-1).astype(np.float32)


def _l_tensor(rgb_pil) -> "tuple[torch.Tensor, np.ndarray]":
    l_channel, ab_channels = rgb_to_lab(rgb_pil)
    l_tensor = torch.from_numpy(l_channel / 100.0).float().unsqueeze(0)
    return l_tensor, ab_channels


class ColorizationDataset(Dataset):
    """Returns ``(l_tensor [1, H, W], ab_bins [H, W])`` for pretraining."""

    def __init__(self, data_path: str, mode: str = "train",
                 image_size: int = 256, crop_size: int = 224):
        crop = (transforms.RandomCrop(crop_size) if mode == "train"
                else transforms.CenterCrop(crop_size))
        steps = [transforms.Resize(image_size), crop]
        if mode == "train":
            steps.append(transforms.RandomHorizontalFlip())
        self.base_transform = transforms.Compose(steps)
        self.base = datasets.ImageFolder(
            adapterlib.dataset_split_dir(data_path, "train"))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_img, _ = self.base[index]
        rgb_img = self.base_transform(rgb_img)
        l_tensor, ab_channels = _l_tensor(rgb_img)
        bins = torch.from_numpy(quantize_ab_fast(ab_channels)).long()
        return l_tensor, bins


class ColorizationProbeDataset(Dataset):
    """Returns ``(l_tensor [1, H, W], label)`` for the linear probe: the frozen
    encoder reads the L channel, deterministically cropped (no augmentation)."""

    def __init__(self, data_path: str, image_size: int = 256,
                 crop_size: int = 224):
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size), transforms.CenterCrop(crop_size)])
        self.base = datasets.ImageFolder(data_path)

    @property
    def classes(self):
        return self.base.classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rgb_img, label = self.base[index]
        rgb_img = self.base_transform(rgb_img)
        l_tensor, _ = _l_tensor(rgb_img)
        return l_tensor, label


def get_class_weights(data_path: str, num_bins: int = 313,
                      sample_size: int = 10000,
                      lambda_smooth: float = 0.5) -> torch.Tensor:
    """Class-rebalancing weights for rare colours (paper Eq.: w = ((1 - lambda)
    * p + lambda / Q)^-1, normalised). Sampled over the training images."""
    pts = get_ab_points()
    base = datasets.ImageFolder(adapterlib.dataset_split_dir(data_path, "train"))
    n = min(sample_size, len(base))
    indices = np.random.choice(len(base), n, replace=False)
    resize = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224)])
    bin_counts = np.zeros(num_bins, dtype=np.float64)
    for idx in indices:
        rgb_img, _ = base[int(idx)]
        _, ab = rgb_to_lab(resize(rgb_img))
        bins, counts = np.unique(quantize_ab_fast(ab), return_counts=True)
        bin_counts[bins] += counts
    p = bin_counts / max(bin_counts.sum(), 1.0)
    w = 1.0 / ((1.0 - lambda_smooth) * p + lambda_smooth / num_bins)
    w = w / (np.sum(w * p) if np.sum(w * p) > 0 else 1.0)  # normalise E[w] = 1
    return torch.from_numpy(w).float()
