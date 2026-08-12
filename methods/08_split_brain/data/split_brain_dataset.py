"""Split-Brain dataset (Zhang et al., 2017), ported from the lab's own code.

An RGB image is converted to CIE Lab. The L channel and the ab channels are each
an input to one cross-channel branch and the (quantised) target of the other:
  - net1 input  = L channel, normalised to ~[-1, 1] as (L - 50) / 50
  - net2 input  = ab channels, normalised as ab / 128
  - net1 target = ab quantised to 313 bins (nearest of the in-gamut codebook)
  - net2 target = L quantised to 50 bins (floor(L / 2))

The RGB->Lab conversion (sRGB -> XYZ(D65) -> Lab, verified vs published CIE Lab
values) and the quantisation are **pure numpy** -- the capture's own comment
states the released ab target IS NumPy argmin (its cKDTree path only accelerates
it and is corrected to match at Voronoi boundaries). So the port keeps the
torch-only closure; scipy / scikit-image are not dependencies. The 313 ab-bin
codebook (`pts_in_hull.npy`) is vendored and its sha256 is verified on load.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import adapterlib

AB_TARGET_CLASSES = 313
L_TARGET_CLASSES = 50
_CODEBOOK_PATH = Path(__file__).resolve().parent / "pts_in_hull.npy"
_CODEBOOK_SHA256 = "b5dec01315c34f43f1c8c089e84c45ae35d1838d8e77ed0e7ca930f79ffa450e"

_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


@lru_cache(maxsize=1)
def load_ab_codebook() -> np.ndarray:
    """The vendored [313, 2] ab-bin centres, sha256-verified. Errors loudly if
    absent or altered -- the port is hermetic and never downloads."""
    if not _CODEBOOK_PATH.is_file():
        raise FileNotFoundError(
            f"{_CODEBOOK_PATH} is missing; the 313 ab-bin codebook must be "
            "vendored (source: richzhang/colorization, see provenance.json)")
    payload = _CODEBOOK_PATH.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _CODEBOOK_SHA256:
        raise RuntimeError(
            f"ab codebook checksum mismatch: expected {_CODEBOOK_SHA256}, "
            f"got {digest}")
    pts = np.load(_CODEBOOK_PATH)
    if pts.shape != (AB_TARGET_CLASSES, 2):
        raise RuntimeError(
            f"ab codebook must be [{AB_TARGET_CLASSES}, 2], got {pts.shape}")
    return pts.astype(np.float64)


def rgb2lab(img_rgb) -> np.ndarray:
    """RGB (H x W x 3 uint8, or PIL image) -> CIE Lab (H x W x 3 float32)."""
    rgb = np.asarray(img_rgb, dtype=np.float64) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = (lin @ _RGB2XYZ.T) / _WHITE_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), (7.787 * xyz) + (16.0 / 116.0))
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def quantize_l(l_channel: np.ndarray) -> np.ndarray:
    """L (in [0, 100]) -> 50 uniform bins."""
    values = np.asarray(l_channel, dtype=np.float32)
    return np.clip(np.floor(values / 2.0), 0, L_TARGET_CLASSES - 1).astype(
        np.int64)


def quantize_ab(ab_channels: np.ndarray) -> np.ndarray:
    """ab (H x W x 2) -> nearest of the 313 in-gamut bins (NumPy argmin, the
    released SplitBrain target)."""
    values = np.asarray(ab_channels, dtype=np.float32)
    codebook = load_ab_codebook()
    H, W = values.shape[:2]
    flat = values.reshape(-1, 2)
    d2 = np.sum((flat[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).reshape(H, W).astype(np.int64)


def _to_lab_tensors(rgb_pil):
    lab = rgb2lab(np.asarray(rgb_pil))
    l_channel, ab_channels = lab[..., 0], lab[..., 1:]
    l_input = torch.from_numpy((l_channel - 50.0) / 50.0).float().unsqueeze(0)
    ab_input = torch.from_numpy(ab_channels / 128.0).permute(2, 0, 1).float()
    return l_input, ab_input, l_channel, ab_channels


class SplitBrainDataset(Dataset):
    """Returns ``(l_input, ab_input, l_target, ab_target, label)`` for step 1."""

    def __init__(self, data_root: str, crop_size: int = 224, train: bool = True):
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip()])
        else:
            resize = int(round(crop_size * 256 / 224))
            self.transform = transforms.Compose([
                transforms.Resize(resize), transforms.CenterCrop(crop_size)])
        self.base = datasets.ImageFolder(
            adapterlib.dataset_split_dir(data_root, "train"))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple:
        rgb_img, label = self.base[index]
        rgb_img = self.transform(rgb_img.convert("RGB"))
        l_input, ab_input, l_channel, ab_channels = _to_lab_tensors(rgb_img)
        l_target = torch.from_numpy(quantize_l(l_channel)).long()
        ab_target = torch.from_numpy(quantize_ab(ab_channels)).long()
        return l_input, ab_input, l_target, ab_target, label


class SplitBrainProbeDataset(Dataset):
    """Returns ``(l_input, ab_input, label)`` for the linear probe: the frozen
    encoders read the deterministically-cropped L and ab channels."""

    def __init__(self, data_root: str, crop_size: int = 224):
        resize = int(round(crop_size * 256 / 224))
        self.transform = transforms.Compose([
            transforms.Resize(resize), transforms.CenterCrop(crop_size)])
        self.base = datasets.ImageFolder(data_root)

    @property
    def classes(self):
        return self.base.classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple:
        rgb_img, label = self.base[index]
        rgb_img = self.transform(rgb_img.convert("RGB"))
        l_input, ab_input, _, _ = _to_lab_tensors(rgb_img)
        return l_input, ab_input, label
