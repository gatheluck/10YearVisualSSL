"""ImageNet dataset for the CMC protocol (Tian et al., 2019), ported from the
lab's own implementation.

An RGB image is converted to CIE **Lab** and returned as a 3-channel tensor
(L, a, b); the model splits it into the L and ab views. Each item is
``(lab_tensor, label, index)`` -- the index is the image's position in the
dataset, which the NCE memory bank uses as the instance identity.

The lab's conversion uses ``skimage.color.rgb2lab`` (with a PIL fallback). This
port **reimplements rgb2lab in numpy** (sRGB -> linear -> XYZ(D65) -> CIE Lab),
verified against published CIE Lab reference values, so the port keeps the
torch-only closure -- scikit-image is not a dependency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# sRGB (D65) -> XYZ matrix and the D65 white point, matching skimage's rgb2lab.
_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]], dtype=np.float64)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

# Per-channel Lab statistics used for normalisation (the extremes of the Lab
# gamut, as in the lab's own code): mean = midpoint, std = half-range.
LAB_MEAN = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
LAB_STD = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]


class RGB2Lab:
    """Convert an RGB PIL image to a CIE Lab float32 ndarray (H x W x 3).

    torchvision's ``ToTensor`` then transposes to C x H x W without any /255
    scaling, since the array is already float.
    """

    def __call__(self, img) -> np.ndarray:
        rgb = np.asarray(img, dtype=np.float64) / 255.0
        lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4,
                       rgb / 12.92)
        xyz = (lin @ _RGB2XYZ.T) / _WHITE_D65
        d = 6.0 / 29.0
        f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1).astype(np.float32)


def _build_transform(mode: str, image_size: int, crop_low: float):
    normalize = transforms.Normalize(LAB_MEAN, LAB_STD)
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(crop_low, 1.0)),
            transforms.RandomHorizontalFlip(),
            RGB2Lab(),
            transforms.ToTensor(),
            normalize,
        ])
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        RGB2Lab(),
        transforms.ToTensor(),
        normalize,
    ])


class CMCDataset(Dataset):
    """ImageFolder wrapper returning ``(lab_tensor, label, index)``."""

    def __init__(self, root: str, mode: str = "train", image_size: int = 224,
                 crop_low: float = 0.2):
        self.mode = mode
        self.base = datasets.ImageFolder(
            root, transform=_build_transform(mode, image_size, crop_low))

    @property
    def classes(self):
        return self.base.classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple:
        image, label = self.base[index]
        return image, label, index
