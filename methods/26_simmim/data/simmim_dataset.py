"""SimMIM data loading and mask generation (Xie et al., 2022).

Ported from the lab's own SimMIM code (which follows microsoft/SimMIM):

  MaskGenerator: a random block mask on the model-patch grid; each sampled mask
    unit covers mask_patch_size / model_patch_size patch tokens.
  SimMIMDataset: an ImageFolder with the SimMIM pre-training augmentation
    (RandomResizedCrop scale 0.67-1.0, HFlip, ImageNet normalise) that attaches a
    random binary mask to each sample. Step 1 uses patch-grid masks.

The port drops the DistributedSampler (single-process) and threads a seeded
generator so a run is reproducible.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MaskGenerator:
    """A random binary block mask on the model patch grid. For Swin-B SimMIM
    (input 192, mask_patch 32, model_patch 4) this returns a 48x48 mask, each
    sampled unit repeated to an 8x8 block (microsoft/SimMIM data_simmim.py)."""

    def __init__(self, input_size, mask_patch_size, model_patch_size, mask_ratio):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert input_size % mask_patch_size == 0, (
            f"input_size ({input_size}) must be divisible by "
            f"mask_patch_size ({mask_patch_size})")
        assert mask_patch_size % model_patch_size == 0, (
            f"mask_patch_size ({mask_patch_size}) must be divisible by "
            f"model_patch_size ({model_patch_size})")
        assert input_size % model_patch_size == 0, (
            f"input_size ({input_size}) must be divisible by "
            f"model_patch_size ({model_patch_size})")

        self.rand_size = input_size // mask_patch_size
        self.scale = mask_patch_size // model_patch_size
        self.token_count = self.rand_size ** 2
        self.mask_count = int(math.ceil(self.token_count * mask_ratio))

    def __call__(self):
        """A binary numpy int32 array on the model patch grid."""
        idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=np.int32)
        mask[idx] = 1
        mask = mask.reshape(self.rand_size, self.rand_size)
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask


class SimMIMDataset(Dataset):
    """ImageFolder wrapped with SimMIM augmentation + masking. Returns
    ``(img (3,H,W), mask (H/patch, W/patch) float, label)`` for step 1."""

    def __init__(self, root, img_size, mask_patch_size, mask_ratio,
                 model_patch_size=None, return_pixel_mask=True):
        if model_patch_size is None:
            model_patch_size = mask_patch_size
        self.dataset = ImageFolder(root, transform=self._build_transform(img_size))
        self.model_patch_size = model_patch_size
        self.return_pixel_mask = return_pixel_mask
        self.mask_gen = MaskGenerator(
            img_size, mask_patch_size, model_patch_size, mask_ratio)

    @staticmethod
    def _build_transform(img_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.RandomResizedCrop(
                img_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        mask = self.mask_gen()
        if self.return_pixel_mask:
            mask = mask.repeat(self.model_patch_size, axis=0) \
                       .repeat(self.model_patch_size, axis=1)
        mask = torch.from_numpy(mask).float()
        return img, mask, label


def get_simmim_dataloader(data_path, img_size, mask_patch_size, mask_ratio,
                          batch_size, num_workers=8, model_patch_size=None,
                          return_pixel_mask=False, seed=0):
    """Single-process DataLoader for SimMIM step 1. Loads from
    ``data_path/train`` (ImageNet-style layout); yields (img, mask, label)."""
    dataset = SimMIMDataset(
        root=str(data_path) + "/train", img_size=img_size,
        mask_patch_size=mask_patch_size, mask_ratio=mask_ratio,
        model_patch_size=model_patch_size, return_pixel_mask=return_pixel_mask)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
