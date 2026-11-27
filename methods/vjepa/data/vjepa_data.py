"""V-JEPA data pipeline (Bardes et al., 2024; arXiv:2404.08471).

An ImageFolder with a standard SSL train transform, collated by the official
`facebookresearch/jepa` 3D multi-block mask collator (`src.masks.multiblock3d`),
imported from the pinned submodule. The collator returns the batched images plus
the encoder (context) and predictor (target) block masks. At num_frames=1 the
masks are 2D image blocks. The MaskCollator is imported **lazily** (its package
top level is a generic `src` name) and the port drops the DistributedSampler
(single-process) and threads a seeded DataLoader generator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import adapterlib

_JEPA_SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "jepa"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(crop_size: int = 224,
                          use_color_jitter: bool = True) -> transforms.Compose:
    steps = [transforms.RandomResizedCrop(crop_size, scale=(0.3, 1.0)),
             transforms.RandomHorizontalFlip()]
    if use_color_jitter:
        steps.append(transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
    steps += [transforms.ToTensor(),
              transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
    return transforms.Compose(steps)


def val_transform(img_size: int = 224) -> transforms.Compose:
    resize = int(round(img_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _prepare_jepa_path() -> None:
    """Drop any cached `src*`/`app*`, remove every other third_party root from
    sys.path, and put third_party/jepa first, so the mask collator resolves to
    THIS submodule's `src` (another submodule port also exposes a top-level `src`,
    a namespace package that would otherwise merge both)."""
    for key in [k for k in sys.modules
                if k in ("src", "app") or k.startswith(("src.", "app."))]:
        del sys.modules[key]
    tp = str(_JEPA_SUBMODULE.parent) + os.sep
    sys.path[:] = [q for q in sys.path if not q.startswith(tp)]
    sys.path.insert(0, str(_JEPA_SUBMODULE))


def _mask_collator(crop_size, num_frames, patch_size, tubelet_size, cfgs_mask):
    _prepare_jepa_path()
    from src.masks.multiblock3d import MaskCollator
    return MaskCollator(crop_size=int(crop_size), num_frames=int(num_frames),
                        patch_size=int(patch_size), tubelet_size=int(tubelet_size),
                        cfgs_mask=cfgs_mask)


def get_vjepa_dataloader(data_path: str, batch_size: int, cfgs_mask,
                         crop_size: int = 224, num_frames: int = 1,
                         patch_size: int = 16, tubelet_size: int = 1,
                         use_color_jitter: bool = True, num_workers: int = 8,
                         seed: int = 0):
    """Single-process V-JEPA DataLoader.

    Yields (images, labels, masks_enc, masks_pred): images [B,C,H,W], and the two
    lists of block masks the encoder and predictor consume.
    """
    dataset = datasets.ImageFolder(
        adapterlib.dataset_split_dir(data_path, "train"),
        transform=build_train_transform(crop_size, use_color_jitter))
    collator = _mask_collator(crop_size, num_frames, patch_size, tubelet_size,
                              cfgs_mask)

    def collate(batch):
        collated, masks_enc, masks_pred = collator(batch)
        images, labels = collated
        return images, labels, masks_enc, masks_pred

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed))
    return loader, dataset
