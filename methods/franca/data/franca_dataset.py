"""Franca multi-crop dataset with the official cyclic inverse-block masking.

Ported from the capture's `methods/36_franca/franca_data.py`. The capture imported
DINOv2's `build_dinov2_augmentation` across method directories; here the DINOv2
augmentation is **vendored** (copied below) so the method is self-contained and its
`data` package does not collide with another method's. The masking is Franca's own
`CyclicMaskGenerator` (inverse block + cyclic 2-D roll); the stratified 0.1-0.65
mask-ratio sampling over all global-crop samples is done in the collate fn, as in
the official code. The port drops the DistributedSampler (single-process) and
threads a seeded generator.
"""

from __future__ import annotations

import math
import random
from functools import partial

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.datasets import ImageFolder


# ── Vendored DINOv2 augmentation (identical to the DINOv2 port's) ─────────
class _GaussianBlur:
    def __init__(self, p: float = 1.0, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p, self.radius_min, self.radius_max = p, radius_min, radius_max

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.radius_min, self.radius_max)
            return TF.gaussian_blur(img, kernel_size=9, sigma=sigma)
        return img


class _Solarize:
    def __init__(self, p: float = 0.2, threshold: int = 128):
        self.p, self.threshold = p, threshold

    def __call__(self, img):
        if random.random() < self.p:
            return TF.solarize(img, self.threshold)
        return img


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_dinov2_augmentation(is_global: bool, global_crop_size: int = 224,
                              local_crop_size: int = 96,
                              global_crops_scale: tuple = (0.32, 1.0),
                              local_crops_scale: tuple = (0.05, 0.32),
                              is_second_global: bool = False) -> T.Compose:
    normalize = T.Compose([T.ToTensor(), T.Normalize(mean=_MEAN, std=_STD)])
    color_jitter = T.RandomApply(
        [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8)
    grayscale = T.RandomGrayscale(p=0.2)
    if is_global:
        crop_size, crop_scale = global_crop_size, global_crops_scale
        blur = _GaussianBlur(p=0.1 if is_second_global else 1.0)
        solar = _Solarize(p=0.2 if is_second_global else 0.0)
    else:
        crop_size, crop_scale = local_crop_size, local_crops_scale
        blur = _GaussianBlur(p=0.5)
        solar = _Solarize(p=0.0)
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=crop_scale,
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5), color_jitter, grayscale, blur, solar, normalize,
    ])


# ── Franca cyclic masking ─────────────────────────────────────────────────────
class CyclicMaskGenerator:
    """Official Franca inverse-block mask followed by a cyclic 2-D roll."""

    def __init__(self, input_size: tuple, min_aspect: float = 0.3,
                 max_aspect: "float | None" = None) -> None:
        self.height, self.width = input_size
        max_aspect = max_aspect or 1.0 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _visible_block(self, num_visible_patches: int) -> np.ndarray:
        min_lar = max(self.log_aspect_ratio[0],
                      math.log(num_visible_patches / (self.width ** 2)))
        max_lar = min(self.log_aspect_ratio[1],
                      math.log(self.height ** 2 / (num_visible_patches + 1e-5)))
        min_lar = min(min_lar, max_lar)
        aspect_ratio = math.exp(random.uniform(min_lar, max_lar))
        block_h = min(int(math.ceil(math.sqrt(num_visible_patches * aspect_ratio))),
                      self.height)
        block_w = min(int(math.ceil(math.sqrt(num_visible_patches / aspect_ratio))),
                      self.width)
        top = 0 if block_h >= self.height else random.randint(0, self.height - block_h)
        left = 0 if block_w >= self.width else random.randint(0, self.width - block_w)
        block = np.zeros((self.height, self.width), dtype=bool)
        block[top: top + block_h, left: left + block_w] = True
        visible_ids = np.flatnonzero(block)[:num_visible_patches]
        block.fill(False)
        block.flat[visible_ids] = True
        shift = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
        return np.roll(block, shift=shift, axis=(0, 1))

    def __call__(self, num_masking_patches: int = 0) -> torch.Tensor:
        total = self.height * self.width
        num_masking_patches = min(max(int(num_masking_patches), 0), total)
        if num_masking_patches == 0:
            return torch.zeros(total, dtype=torch.bool)
        if num_masking_patches == total:
            return torch.ones(total, dtype=torch.bool)
        visible = self._visible_block(total - num_masking_patches)
        return torch.from_numpy((~visible).reshape(-1).copy())


def generate_global_masks(n_global_samples: int, n_tokens: int, mask_ratio: tuple,
                          mask_probability: float,
                          mask_generator: CyclicMaskGenerator) -> list:
    ratio_min, ratio_max = (float(value) for value in mask_ratio)
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1]")
    if not 0.0 <= ratio_min <= ratio_max <= 1.0:
        raise ValueError("mask_ratio must satisfy 0 <= min <= max <= 1")
    n_masked_samples = int(n_global_samples * mask_probability)
    ratio_edges = torch.linspace(ratio_min, ratio_max, n_masked_samples + 1)
    masks = []
    for index in range(n_masked_samples):
        sampled_ratio = random.uniform(float(ratio_edges[index]),
                                       float(ratio_edges[index + 1]))
        masks.append(mask_generator(int(n_tokens * sampled_ratio)))
    masks.extend(mask_generator(0)
                 for _ in range(n_masked_samples, n_global_samples))
    random.shuffle(masks)
    return masks


class FrancaMultiCropDataset(Dataset):
    def __init__(self, dataset: Dataset, cfg: dict) -> None:
        self.dataset = dataset
        d = cfg["data"]
        self.global_transforms = [
            build_dinov2_augmentation(
                is_global=True, global_crop_size=int(d["global_crop_size"]),
                local_crop_size=int(d["local_crop_size"]),
                global_crops_scale=tuple(d["global_crops_scale"]),
                local_crops_scale=tuple(d["local_crops_scale"]),
                is_second_global=index == 1)
            for index in range(int(d["n_global_crops"]))]
        self.local_transforms = [
            build_dinov2_augmentation(
                is_global=False, global_crop_size=int(d["global_crop_size"]),
                local_crop_size=int(d["local_crop_size"]),
                global_crops_scale=tuple(d["global_crops_scale"]),
                local_crops_scale=tuple(d["local_crops_scale"]))
            for _ in range(int(d["n_local_crops"]))]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        global_crops = [t(image) for t in self.global_transforms]
        local_crops = [t(image) for t in self.local_transforms]
        return global_crops, local_crops, label


def collate_franca_batch(batch, n_tokens: int, mask_ratio: tuple,
                         mask_probability: float,
                         mask_generator: CyclicMaskGenerator):
    batch_size = len(batch)
    n_global_crops = len(batch[0][0])
    n_local_crops = len(batch[0][1])
    global_crops = [torch.stack([s[0][c] for s in batch])
                    for c in range(n_global_crops)]
    local_crops = [torch.stack([s[1][c] for s in batch])
                   for c in range(n_local_crops)]
    flat_masks = generate_global_masks(
        n_global_samples=batch_size * n_global_crops, n_tokens=n_tokens,
        mask_ratio=mask_ratio, mask_probability=mask_probability,
        mask_generator=mask_generator)
    masks = [torch.stack(flat_masks[start: start + batch_size])
             for start in range(0, len(flat_masks), batch_size)]
    labels = torch.tensor([s[2] for s in batch], dtype=torch.long)
    return global_crops, local_crops, masks, labels


def get_franca_dataloader(data_path: str, cfg: dict, batch_size: int,
                          distributed: bool = False, seed: int = 0):
    dataset = FrancaMultiCropDataset(ImageFolder(data_path), cfg)
    sampler = RandomSampler(dataset)
    grid_size = int(cfg["data"]["global_crop_size"]) // int(cfg["model"]["patch_size"])
    n_tokens = grid_size ** 2
    mask_generator = CyclicMaskGenerator((grid_size, grid_size), min_aspect=0.3)
    collate_fn = partial(collate_franca_batch, n_tokens=n_tokens,
                         mask_ratio=tuple(cfg["ibot"]["mask_ratio_min_max"]),
                         mask_probability=float(cfg["ibot"]["mask_sample_probability"]),
                         mask_generator=mask_generator)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=int(cfg["data"]["num_workers"]), pin_memory=True,
                        drop_last=True, generator=torch.Generator().manual_seed(seed),
                        collate_fn=collate_fn)
    return loader, sampler
