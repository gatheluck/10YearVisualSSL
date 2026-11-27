"""DINOv2 multi-crop dataset with batch-time iBOT block masking.

Ported from the capture's `methods/28_dinov2/data/dinov2_dataset.py`: 2 global
crops + N local crops, and a stratified 0.1-0.5 mask-ratio sampling applied over
all global-crop samples in the batch (in the collate fn, as in the official code).
The port drops the DistributedSampler (single-process: RandomSampler) and threads
a seeded generator so a run is reproducible.
"""

from __future__ import annotations

import math
import random
from functools import partial

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder


class BlockMaskGenerator:
    """Random iBOT block masks (flat bool, True = masked)."""

    def __init__(self, num_patches: int, masking_ratio: "float | None" = None,
                 min_num_patches: int = 4, min_aspect: float = 0.3,
                 max_aspect: float = 3.0):
        self.default_num_masking_patches = (
            None if masking_ratio is None else int(num_patches * masking_ratio))
        self.min_num_patches = min_num_patches
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        side = int(math.sqrt(num_patches))
        assert side * side == num_patches, \
            f"num_patches={num_patches} must be a perfect square."
        self.h = self.w = side
        self.num_patches = num_patches

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect = math.exp(
                random.uniform(math.log(self.min_aspect), math.log(self.max_aspect)))
            h_b = int(round(math.sqrt(target_area * aspect)))
            w_b = int(round(math.sqrt(target_area / aspect)))
            if w_b >= self.w or h_b >= self.h:
                continue
            top = random.randint(0, self.h - h_b)
            left = random.randint(0, self.w - w_b)
            already_masked = mask[top: top + h_b, left: left + w_b].sum()
            if 0 < h_b * w_b - already_masked <= max_mask_patches:
                block = mask[top: top + h_b, left: left + w_b]
                delta = int((~block).sum())
                block[:] = True
            if delta > 0:
                break
        return delta

    def __call__(self, num_masking_patches: "int | None" = None) -> torch.Tensor:
        if num_masking_patches is None:
            if self.default_num_masking_patches is None:
                raise ValueError("num_masking_patches is required")
            num_masking_patches = self.default_num_masking_patches
        num_masking_patches = max(0, min(int(num_masking_patches), self.num_patches))
        mask = np.zeros((self.h, self.w), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            delta = self._mask(mask, num_masking_patches - mask_count)
            if delta == 0:
                break
            mask_count += delta
        return torch.from_numpy(mask.flatten())


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


def collate_dinov2_batch(batch, mask_ratio_min_max=(0.1, 0.5),
                         mask_sample_probability: float = 0.5,
                         num_patches: "int | None" = None,
                         mask_generator: "BlockMaskGenerator | None" = None):
    global_crops_list = [item[0] for item in batch]
    local_crops_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    n_global = len(global_crops_list[0])
    n_local = len(local_crops_list[0])
    global_crops = [torch.stack([s[i] for s in global_crops_list]) for i in range(n_global)]
    local_crops = [torch.stack([s[i] for s in local_crops_list]) for i in range(n_local)]
    if num_patches is None or mask_generator is None:
        raise ValueError("num_patches and mask_generator are required for collation")
    ratio_min, ratio_max = (float(v) for v in mask_ratio_min_max)
    if not 0.0 <= ratio_min <= ratio_max <= 1.0:
        raise ValueError("mask_ratio_min_max must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= mask_sample_probability <= 1.0:
        raise ValueError("mask_sample_probability must be in [0, 1]")
    num_global_samples = len(batch) * n_global
    num_masked_samples = int(num_global_samples * mask_sample_probability)
    ratio_boundaries = torch.linspace(ratio_min, ratio_max, num_masked_samples + 1).tolist()
    flat_masks = []
    for index in range(num_masked_samples):
        ratio = random.uniform(ratio_boundaries[index], ratio_boundaries[index + 1])
        flat_masks.append(mask_generator(int(num_patches * ratio)))
    flat_masks.extend(
        torch.zeros(num_patches, dtype=torch.bool)
        for _ in range(num_masked_samples, num_global_samples))
    random.shuffle(flat_masks)
    collated_masks = torch.stack(flat_masks).view(n_global, len(batch), num_patches)
    masks = list(collated_masks.unbind(0))
    labels = torch.tensor(labels, dtype=torch.long)
    return global_crops, local_crops, masks, labels


class DINOv2MultiCropDataset(torch.utils.data.Dataset):
    """ImageFolder + DINOv2 multi-crop augmentation. Returns
    (global_crops, local_crops, label); masks are sampled by collation."""

    def __init__(self, dataset, n_global_crops: int = 2, n_local_crops: int = 10,
                 global_crop_size: int = 224, local_crop_size: int = 96,
                 global_crops_scale: tuple = (0.32, 1.0),
                 local_crops_scale: tuple = (0.05, 0.32), patch_size: int = 16):
        self.dataset = dataset
        self.n_global = n_global_crops
        self.n_local = n_local_crops
        self.global_transforms = [
            build_dinov2_augmentation(
                is_global=True, global_crop_size=global_crop_size,
                local_crop_size=local_crop_size, global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale, is_second_global=(i == 1))
            for i in range(n_global_crops)]
        self.local_transforms = [
            build_dinov2_augmentation(
                is_global=False, global_crop_size=global_crop_size,
                local_crop_size=local_crop_size, global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale)
            for _ in range(n_local_crops)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        global_crops = [t(img) for t in self.global_transforms]
        local_crops = [t(img) for t in self.local_transforms]
        return global_crops, local_crops, label


def get_dinov2_dataloader(data_path: str, cfg: dict, batch_size: int,
                          distributed: bool = False, seed: int = 0):
    """Single-process DINOv2 multi-crop dataloader. Reads data_path directly
    (the adapter passes <data_root>/train)."""
    base_dataset = ImageFolder(data_path)
    dataset = DINOv2MultiCropDataset(
        dataset=base_dataset,
        n_global_crops=cfg["data"]["n_global_crops"],
        n_local_crops=cfg["data"]["n_local_crops"],
        global_crop_size=cfg["data"]["global_crop_size"],
        local_crop_size=cfg["data"]["local_crop_size"],
        global_crops_scale=tuple(cfg["data"]["global_crops_scale"]),
        local_crops_scale=tuple(cfg["data"]["local_crops_scale"]),
        patch_size=cfg["model"]["patch_size"])
    num_patches = (cfg["data"]["global_crop_size"] // cfg["model"]["patch_size"]) ** 2
    mask_ratio_min_max = tuple(cfg["ibot"].get(
        "mask_ratio_min_max", [cfg["ibot"].get("masking_ratio", 0.5)] * 2))
    mask_sample_probability = cfg["ibot"].get("mask_sample_probability", 1.0)
    mask_generator = BlockMaskGenerator(num_patches)
    sampler = (torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
               if distributed else torch.utils.data.RandomSampler(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=partial(collate_dinov2_batch,
                           mask_ratio_min_max=mask_ratio_min_max,
                           mask_sample_probability=mask_sample_probability,
                           num_patches=num_patches, mask_generator=mask_generator))
    return loader, sampler
