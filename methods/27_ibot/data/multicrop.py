"""
Multi-crop augmentation dataset and block masking for iBOT.

Multi-crop strategy (same as DINO / iBOT):
  - 2 global views at 224×224
  - n_local_crops local views at 96×96 (8 in the original iBOT code; 10 in some ablations)

Block masking:
  - Applied only to the student's global views.
  - Samples prediction ratios with the official iBOT pred_ratio/pred_ratio_var logic.
  - Uses a greedy block-wise strategy: randomly draw rectangular blocks until
    the target number of masked patches is reached (similar to BEiT).

Reference: Zhou et al., arXiv:2111.07832, Appendix A.
"""

import math
import random
import numpy as np
import torch
from PIL import ImageFilter, ImageOps
from torchvision import datasets, transforms


# ── Augmentation primitives ──────────────────────────────────────────────────

class GaussianBlur:
    """Apply Gaussian blur with a random kernel radius."""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p          = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img


class Solarization:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


# ── Block masking ────────────────────────────────────────────────────────────

class BlockMaskGenerator:
    """
    Generates random block masks for iBOT.

    The strategy iteratively places random rectangular blocks of patches
    until the desired mask ratio is achieved, closely following the BEiT /
    iBOT block masking approach.

    Args:
        input_size   : Spatial resolution of the image (e.g., 224).
        patch_size   : Patch size (e.g., 16).
        pred_ratio   : Official iBOT prediction ratio list or scalar.
        pred_ratio_var: Official iBOT prediction ratio variance list or scalar.
        min_aspect   : Minimum aspect ratio of a block.
        max_aspect   : Maximum aspect ratio of a block.
    """

    def __init__(self, input_size=224, patch_size=16,
                 pred_ratio=(0.0, 0.3), pred_ratio_var=(0.0, 0.2),
                 pred_shape="block", pred_start_epoch=0,
                 min_aspect=0.3, max_aspect=None):
        self.num_patches_h = input_size // patch_size
        self.num_patches_w = input_size // patch_size
        self.num_patches   = self.num_patches_h * self.num_patches_w
        self.pred_ratio = self._normalize_ratio_arg(pred_ratio)
        self.pred_ratio_var = self._normalize_ratio_arg(pred_ratio_var)
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        if isinstance(self.pred_ratio, list) and len(self.pred_ratio) != len(self.pred_ratio_var):
            raise ValueError("pred_ratio and pred_ratio_var must have the same length")
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        self.epoch = 0
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect or (1 / min_aspect)

    @staticmethod
    def _normalize_ratio_arg(value):
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            return [float(v) for v in value]
        return float(value)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_pred_ratio(self):
        """Match bytedance/ibot ImageFolderMask.get_pred_ratio sampling."""
        if self.epoch < self.pred_start_epoch:
            return 0.0
        if isinstance(self.pred_ratio, list):
            pred_ratios = []
            for ratio, ratio_var in zip(self.pred_ratio, self.pred_ratio_var):
                if ratio < ratio_var:
                    raise ValueError("Each pred_ratio entry must be >= pred_ratio_var")
                pred_ratios.append(
                    random.uniform(ratio - ratio_var, ratio + ratio_var)
                    if ratio_var > 0 else ratio
                )
            return random.choice(pred_ratios)
        if self.pred_ratio < self.pred_ratio_var:
            raise ValueError("pred_ratio must be >= pred_ratio_var")
        return (
            random.uniform(self.pred_ratio - self.pred_ratio_var,
                           self.pred_ratio + self.pred_ratio_var)
            if self.pred_ratio_var > 0 else self.pred_ratio
        )

    def __call__(self, batch_size=1):
        """Generate a batch of block masks.

        Returns:
            masks: [batch_size, num_patches] bool tensor  (True = masked)
        """
        masks = []
        for _ in range(batch_size):
            pred_ratio = self.get_pred_ratio()
            num_mask = int(self.num_patches * pred_ratio)
            if num_mask <= 0:
                mask = torch.zeros(self.num_patches, dtype=torch.bool)
            elif self.pred_shape == "block":
                mask = self._make_block_mask(num_mask)
            elif self.pred_shape == "rand":
                mask = torch.zeros(self.num_patches, dtype=torch.bool)
                mask[torch.randperm(self.num_patches)[:num_mask]] = True
            else:
                raise ValueError(f"Unsupported pred_shape: {self.pred_shape}")
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def _make_block_mask(self, num_mask):
        """Create a single block mask with approximately `num_mask` masked patches."""
        H = self.num_patches_h
        W = self.num_patches_w
        mask = torch.zeros(H * W, dtype=torch.bool)
        mask_grid = mask.view(H, W)
        mask_count = 0
        low = (min(H, W) // 3) ** 2

        while mask_count < num_mask:
            max_mask_patches = num_mask - mask_count
            delta = 0
            for _attempt in range(10):
                target_area = random.uniform(low, max_mask_patches)
                aspect = math.exp(random.uniform(math.log(self.min_aspect), math.log(self.max_aspect)))
                bh = int(round(math.sqrt(target_area * aspect)))
                bw = int(round(math.sqrt(target_area / aspect)))
                if bw < W and bh < H:
                    top = random.randint(0, H - bh)
                    left = random.randint(0, W - bw)
                    num_masked = mask_grid[top: top + bh, left: left + bw].sum().item()
                    if 0 < bh * bw - num_masked <= max_mask_patches:
                        for r in range(top, top + bh):
                            for c in range(left, left + bw):
                                if not mask_grid[r, c]:
                                    mask_grid[r, c] = True
                                    delta += 1
                    if delta > 0:
                        break
            if delta == 0:
                break
            mask_count += delta
        return mask


# ── Multi-crop augmentation ──────────────────────────────────────────────────

def _build_global_aug(img_size=224, scale=(0.4, 1.0), step="step1"):
    """
    Global-crop augmentation.
    Both iBOT stages use the official asymmetric DINO/iBOT global views.
    """
    if step not in {"step1", "step2"}:
        raise ValueError(f"Unsupported iBOT training step: {step}")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    flip_jitter = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]

    aug1 = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
        *flip_jitter,
        GaussianBlur(p=1.0),
        transforms.ToTensor(),
        norm,
    ])
    aug2 = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
        *flip_jitter,
        GaussianBlur(p=0.1),
        Solarization(p=0.2),
        transforms.ToTensor(),
        norm,
    ])
    return [aug1, aug2]


def _build_local_aug(img_size=96, scale=(0.05, 0.25)):
    """Local-crop augmentation (96×96, no masking)."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        norm,
    ])


# ── Dataset Wrapper ──────────────────────────────────────────────────────────

class iBOTDataset(torch.utils.data.Dataset):
    """
    Wraps ImageFolder to produce multi-crop views + block masks for iBOT.

    Returns:
        A tuple (crops, masks, label) where:
          crops  : list of tensors — [global1, global2, local1, ..., localN]
          masks  : list of bool tensors [B,N] for the two global views (generated per-item)
          label  : integer class index
    """

    def __init__(
        self,
        data_path,
        n_local_crops=10,
        global_size=224,
        local_size=96,
        patch_size=16,
        global_crops_scale=(0.25, 1.0),
        local_crops_scale=(0.05, 0.25),
        pred_ratio=(0.0, 0.3),
        pred_ratio_var=(0.0, 0.2),
        pred_shape="block",
        pred_start_epoch=0,
        step="step1",
    ):
        self.dataset = datasets.ImageFolder(data_path)
        self.n_local_crops = n_local_crops

        self.global_augs = _build_global_aug(global_size, scale=tuple(global_crops_scale), step=step)
        self.local_aug   = _build_local_aug(local_size, scale=tuple(local_crops_scale))

        num_patches = (global_size // patch_size) ** 2
        self.mask_gen = BlockMaskGenerator(
            input_size=global_size,
            patch_size=patch_size,
            pred_ratio=pred_ratio,
            pred_ratio_var=pred_ratio_var,
            pred_shape=pred_shape,
            pred_start_epoch=pred_start_epoch,
        )
        self._num_patches = num_patches

    def set_epoch(self, epoch):
        self.mask_gen.set_epoch(epoch)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Global views (with per-image masks for the student)
        global_crops = [aug(img) for aug in self.global_augs]
        # Generate one mask per global view (shape [num_patches])
        masks = [self.mask_gen(batch_size=1)[0] for _ in range(len(global_crops))]

        # Local views (no masking)
        local_crops = [self.local_aug(img) for _ in range(self.n_local_crops)]

        crops = global_crops + local_crops
        return crops, masks, label


def ibot_collate_fn(batch):
    """
    Custom collate: stack crops and masks into batch tensors.

    Returns:
        crops : list of [B, C, H, W] tensors (n_crops entries)
        masks : list of [B, N] bool tensors  (n_global entries)
        labels: [B] int tensor
    """
    all_crops, all_masks, all_labels = zip(*batch)
    n_crops  = len(all_crops[0])
    n_global = len(all_masks[0])

    crops  = [torch.stack([all_crops[b][c] for b in range(len(batch))]) for c in range(n_crops)]
    masks  = [torch.stack([all_masks[b][m] for b in range(len(batch))]) for m in range(n_global)]
    labels = torch.tensor(all_labels, dtype=torch.long)
    return crops, masks, labels


def get_ibot_dataloader(
    data_path,
    batch_size,
    num_workers=8,
    n_local_crops=10,
    global_size=224,
    local_size=96,
    patch_size=16,
    global_crops_scale=(0.25, 1.0),
    local_crops_scale=(0.05, 0.25),
    pred_ratio=(0.0, 0.3),
    pred_ratio_var=(0.0, 0.2),
    mask_ratio_min=None,
    mask_ratio_max=None,
    pred_shape="block",
    pred_start_epoch=0,
    step="step1",
    distributed=False,
):
    """Build the iBOT training dataloader."""
    if mask_ratio_min is not None or mask_ratio_max is not None:
        if mask_ratio_min is None or mask_ratio_max is None:
            raise ValueError("mask_ratio_min and mask_ratio_max must be set together")
        lo = float(mask_ratio_min)
        hi = float(mask_ratio_max)
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"Invalid mask ratio range: [{lo}, {hi}]")
        center = 0.5 * (lo + hi)
        spread = 0.5 * (hi - lo)
        pred_ratio = center
        pred_ratio_var = spread

    dataset = iBOTDataset(
        data_path=data_path,
        n_local_crops=n_local_crops,
        global_size=global_size,
        local_size=local_size,
        patch_size=patch_size,
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        pred_ratio=pred_ratio,
        pred_ratio_var=pred_ratio_var,
        pred_shape=pred_shape,
        pred_start_epoch=pred_start_epoch,
        step=step,
    )

    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        if distributed else
        torch.utils.data.RandomSampler(dataset)
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ibot_collate_fn,
    )
    return loader, sampler
