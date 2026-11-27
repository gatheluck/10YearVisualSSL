"""Blockwise masking for BEiT (Bao et al., 2021), ported from the lab's code.

Samples rectangular blocks until >= num_masking_patches patches are masked, then
trims/tops-up to exactly that count. Returns a flat (num_patches,) bool mask.
"""

from __future__ import annotations

import math
import random

import numpy as np


class BEiTMaskingGenerator:
    def __init__(self, input_size=(14, 14), num_masking_patches: int = 75,
                 min_num_patches: int = 16, max_num_patches: "int | None" = None,
                 min_aspect: float = 0.3, max_aspect: "float | None" = None):
        if not isinstance(input_size, (tuple, list)):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = (num_masking_patches if max_num_patches is None
                                else max_num_patches)
        max_aspect = max_aspect or (1.0 / min_aspect)
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _get_block_shape(self) -> tuple:
        for _ in range(10):
            area = random.randint(self.min_num_patches, self.max_num_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(area * aspect_ratio)))
            w = int(round(math.sqrt(area / aspect_ratio)))
            if h >= 1 and w >= 1 and h <= self.height and w <= self.width:
                return h, w
        return 1, 1

    def __call__(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        num_masked = 0
        for _ in range(100):
            if num_masked >= self.num_masking_patches:
                break
            h, w = self._get_block_shape()
            if num_masked + h * w > self.num_masking_patches + self.max_num_patches:
                continue
            top = random.randint(0, self.height - h)
            left = random.randint(0, self.width - w)
            mask[top:top + h, left:left + w] = True
            num_masked = int(mask.sum())

        if num_masked > self.num_masking_patches:
            masked_indices = np.argwhere(mask.ravel()).ravel()
            np.random.shuffle(masked_indices)
            excess = num_masked - self.num_masking_patches
            for idx in masked_indices[:excess]:
                row, col = divmod(int(idx), self.width)
                mask[row, col] = False

        if int(mask.sum()) < self.num_masking_patches:
            unmasked = np.argwhere(~mask.ravel()).ravel()
            np.random.shuffle(unmasked)
            need = self.num_masking_patches - int(mask.sum())
            for idx in unmasked[:need]:
                row, col = divmod(int(idx), self.width)
                mask[row, col] = True

        return mask.ravel()
