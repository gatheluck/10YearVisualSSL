"""Jigsaw puzzle dataset (Noroozi & Favaro, ECCV 2016), ported from the lab's own
implementation.

An image is resized, a 3x3 grid region is cropped, split into 9 tiles (with a
small gap so edge continuity is not a shortcut), the tiles are reordered by one
of a fixed permutation set, and the label is which permutation was applied. The
permutation set is derangements with pairwise Hamming distance >= 3, generated
deterministically (seed 42), so two runs share the same set.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class JigsawPuzzleDataset(data.Dataset):
    def __init__(
        self,
        image_folder: str,
        permutations: Optional[List[List[int]]] = None,
        num_permutations: int = 100,
        tile_size: int = 75,
        tile_gap: int = 2,
        image_size: int = 255,
        mode: str = "train",
        transform=None,
    ):
        self.image_folder = image_folder
        self.tile_size = tile_size
        self.tile_gap = tile_gap
        self.image_size = image_size
        self.mode = mode
        self.transform = transform
        self.base_transform = transforms.ToTensor()

        self.permutations = (permutations if permutations is not None
                             else self._generate_permutations(num_permutations))
        self.num_permutations = len(self.permutations)
        self.image_paths = self._find_images(image_folder)

        self.grid_size = 3 * tile_size + 2 * tile_gap
        if self.grid_size > image_size:
            raise ValueError(
                f"grid_size {self.grid_size} (3*tile_size + 2*tile_gap) exceeds "
                f"image_size {image_size}; the tiles would not fit")

    @staticmethod
    def _find_images(folder: str) -> List[str]:
        paths = [str(p) for p in Path(folder).rglob("*")
                 if p.suffix.lower() in _IMG_EXTS]
        if not paths:
            raise RuntimeError(f"no images under {folder}")
        return sorted(paths)

    def _generate_permutations(self, num_permutations: int) -> List[List[int]]:
        """Derangements (no tile in its original place) with pairwise Hamming
        distance >= 3, deterministically (seed 42)."""
        np.random.seed(42)
        candidates: List[List[int]] = []
        for _ in range(100000):
            perm = np.random.permutation(9).tolist()
            if not all(perm[i] != i for i in range(9)):
                continue
            if all(sum(a != b for a, b in zip(perm, e)) >= 3 for e in candidates):
                candidates.append(perm)
            if len(candidates) >= num_permutations:
                break
        while len(candidates) < num_permutations:
            perm = np.random.permutation(9).tolist()
            if all(perm[i] != i for i in range(9)):
                candidates.append(perm)
        return candidates[:num_permutations]

    def _extract_tiles(self, img: Image.Image) -> List[Image.Image]:
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        w, h = img.size
        if self.mode == "train":
            max_x, max_y = w - self.grid_size, h - self.grid_size
            crop_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            crop_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        else:
            crop_x = (w - self.grid_size) // 2
            crop_y = (h - self.grid_size) // 2
        region = img.crop((crop_x, crop_y, crop_x + self.grid_size,
                           crop_y + self.grid_size))
        step = self.tile_size + self.tile_gap
        tiles = []
        for row in range(3):
            for col in range(3):
                left, top = col * step, row * step
                tiles.append(region.crop((left, top, left + self.tile_size,
                                          top + self.tile_size)))
        return tiles

    def _apply_permutation(self, tiles, perm_idx):
        permuted = [tiles[i] for i in self.permutations[perm_idx]]
        out = []
        for tile in permuted:
            if self.transform:
                tile = self.transform(tile)
            out.append(self.base_transform(tile))
        return out

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        tiles = self._extract_tiles(img)
        if self.mode == "train":
            perm_idx = int(np.random.randint(0, self.num_permutations))
        else:
            perm_idx = idx % self.num_permutations
        tile_tensors = self._apply_permutation(tiles, perm_idx)
        return torch.stack(tile_tensors, dim=0), torch.tensor(perm_idx,
                                                              dtype=torch.long)
