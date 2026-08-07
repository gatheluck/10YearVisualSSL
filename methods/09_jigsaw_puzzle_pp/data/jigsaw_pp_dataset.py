"""Jigsaw Puzzle++ pretext dataset (Noroozi et al., CVPR 2018), ported from the
lab's own implementation.

Differences from the original Jigsaw Puzzle, all kept here:
- up to 2 tiles are replaced with tiles from another random image (occlusions),
- a high-Hamming-distance permutation set (701 in the paper),
- 70% of images are converted to grayscale during training,
- each tile is normalized independently by its own per-channel mean/std.

The permutation set is generated deterministically (a seeded ``random.Random``)
so two runs share the same set; the capture's on-disk ``.pkl`` cache is dropped.
The capture's ``puzzle_size`` canvas-assembly path is dropped too: this port
returns the 9 tiles stacked as ``[9, 3, tile, tile]``, which is what the VGG16
model consumes.
"""

from __future__ import annotations

import os
import random
from itertools import permutations as iter_perms
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float32 [C, H, W] tensor in [0, 1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr.copy())
    return t.permute(2, 0, 1).float().div_(255.0)


def _normalize_tile_independently(tile: torch.Tensor) -> torch.Tensor:
    """Normalize one tile by its own per-channel mean/std."""
    mean = tile.mean(dim=(1, 2), keepdim=True)
    std = tile.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (tile - mean) / std


def _hamming_distance(p1: List[int], p2: List[int]) -> int:
    return sum(a != b for a, b in zip(p1, p2))


def generate_jigsaw_pp_permutations(num_tiles: int = 9, min_hamming: int = 3,
                                    target_count: int = 701,
                                    seed: int = 42) -> List[List[int]]:
    """Generate permutations with pairwise Hamming distance >= min_hamming,
    deterministically (the paper: 701 permutations, min HD 3)."""
    rng = random.Random(seed)
    all_perms = list(iter_perms(range(num_tiles)))
    rng.shuffle(all_perms)

    selected: List[List[int]] = []
    for perm in all_perms:
        perm = list(perm)
        if all(_hamming_distance(perm, s) >= min_hamming for s in selected):
            selected.append(perm)
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        all_perms2 = list(iter_perms(range(num_tiles)))
        rng.shuffle(all_perms2)
        for perm in all_perms2:
            perm = list(perm)
            if perm not in selected:
                selected.append(perm)
            if len(selected) >= target_count:
                break

    return selected[:target_count]


class JigsawPPDataset(data.Dataset):
    def __init__(
        self,
        image_folder: str,
        permutations: Optional[List[List[int]]] = None,
        num_permutations: int = 701,
        tile_size: int = 75,
        tile_gap: int = 0,
        image_size: int = 255,
        mode: str = "train",
        grayscale_prob: float = 0.7,
        max_occlusions: int = 2,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.tile_size = tile_size
        self.tile_gap = tile_gap
        self.image_size = image_size
        self.mode = mode
        self.grayscale_prob = grayscale_prob if mode == "train" else 0.0
        self.max_occlusions = max_occlusions if mode == "train" else 0
        self.num_tiles = 9
        self.grid_size = 3 * tile_size + 2 * tile_gap

        self.permutations = (permutations if permutations is not None
                             else generate_jigsaw_pp_permutations(
                                 target_count=num_permutations))
        self.num_permutations = len(self.permutations)

        self.image_paths = self._find_images(image_folder)
        if not self.image_paths:
            raise ValueError(
                f"JigsawPPDataset found no supported images in {image_folder}")
        if self.max_occlusions > 0 and len(self.image_paths) < 2:
            raise ValueError(
                "Jigsaw++ occlusion replacement requires at least two source "
                "images")

        self._to_gray = transforms.Grayscale(num_output_channels=3)

    @staticmethod
    def _find_images(folder: str) -> List[str]:
        paths = [os.path.join(root, fn)
                 for root, _, files in os.walk(folder)
                 for fn in files
                 if os.path.splitext(fn)[1].lower() in _IMG_EXTS]
        return sorted(paths)

    def _load_and_resize(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if min(w, h) != self.image_size:
            scale = self.image_size / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return img

    def _random_crop_grid(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.grid_size or h < self.grid_size:
            scale = max(self.grid_size / w, self.grid_size / h) * 1.1
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            w, h = img.size
        if self.mode == "train":
            x0 = random.randint(0, w - self.grid_size)
            y0 = random.randint(0, h - self.grid_size)
        else:
            x0 = (w - self.grid_size) // 2
            y0 = (h - self.grid_size) // 2
        return img.crop((x0, y0, x0 + self.grid_size, y0 + self.grid_size))

    def _extract_tiles(self, grid: Image.Image) -> List[Image.Image]:
        tiles = []
        step = self.tile_size + self.tile_gap
        for row in range(3):
            for col in range(3):
                x, y = col * step, row * step
                tiles.append(grid.crop(
                    (x, y, x + self.tile_size, y + self.tile_size)))
        return tiles

    def _tile_to_tensor(self, tile: Image.Image) -> torch.Tensor:
        return _normalize_tile_independently(_pil_to_float_tensor(tile))

    def __len__(self) -> int:
        return len(self.image_paths)

    def _sample_occlusion_source_index(self, idx: int) -> int:
        """Sample uniformly from every image index except the puzzle source."""
        length = len(self.image_paths)
        sampled = random.randint(0, length - 2)
        return sampled if sampled < idx else sampled + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        img = self._load_and_resize(path)
        grid = self._random_crop_grid(img)
        if random.random() < self.grayscale_prob:
            grid = self._to_gray(grid)
        tiles = self._extract_tiles(grid)

        if self.max_occlusions > 0:
            n_occ = random.randint(0, self.max_occlusions)
            if n_occ > 0:
                occ_positions = random.sample(range(self.num_tiles), n_occ)
                rand_idx = self._sample_occlusion_source_index(idx)
                rand_img = self._load_and_resize(self.image_paths[rand_idx])
                if random.random() < self.grayscale_prob:
                    rand_img = self._to_gray(rand_img)
                rand_tiles = self._extract_tiles(self._random_crop_grid(rand_img))
                for pos in occ_positions:
                    tiles[pos] = rand_tiles[pos]

        if self.mode == "train":
            perm_idx = random.randint(0, self.num_permutations - 1)
        else:
            perm_idx = idx % self.num_permutations
        permutation = self.permutations[perm_idx]
        permuted = [tiles[i] for i in permutation]

        stacked = torch.stack([self._tile_to_tensor(t) for t in permuted], dim=0)
        return stacked, torch.tensor(perm_idx, dtype=torch.long)
