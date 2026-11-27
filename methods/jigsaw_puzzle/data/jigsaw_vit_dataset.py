"""ViT Step-2 jigsaw dataset: the 9 permuted tiles reassembled into one image.

The capture's Step-2 feeds the ViT a **single reassembled puzzle image**
(`[3, G, G]`, G = 3*tile_size + 2*tile_gap = 224 for the shipped config), not the
9 separate tiles the native CFN path uses. This subclasses the native
`JigsawPuzzleDataset` to reuse its tile extraction and the deterministic
permutation set, and assembles the permuted tiles onto a canvas with the tile
gap, ImageNet-normalised. The native dataset is untouched.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .jigsaw_dataset import JigsawPuzzleDataset

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class JigsawPuzzleViTDataset(JigsawPuzzleDataset):
    """Returns `(puzzle_image [3, G, G], permutation_index)`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        tiles = self._extract_tiles(img)                      # 9 PIL tiles
        if self.mode == "train":
            perm_idx = int(np.random.randint(0, self.num_permutations))
        else:
            perm_idx = idx % self.num_permutations
        permuted = [tiles[i] for i in self.permutations[perm_idx]]
        step = self.tile_size + self.tile_gap
        canvas = torch.zeros(3, self.grid_size, self.grid_size)
        for k, tile in enumerate(permuted):
            if self.transform:
                tile = self.transform(tile)
            row, col = divmod(k, 3)
            top, left = row * step, col * step
            canvas[:, top:top + self.tile_size,
                   left:left + self.tile_size] = self._to_norm(tile)
        return canvas, torch.tensor(perm_idx, dtype=torch.long)
