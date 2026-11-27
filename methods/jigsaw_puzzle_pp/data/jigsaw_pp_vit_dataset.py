"""ViT Step-2 Jigsaw++ dataset: the 9 processed tiles reassembled into one image.

The capture's Step-2 feeds the ViT a single reassembled puzzle image, not the 9
separate tiles the native VGG16 path uses. This subclasses `JigsawPPDataset` to
reuse its grid crop, 70% grayscale, 0-2 occlusion replacement, permutation set,
and per-tile independent normalization, and assembles the permuted tiles onto a
`[3, G, G]` canvas (G = 3*tile_size + 2*tile_gap = 224 for the shipped config).
The native tiles dataset is untouched.
"""

from __future__ import annotations

import random

import torch

from .jigsaw_pp_dataset import JigsawPPDataset


class JigsawPPViTDataset(JigsawPPDataset):
    """Returns `(puzzle_image [3, G, G], permutation_index)`."""

    def __getitem__(self, idx: int):
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
                rand_tiles = self._extract_tiles(
                    self._random_crop_grid(rand_img))
                for pos in occ_positions:
                    tiles[pos] = rand_tiles[pos]

        if self.mode == "train":
            perm_idx = random.randint(0, self.num_permutations - 1)
        else:
            perm_idx = idx % self.num_permutations
        permuted = [tiles[i] for i in self.permutations[perm_idx]]

        step = self.tile_size + self.tile_gap
        canvas = torch.zeros(3, self.grid_size, self.grid_size)
        for k, tile in enumerate(permuted):
            row, col = divmod(k, 3)
            top, left = row * step, col * step
            canvas[:, top:top + self.tile_size,
                   left:left + self.tile_size] = self._tile_to_tensor(tile)
        return canvas, torch.tensor(perm_idx, dtype=torch.long)
