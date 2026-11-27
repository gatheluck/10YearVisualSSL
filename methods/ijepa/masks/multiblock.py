"""Multi-block masking for I-JEPA (Assran et al., 2023).

Ported from the lab's own code (which follows facebookresearch/ijepa
src/masks/multiblock.py). Used as the DataLoader ``collate_fn``: it wraps the
standard ImageFolder collate and appends (enc_masks, pred_masks).

  enc_masks  -- context encoder input: indices of patches the context encoder
                processes (one large rectangular block, minus the target blocks).
  pred_masks -- prediction targets: M rectangular target blocks.

Returned tensors:
  images     : [B, C, H, W]
  labels     : [B]
  enc_masks  : [B, n_ctx]   (LongTensor of patch indices for the context encoder)
  pred_masks : list of M [B, n_tgt_i]  (one tensor per target block)
"""

from __future__ import annotations

import math

import torch
from torch.utils.data.dataloader import default_collate


class MultiBlockMaskCollator:
    """Generates multi-block masks for a batch of images."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        enc_mask_scale: tuple = (0.85, 1.0),
        enc_mask_aspect: tuple = (1.0, 1.0),
        pred_mask_scale: tuple = (0.15, 0.2),
        pred_mask_aspect: tuple = (0.75, 1.5),
        num_enc_masks: int = 1,
        num_pred_masks: int = 4,
        allow_overlap: bool = False,
        min_keep: int = 10,
    ):
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.enc_mask_scale = enc_mask_scale
        self.enc_mask_aspect = enc_mask_aspect
        self.pred_mask_scale = pred_mask_scale
        self.pred_mask_aspect = pred_mask_aspect
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.allow_overlap = allow_overlap
        self.min_keep = min_keep

    def _rand_block(self, scale_range: tuple, aspect_range: tuple,
                    acceptable: "torch.Tensor | None" = None):
        """Sample a random rectangular block on the patch grid; return a flat
        LongTensor of patch indices. If ``acceptable`` is given, constrain the
        block to mostly-acceptable patches (falls back after 10 retries)."""
        G = self.grid
        for _ in range(10):
            scale = torch.empty(1).uniform_(*scale_range).item()
            aspect = math.exp(
                torch.empty(1).uniform_(
                    math.log(aspect_range[0]), math.log(aspect_range[1])).item())
            area = scale * self.num_patches
            h = max(1, min(G, int(round(math.sqrt(area / aspect)))))
            w = max(1, min(G, int(round(math.sqrt(area * aspect)))))
            h = min(h, G)
            w = min(w, G)

            r = torch.randint(0, G - h + 1, (1,)).item()
            c = torch.randint(0, G - w + 1, (1,)).item()

            rows = torch.arange(r, r + h)
            cols = torch.arange(c, c + w)
            grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
            indices = (grid_r * G + grid_c).reshape(-1)  # flat patch indices

            if acceptable is None:
                return indices
            if acceptable[indices].float().mean().item() > 0.5:
                return indices

        return indices

    def _make_masks_for_sample(self):
        """Return (enc_mask_indices, [pred_mask_indices, ...]) for one image."""
        N = self.num_patches

        pred_indices_list = []
        used = torch.zeros(N, dtype=torch.bool)
        for _ in range(self.num_pred_masks):
            constraint = ~used if not self.allow_overlap else None
            idx = self._rand_block(self.pred_mask_scale, self.pred_mask_aspect,
                                   acceptable=constraint)
            pred_indices_list.append(idx)
            if not self.allow_overlap:
                used[idx] = True

        enc_block = self._rand_block(self.enc_mask_scale, self.enc_mask_aspect)

        target_set = torch.zeros(N, dtype=torch.bool)
        for idx in pred_indices_list:
            target_set[idx] = True

        enc_mask_bool = torch.zeros(N, dtype=torch.bool)
        enc_mask_bool[enc_block] = True
        enc_mask_bool[target_set] = False

        n_ctx = enc_mask_bool.sum().item()
        if n_ctx < self.min_keep:
            enc_mask_bool = ~target_set
            if enc_mask_bool.sum().item() < self.min_keep:
                enc_mask_bool = torch.ones(N, dtype=torch.bool)

        enc_indices = enc_mask_bool.nonzero(as_tuple=False).squeeze(1)
        return enc_indices, pred_indices_list

    def __call__(self, batch):
        """batch: list of (image_tensor, label). Returns (images, labels,
        enc_masks, pred_masks); masks are cropped to the per-batch minimum length
        so all samples stack."""
        images, labels = default_collate(batch)
        B = images.shape[0]

        enc_samples = []
        pred_samples_by_block = [[] for _ in range(self.num_pred_masks)]
        for _ in range(B):
            enc_ids, pred_ids_list = self._make_masks_for_sample()
            enc_samples.append(enc_ids)
            for j, pred_ids in enumerate(pred_ids_list):
                pred_samples_by_block[j].append(pred_ids)

        min_ctx = min(x.numel() for x in enc_samples)
        enc_masks = torch.stack([x[:min_ctx] for x in enc_samples], dim=0)

        pred_masks = []
        for block_samples in pred_samples_by_block:
            min_pred = min(x.numel() for x in block_samples)
            pred_masks.append(
                torch.stack([x[:min_pred] for x in block_samples], dim=0))

        return images, labels, enc_masks, pred_masks
