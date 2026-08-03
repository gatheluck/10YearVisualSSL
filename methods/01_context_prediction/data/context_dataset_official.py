"""
Official-style Context Prediction data stream.

This follows the released Doersch et al. deepcontext loader more closely than
the legacy local dataset:

- resize each image to a random target area in [150k, 450k] pixels
- sample 96x96 patches on a jittered grid with a 48px gap
- jitter every sampled patch by +/-7 pixels
- always keep one random color channel and replace the others with noise
- RMS-normalize each patch and scale it by 50

The original Caffe code builds batches from reusable patch grids.  This PyTorch
port emits one directed patch pair per sample so it can be used with standard
DataLoader/DDP while preserving the sampling distribution and labels.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


def _pil_bilinear():
    return getattr(Image, "Resampling", Image).BILINEAR


def _find_images(folder: str) -> List[str]:
    image_paths: List[str] = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if os.path.splitext(file_name)[1] in VALID_EXTENSIONS:
                image_paths.append(os.path.join(root, file_name))
    image_paths.sort()
    if not image_paths:
        raise FileNotFoundError(f"No images found under {folder}")
    return image_paths


def pos_to_label(offset: Tuple[int, int]) -> int:
    """Match deepcontext/train.py::pos2lbl exactly, with Python 3 ints."""
    posx, posy = offset
    if posy == -1:
        return posx + 1
    if posy == 0:
        return (posx + 7) // 2
    if posy == 1:
        return posx + 6
    raise ValueError(f"Invalid context offset: {offset}")


@dataclass(frozen=True)
class GridPair:
    first_xy: Tuple[int, int]
    second_xy: Tuple[int, int]
    label: int


class OfficialContextPredictionDataset(data.Dataset):
    def __init__(
        self,
        image_folder: str,
        patch_size: int = 96,
        patch_gap: int = 48,
        jitter: int = 7,
        resize_target_pixels: Tuple[int, int] = (150_000, 450_000),
        samples_per_epoch: Optional[int] = None,
        mode: str = "train",
        return_paths: bool = False,
    ) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.image_paths = _find_images(image_folder)
        self.patch_size = patch_size
        self.patch_gap = patch_gap
        self.jitter = jitter
        self.resize_target_pixels = resize_target_pixels
        self.samples_per_epoch = samples_per_epoch or len(self.image_paths)
        self.mode = mode
        self.return_paths = return_paths
        self.grid_step = patch_size + patch_gap
        self.min_side = patch_size * 2 + patch_gap + jitter
        self.forward_offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0)]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_image(self, idx: int) -> Tuple[Image.Image, str]:
        path = self.image_paths[idx % len(self.image_paths)]
        img = Image.open(path).convert("RGB")
        return img, path

    def _resize_like_deepcontext(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        lo, hi = self.resize_target_pixels
        if self.mode == "train":
            target_pixels = random.randint(lo, hi)
        else:
            target_pixels = (lo + hi) // 2
        scale = math.sqrt(float(target_pixels) / float(max(width * height, 1)))
        new_w = max(self.min_side, int(round(width * scale)))
        new_h = max(self.min_side, int(round(height * scale)))
        if new_w != width or new_h != height:
            img = img.resize((new_w, new_h), _pil_bilinear())
        return img

    def _sample_grid_pair(self, img: Image.Image) -> GridPair:
        width, height = img.size
        gridstartx = getattr(self, "_last_gridstartx", 0)
        gridstarty = getattr(self, "_last_gridstarty", 0)
        gridszx = int((width + self.patch_gap - gridstartx) / self.grid_step)
        gridszy = int((height + self.patch_gap - gridstarty) / self.grid_step)
        gridszx = max(2, gridszx)
        gridszy = max(2, gridszy)

        candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for y in range(gridszy):
            for x in range(gridszx):
                for dx, dy in self.forward_offsets:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or ny < 0 or nx >= gridszx or ny >= gridszy:
                        continue
                    candidates.append(((x, y), (nx, ny)))

        first, second = random.choice(candidates)
        dx = second[0] - first[0]
        dy = second[1] - first[1]

        if random.random() < 0.5:
            label = pos_to_label((dx, dy))
            return GridPair(first, second, label)

        label = pos_to_label((-dx, -dy))
        return GridPair(second, first, label)

    def _sample_patch(self, img: Image.Image, xy: Tuple[int, int]) -> Image.Image:
        width, height = img.size
        gridstartx = getattr(self, "_last_gridstartx", 0)
        gridstarty = getattr(self, "_last_gridstarty", 0)
        xpix = gridstartx + xy[0] * self.grid_step + random.randint(-self.jitter, self.jitter)
        ypix = gridstarty + xy[1] * self.grid_step + random.randint(-self.jitter, self.jitter)
        xpix = min(max(xpix, 0), width - self.patch_size)
        ypix = min(max(ypix, 0), height - self.patch_size)
        return img.crop((xpix, ypix, xpix + self.patch_size, ypix + self.patch_size))

    def _prep_patch(self, patch: Image.Image) -> torch.Tensor:
        if self.mode == "train" and random.random() < 0.33:
            randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
            patch = patch.resize((randpix, randpix), _pil_bilinear())
            patch = patch.resize((self.patch_size, self.patch_size), _pil_bilinear())

        arr = np.asarray(patch, dtype=np.float32)
        keep = random.randint(0, 2)
        out = np.empty_like(arr, dtype=np.float32)
        for channel in range(3):
            if channel == keep:
                kept = arr[:, :, channel]
                out[:, :, channel] = kept - float(np.mean(kept))
            else:
                out[:, :, channel] = np.random.uniform(
                    0.0, 1.0, size=arr[:, :, channel].shape
                ).astype(np.float32) - 0.5

        rms = float(np.sqrt(np.mean(np.square(out))))
        if rms < 1e-6 or not np.isfinite(rms):
            rms = 1.0
        out = out / rms * 50.0
        out = np.ascontiguousarray(out.transpose(2, 0, 1))
        return torch.from_numpy(out)

    def __getitem__(self, idx: int):
        for retry in range(10):
            try:
                img, path = self._load_image(idx + retry)
                img = self._resize_like_deepcontext(img)
                width, height = img.size
                if width <= self.min_side or height <= self.min_side:
                    continue

                self._last_gridstartx = random.randint(0, self.grid_step - 1)
                self._last_gridstarty = random.randint(0, self.grid_step - 1)
                pair = self._sample_grid_pair(img)
                first = self._prep_patch(self._sample_patch(img, pair.first_xy))
                second = self._prep_patch(self._sample_patch(img, pair.second_xy))
                label = torch.tensor(pair.label, dtype=torch.long)
                if self.return_paths:
                    return first, second, label, path
                return first, second, label
            except Exception:
                continue
        raise RuntimeError(f"Failed to sample a context pair near index {idx}")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_official_context_loader(
    image_folder: str,
    batch_size: int,
    num_workers: int,
    samples_per_epoch: int,
    mode: str = "train",
    sampler=None,
    patch_size: int = 96,
    patch_gap: int = 48,
    jitter: int = 7,
    resize_target_pixels: Tuple[int, int] = (150_000, 450_000),
):
    dataset = OfficialContextPredictionDataset(
        image_folder=image_folder,
        patch_size=patch_size,
        patch_gap=patch_gap,
        jitter=jitter,
        resize_target_pixels=resize_target_pixels,
        samples_per_epoch=samples_per_epoch,
        mode=mode,
    )
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train"),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )
