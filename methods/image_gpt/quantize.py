"""Colour quantisation: turn pixels into colour-cluster tokens.

iGPT feeds a transformer discrete colour tokens, so an image is mapped to the
nearest of `n_clusters` colour centres found by k-means. The lab used sklearn's
MiniBatchKMeans; this ports it to a small **deterministic** Lloyd k-means in
numpy, for two measured reasons: the lab's saved clusters are not in the capture
(so exact clusters cannot be reproduced regardless), and keeping the dependency
stack at torch/torchvision/numpy matches the self-contained methods and keeps the
lock light. Colour quantisation is preprocessing, not the SSL representation.

Determinism: the initial centres are drawn with a seeded RandomState and the
iteration is fixed, so the same pixels and seed always give the same clusters --
which is what makes a run reproducible and what the tests pin.
"""

from __future__ import annotations

import numpy as np
import torch


def kmeans(pixels: np.ndarray, n_clusters: int, seed: int,
           iters: int = 25) -> np.ndarray:
    """Deterministic Lloyd k-means over `pixels` ([N, 3]) -> centres
    ([n_clusters, 3], float32)."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError(f"pixels must be [N, 3], got {pixels.shape}")
    n = pixels.shape[0]
    if n < n_clusters:
        raise ValueError(
            f"need at least n_clusters={n_clusters} pixels, got {n}")
    rng = np.random.RandomState(seed)
    # Distinct seeded initial centres.
    centres = pixels[rng.choice(n, size=n_clusters, replace=False)].copy()
    for _ in range(iters):
        # Assign each pixel to its nearest centre.
        d = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new = centres.copy()
        for c in range(n_clusters):
            members = pixels[labels == c]
            if len(members):
                new[c] = members.mean(axis=0)
            # An empty cluster keeps its centre: deterministic, and rare once
            # seeded from real points.
        if np.allclose(new, centres):
            centres = new
            break
        centres = new
    return centres.astype(np.float32)


def quantize_images(images: torch.Tensor, clusters: np.ndarray) -> torch.Tensor:
    """Map `images` ([B, 3, H, W], any range) to nearest-cluster token indices
    ([B, H*W], long, values in [0, n_clusters))."""
    clusters = np.asarray(clusters, dtype=np.float32)
    b, c, h, w = images.shape
    flat = images.detach().permute(0, 2, 3, 1).reshape(-1, c).cpu().numpy()
    d = ((flat[:, None, :] - clusters[None, :, :]) ** 2).sum(axis=2)
    idx = d.argmin(axis=1).reshape(b, h * w)
    return torch.from_numpy(idx).long()
