"""Inspected image/patch neighbor-set agreement, without feature extraction.

The source calls the per-row intersection of two k-NN sets "mutual agreement".
It does not first turn either directed neighborhood into reciprocal edges.
"""
from __future__ import annotations

import numpy as np
import torch


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _matrix(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError('features must be a nonempty tensor matrix')
    if value.is_complex():
        raise ValueError('features must be real-valued')
    value = value.detach().to(dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError('features must be finite')
    return value


@torch.no_grad()
def nearest_neighbors(features, *, k=10, image_ids=None, chunk_size=1024, search_k=None, pre_normalized=False):
    """Cosine neighbors, excluding self or every token of the query image.

    Uses the source's torch.topk tie behavior on the selected device. Exact
    ties are not promised to select the same indices across devices/backends.
    search_k=20 reproduces the image pool's top-20 followed by first-ten slice.
    """
    _positive_int(k, 'k')
    _positive_int(chunk_size, 'chunk_size')
    search_k = k if search_k is None else search_k
    _positive_int(search_k, 'search_k')
    x = _matrix(features)
    n = x.shape[0]
    if not k <= search_k < n:
        raise ValueError('require k <= search_k < number of rows')
    norm = x.norm(dim=1, keepdim=True)
    if not torch.isfinite(norm).all() or (norm == 0).any():
        raise ValueError('feature norms must be finite and positive')
    if type(pre_normalized) is not bool:
        raise ValueError('pre_normalized must be boolean')
    if pre_normalized:
        if not torch.allclose(norm, torch.ones_like(norm), atol=1e-5, rtol=0):
            raise ValueError('pre-normalized vectors must have unit norms')
    else:
        x = x / norm
    ids = torch.arange(n, device=x.device) if image_ids is None else torch.as_tensor(image_ids, device=x.device)
    if ids.shape != (n,) or ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('image IDs must be one integer per feature row')
    _, counts = ids.unique(return_counts=True)
    if n - int(counts.max()) < search_k:
        raise ValueError('not enough neighbors outside each image')
    indices = torch.empty((n, k), dtype=torch.int64, device=x.device)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        similarity = x[start:stop] @ x.T
        similarity[ids[start:stop, None] == ids[None, :]] = -float('inf')
        indices[start:stop] = similarity.topk(search_k, dim=1).indices[:, :k]
    return indices.sort(dim=1).values


def _indices(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError('neighbors must be a nonempty matrix')
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('neighbor indices must be integers')
    n, k = value.shape
    if k >= n or (value < 0).any() or (value >= n).any():
        raise ValueError('neighbor index outside row population')
    ordered = value.sort(dim=1).values
    if k > 1 and (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError('duplicate neighbor in a row')
    return ordered


@torch.no_grad()
def agreement_matrix(neighborhoods):
    """Mean per-query |A intersect B| / k, returned as float64 NumPy matrix."""
    if not neighborhoods:
        raise ValueError('at least one neighborhood is required')
    sets = [_indices(value) for value in neighborhoods]
    if any(value.shape != sets[0].shape or value.device != sets[0].device for value in sets):
        raise ValueError('neighbor populations, k and devices must match')
    result = np.eye(len(sets), dtype=np.float64)
    k = sets[0].shape[1]
    for i, left in enumerate(sets):
        for j in range(i + 1, len(sets)):
            intersection = (left.unsqueeze(2) == sets[j].unsqueeze(1)).any(2).sum(1)
            result[i, j] = result[j, i] = int(intersection.sum()) / (left.shape[0] * k)
    return result


@torch.no_grad()
def patch_vectors(maps, *, grid=7, center_positions=False):
    """Flatten already-pooled NCHW grids in image,row,column order.

    Centering subtracts the per-position mean of normalized vectors, then
    normalizes again, matching the source's position-control analysis.
    """
    _positive_int(grid, 'grid')
    if not isinstance(maps, torch.Tensor) or maps.ndim != 4 or min(maps.shape) < 1 or maps.shape[2:] != (grid, grid):
        raise ValueError('expected nonempty N,C,grid,grid patch maps')
    n, channels = maps.shape[:2]
    x = _matrix(maps.permute(0, 2, 3, 1).reshape(n * grid * grid, channels))
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if center_positions:
        x = x.reshape(n, grid * grid, channels)
        x = (x - x.mean(0, keepdim=True)).reshape(n * grid * grid, channels)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x, torch.arange(n, device=x.device).repeat_interleave(grid * grid)


def position_shares(indices, *, grid=7):
    """Fraction of neighbors sharing grid position, row or column."""
    _positive_int(grid, 'grid')
    indices = _indices(indices)
    n = indices.shape[0]
    if n % (grid * grid):
        raise ValueError('patch population must contain complete image grids')
    positions = torch.arange(n, device=indices.device) % (grid * grid)
    query, neighbor = positions[:, None], positions[indices.long()]
    return {
        'position': float((neighbor == query).float().mean()),
        'row': float((neighbor // grid == query // grid).float().mean()),
        'column': float((neighbor % grid == query % grid).float().mean()),
    }


def patch_rows(*, total=10000, count=2000, image_set='a'):
    """Source subset seeds: sorted seed-2 set A, seed-3 disjoint set B."""
    _positive_int(total, 'total')
    _positive_int(count, 'count')
    if image_set not in ('a', 'b') or count > total or (image_set == 'b' and 2 * count > total):
        raise ValueError('invalid patch subset specification')
    first = np.sort(np.random.default_rng(2).choice(total, count, replace=False))
    if image_set == 'a':
        return first
    remaining = np.setdiff1d(np.arange(total), first)
    return np.sort(np.random.default_rng(3).choice(remaining, count, replace=False))
