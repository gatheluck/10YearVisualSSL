"""313-bin ab quantisation for colorization (Zhang et al., 2016).

The 313 in-gamut ab-bin centres (`pts_in_hull.npy`) are the paper's constant,
**vendored** alongside this file (source: richzhang/colorization, see
provenance.json) -- there is no runtime download. Quantisation is pure numpy: a
one-time 221x221 lookup table over the ab grid gives O(H*W) nearest-bin lookup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PTS_FILE = Path(__file__).resolve().parent / "pts_in_hull.npy"

_LUT_A_MIN, _LUT_A_MAX = -110.0, 110.0
_LUT_B_MIN, _LUT_B_MAX = -110.0, 110.0

_PTS: "np.ndarray | None" = None
_AB_LUT: "np.ndarray | None" = None


def get_ab_points() -> np.ndarray:
    """The vendored [313, 2] ab-bin centres. Errors loudly if absent -- the port
    is hermetic and never downloads."""
    global _PTS
    if _PTS is None:
        if not _PTS_FILE.is_file():
            raise FileNotFoundError(
                f"{_PTS_FILE} is missing; the 313 ab-bin constant must be "
                "vendored (source: richzhang/colorization, see provenance.json)")
        _PTS = np.load(_PTS_FILE).astype(np.float64)
    return _PTS


def quantize_ab_to_bins(ab_values: np.ndarray,
                        pts_in_hull: "np.ndarray | None" = None) -> np.ndarray:
    """Nearest-bin quantisation by brute force [H, W] -> [H, W] indices."""
    if pts_in_hull is None:
        pts_in_hull = get_ab_points()
    H, W = ab_values.shape[:2]
    ab_flat = ab_values.reshape(-1, 2)
    distances = np.sum(
        (ab_flat[:, np.newaxis, :] - pts_in_hull[np.newaxis, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).reshape(H, W)


def _build_ab_lut() -> np.ndarray:
    pts = get_ab_points()
    a_size = int(round(_LUT_A_MAX - _LUT_A_MIN)) + 1  # 221
    b_size = int(round(_LUT_B_MAX - _LUT_B_MIN)) + 1  # 221
    a_vals = np.linspace(_LUT_A_MIN, _LUT_A_MAX, a_size)
    b_vals = np.linspace(_LUT_B_MIN, _LUT_B_MAX, b_size)
    aa, bb = np.meshgrid(a_vals, b_vals)
    ab_grid = np.stack([aa.ravel(), bb.ravel()], axis=1)
    dist2 = np.sum((ab_grid[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2,
                   axis=2)
    return np.argmin(dist2, axis=1).reshape(b_size, a_size).astype(np.int16)


def get_ab_lut() -> np.ndarray:
    global _AB_LUT
    if _AB_LUT is None:
        _AB_LUT = _build_ab_lut()
    return _AB_LUT


def quantize_ab_fast(ab_values: np.ndarray) -> np.ndarray:
    """Fast O(H*W) ab quantisation via the precomputed lookup table."""
    lut = get_ab_lut()
    a_size, b_size = lut.shape[1], lut.shape[0]
    a_idx = np.clip(np.round(
        (ab_values[..., 0] - _LUT_A_MIN) / (_LUT_A_MAX - _LUT_A_MIN)
        * (a_size - 1)).astype(np.int32), 0, a_size - 1)
    b_idx = np.clip(np.round(
        (ab_values[..., 1] - _LUT_B_MIN) / (_LUT_B_MAX - _LUT_B_MIN)
        * (b_size - 1)).astype(np.int32), 0, b_size - 1)
    return lut[b_idx, a_idx].astype(np.int64)


def bins_to_ab_values(bin_indices: np.ndarray,
                      pts_in_hull: "np.ndarray | None" = None) -> np.ndarray:
    """Map bin indices [H, W] back to ab values [H, W, 2]."""
    if pts_in_hull is None:
        pts_in_hull = get_ab_points()
    H, W = bin_indices.shape
    return pts_in_hull[bin_indices.flatten()].reshape(H, W, 2)
