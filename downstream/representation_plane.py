"""Rank-based task directions and held-family-out evaluation.

Inputs are explicit observations, not manuscript result claims. Ranks are global
within each task's selected population, including during held-family-out fits.
"""

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np


def _vector(values, name):
    try:
        if np.iscomplexobj(values):
            raise ValueError(f"{name} must be real-valued")
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric vector") from error
    if vector.ndim != 1 or not vector.size:
        raise ValueError(f"{name} must be a nonempty vector")
    return vector


def percentile_rank(values):
    """Average one-based ranks divided by population size; ties stay tied."""
    vector = _vector(values, "rank input")
    if not np.isfinite(vector).all():
        raise ValueError("rank input must be finite")
    _, inverse, counts = np.unique(vector, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average = ends - (counts - 1) / 2
    return average[inverse] / len(vector)


def _correlation(left, right):
    a = percentile_rank(left)
    b = percentile_rank(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        raise ValueError("rank correlation requires nonconstant observations")
    return float(np.clip(a @ b / denominator, -1, 1))


def _fit(design, target):
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("full and held-family fits require identifiable coordinates")
    return coefficients


def _angle(coefficients):
    if np.linalg.norm(coefficients[1:]) < 1e-12:
        raise ValueError("constant fitted task has no direction")
    return float(np.degrees(np.arctan2(coefficients[2], coefficients[1])))


def fit_task(x, y, scores, families, *, lower_is_better=False, valid=None):
    """Fit a task on re-ranked selected rows and predict every held-out family.

    Missing scores require an explicit boolean ``valid`` mask. Coordinates and
    target are ranked after selection, once for the whole task population, not
    separately inside folds. Lower-is-better scores are negated before ranking.
    Singular fits are rejected instead of reporting an arbitrary direction.
    """
    x = _vector(x, "x")
    y = _vector(y, "y")
    scores = _vector(scores, "scores")
    if type(lower_is_better) is not bool:
        raise ValueError("lower_is_better must be an explicit boolean")
    if len(y) != len(x) or len(scores) != len(x):
        raise ValueError("coordinates and task scores must have equal lengths")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("coordinates must be finite")
    labels = list(families)
    if len(labels) != len(x) or any(not isinstance(v, str) or not v.strip() for v in labels):
        raise ValueError("each observation requires a nonempty family label")
    keep = np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid)
    if keep.dtype != np.bool_ or keep.shape != x.shape or not keep.any():
        raise ValueError("valid must be a nonempty aligned boolean selection")
    indices = np.flatnonzero(keep)
    target = percentile_rank(-scores[keep] if lower_is_better else scores[keep])
    if np.ptp(target) == 0:
        raise ValueError("task scores must be nonconstant")
    coordinates = np.column_stack([percentile_rank(x[keep]), percentile_rank(y[keep])])
    design = np.column_stack([np.ones(len(indices)), coordinates])
    selected_families = np.asarray(labels)[keep]
    groups = sorted(set(selected_families))
    if len(groups) < 2:
        raise ValueError("held-family evaluation requires at least two families")
    coefficients = _fit(design, target)
    predictions = np.empty(len(indices))
    predictions_x = np.empty(len(indices))
    folds = []
    for label in groups:
        held = selected_families == label
        train = ~held
        fold = _fit(design[train], target[train])
        predictions[held] = design[held] @ fold
        predictions_x[held] = design[held, :2] @ _fit(design[train, :2], target[train])
        folds.append({"family": str(label), "held_out": int(held.sum()),
                      "coefficients": fold.tolist(), "theta": _angle(fold)})
    angles = [f["theta"] for f in folds]
    return {
        "n": len(indices), "indices": indices.tolist(),
        "coordinates": coordinates.tolist(), "target": target.tolist(),
        "coefficients": coefficients.tolist(), "theta": _angle(coefficients),
        "folds": folds, "tmin": min(angles), "tmax": max(angles),
        "predictions_x": predictions_x.tolist(), "predictions_xy": predictions.tolist(),
        "rho_x": _correlation(predictions_x, target),
        "rho_xy": _correlation(predictions, target),
    }


def rotate_directions(tasks, *, rotation_degrees=0.0):
    """Return explicitly rotated display angles, never alter fits or coordinates.

    An explicit common offset reproduces the source's optional arrow-only
    rotation. The default is zero so display and fitted-coordinate angles agree.
    Fold ranges are arithmetic extrema, not circular confidence intervals.
    """
    if (isinstance(rotation_degrees, bool) or not isinstance(rotation_degrees, Real)
            or not math.isfinite(rotation_degrees)):
        raise ValueError("rotation must be an explicit finite angle")
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("at least one fitted task is required")
    output = {}
    for name, task in tasks.items():
        try:
            theta, low, high = (float(task[k]) for k in ("theta", "tmin", "tmax"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each task requires fitted theta and fold extrema") from error
        if not all(math.isfinite(v) for v in (theta, low, high)) or low > high:
            raise ValueError("task angles and ordered fold extrema must be finite")
        output[name] = {"theta": theta + rotation_degrees,
                        "fold_range": [low + rotation_degrees, high + rotation_degrees]}
    return {"display_rotation_deg": float(rotation_degrees), "tasks": output}
