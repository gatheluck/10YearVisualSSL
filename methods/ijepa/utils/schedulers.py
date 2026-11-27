"""Cosine schedulers for I-JEPA (LR, weight-decay, EMA).

Ported verbatim from the lab's own code. All schedulers return a pre-computed
value array indexed by training step (matching the original I-JEPA, which uses
``ipe_scale`` to extend the cosine period beyond the actual training horizon).
"""

from __future__ import annotations

import numpy as np


def cosine_scheduler(base_value: float, final_value: float, total_steps: int,
                     warmup_steps: int = 0,
                     start_warmup_value: float = 0.0) -> np.ndarray:
    """A length-``total_steps`` array: linear warm-up from start_warmup_value ->
    base_value over [0, warmup_steps), then cosine decay base_value ->
    final_value over the rest."""
    schedule = np.ones(total_steps, dtype=np.float64) * final_value

    cosine_steps = total_steps - warmup_steps
    if cosine_steps > 0:
        iters = np.arange(cosine_steps)
        schedule[warmup_steps:] = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / cosine_steps))

    if warmup_steps > 0:
        schedule[:warmup_steps] = np.linspace(start_warmup_value, base_value,
                                              warmup_steps)

    return schedule.astype(np.float32)


def ema_scheduler(base_value: float, final_value: float,
                  total_steps: int) -> np.ndarray:
    """Cosine EMA momentum schedule (base -> final, increasing; ~0.996 -> 1.0)."""
    iters = np.arange(total_steps)
    schedule = final_value - (final_value - base_value) * (
        0.5 * (1 + np.cos(np.pi * iters / total_steps)))
    return schedule.astype(np.float32)
