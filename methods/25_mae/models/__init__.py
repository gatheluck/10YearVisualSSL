"""The MAE model, built in one place so the trainer, the evaluator and the
adapter construct the same architecture from the same settings."""

from __future__ import annotations

from .mae_vit import (MaskedAutoencoder, MAEEncoder, mae_vit_base_patch16,
                      mae_vit_large_patch16, mae_vit_huge_patch14)

# The recipe variants, chosen by `arch` in the config.
VARIANTS = {
    "vit_base_patch16": mae_vit_base_patch16,
    "vit_large_patch16": mae_vit_large_patch16,
    "vit_huge_patch14": mae_vit_huge_patch14,
}


def build_mae(arch: str, **overrides) -> MaskedAutoencoder:
    """Build an MAE by variant name, with any architecture overrides (a tiny one
    for the hermetic smoke)."""
    if arch not in VARIANTS:
        raise ValueError(
            f"unknown MAE arch {arch!r}; known: {', '.join(sorted(VARIANTS))}")
    return VARIANTS[arch](**overrides)


__all__ = ["MaskedAutoencoder", "MAEEncoder", "build_mae", "VARIANTS",
           "mae_vit_base_patch16", "mae_vit_large_patch16",
           "mae_vit_huge_patch14"]
