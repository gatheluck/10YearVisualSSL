"""The jigsaw++ pretext dataset and its permutation generator, in one place."""

from __future__ import annotations

from .jigsaw_pp_dataset import (JigsawPPDataset,
                                generate_jigsaw_pp_permutations)

__all__ = ["JigsawPPDataset", "generate_jigsaw_pp_permutations"]
