"""SeLa Sinkhorn-Knopp optimal-transport label assignment (Asano et al., 2020)."""

from .sinkhorn import sinkhorn_knopp, compute_hard_sinkhorn_assignments

__all__ = ["sinkhorn_knopp", "compute_hard_sinkhorn_assignments"]
