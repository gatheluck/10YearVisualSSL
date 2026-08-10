"""
Gram Anchoring Loss (DINOv3).

Reference: DINOv3 (Siméoni et al., 2025), Section 4.

Gram anchoring prevents the degradation of dense (patch-level) feature quality
during long training schedules of large SSL models.

The loss matches the Gram matrix (pairwise cosine similarities between patches)
of the student to that of an early-stage "Gram teacher" model.

    L_Gram = || X_S X_S^T - X_G X_G^T ||_F^2

where:
    X_S : (P, D) L2-normalized patch features from the student (global crop).
    X_G : (P, D) L2-normalized patch features from the Gram teacher (same crop,
          possibly at 2x resolution then downsampled for finer consistency).

The Gram teacher is frozen at an early checkpoint (e.g., 200k iterations for
the ViT-7B case) and updated every 10k steps.

The canonical unified Step2 protocol snapshots a separate teacher after epoch
250, applies this loss during epochs 251-300, and sparsely refreshes that
teacher. Both student and clean teacher crops remain at the unified 224 global
resolution; high-resolution adaptation is outside the Step2 definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """
    Gram Matrix Anchoring Loss.

    The caller applies the configured loss weight, matching the other local
    objective components.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_patches: torch.Tensor,
        gram_teacher_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_patches      : (B, P, D) student patch features (last layer, after norm).
            gram_teacher_patches : (B, P, D) Gram teacher patch features (detached, frozen).

        Returns:
            loss: Frobenius-norm difference of Gram matrices, averaged over batch.
        """
        # L2-normalize patch features along feature dimension
        xs = F.normalize(student_patches, dim=-1)            # (B, P, D)
        xg = gram_teacher_patches.detach()
        xg = F.normalize(xg, dim=-1)                         # (B, P, D)

        # Gram matrices: (B, P, P)
        gram_s = torch.bmm(xs, xs.transpose(1, 2))           # (B, P, P)
        gram_g = torch.bmm(xg, xg.transpose(1, 2))           # (B, P, P)

        return self.mse(gram_s, gram_g)
