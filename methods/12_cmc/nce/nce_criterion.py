"""NCE and InfoNCE loss functions for CMC (Tian et al., 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-7


class NCECriterion(nn.Module):
    """Noise-Contrastive Estimation loss (Eq. 12 in the CMC paper).

    The first column of ``x`` is the positive score; the remaining K columns are
    the noise (negative) scores.
    """

    def __init__(self, n_data: int):
        super().__init__()
        self.n_data = n_data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        m = x.size(1) - 1        # number of noise samples K
        Pn = 1.0 / self.n_data   # uniform noise distribution

        P_pos = x[:, 0]
        log_D1 = torch.log(P_pos / (P_pos + m * Pn + EPS))

        P_neg = x[:, 1:]
        log_D0 = torch.log(m * Pn / (P_neg + m * Pn + EPS))

        loss = -(log_D1.sum() + log_D0.sum()) / bsz
        return loss


class NCESoftmaxLoss(nn.Module):
    """InfoNCE / softmax cross-entropy contrastive loss.

    The positive logit must be in column 0; the remaining columns are negatives.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        label = torch.zeros(bsz, dtype=torch.long, device=x.device)
        return self.criterion(x, label)
