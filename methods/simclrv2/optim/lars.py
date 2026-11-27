"""LARS (Layer-wise Adaptive Rate Scaling) optimizer (You et al., 2017), as used by SimCLR v2, ported
from the lab's own implementation.

SimCLR v2 uses LARS for large-batch (4096) training. Bias and 1-D normalization
parameters are excluded from LARS scaling and updated with plain SGD (scaled only
by the global learning rate). torch-only.
"""

from __future__ import annotations

import torch


class LARS(torch.optim.Optimizer):
    """LARS optimizer with momentum.

    Args:
        params:                model parameters
        lr:                    global learning rate (after any schedule)
        momentum:              SGD momentum (default 0.9)
        weight_decay:          L2 regularisation coefficient (default 1e-6)
        eta:                   LARS trust coefficient (default 0.001)
        exclude_bias_and_norm: skip LARS for 1-D params (bias / BN) (default True)
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-6,
        eta: float = 0.001,
        exclude_bias_and_norm: bool = True,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            eta=eta, exclude_bias_and_norm=exclude_bias_and_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            momentum     = group["momentum"]
            weight_decay = group["weight_decay"]
            eta          = group["eta"]
            exclude      = group["exclude_bias_and_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.clone()

                # 1-D parameters (bias, BN weight/bias): plain SGD, no LARS scaling
                is_1d = (p.ndim == 1)
                if exclude and is_1d:
                    effective_lr = lr
                    dp = grad
                else:
                    p_norm = p.data.norm(2).item()
                    g_norm = grad.norm(2).item()
                    if p_norm > 0 and g_norm > 0:
                        # LARS trust ratio: eta * ||w|| / (||grad|| + wd*||w||)
                        adaptive = eta * p_norm / (g_norm + weight_decay * p_norm + 1e-12)
                        effective_lr = lr * adaptive
                    else:
                        effective_lr = lr
                    dp = grad + weight_decay * p.data

                state = self.state[p]
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(p)
                v = state["velocity"]
                v.mul_(momentum).add_(dp, alpha=effective_lr)
                p.data.add_(-v)

        return loss
