"""
LARS optimizer for Barlow Twins (Step 1).

Faithfully reproduced from the official Barlow Twins repository
(facebookresearch/barlowtwins) with minor documentation additions.

Key design:
  - Weights (ndim > 1) : LARS adaptation + weight decay
  - Biases/BN  (ndim 1): plain SGD, no LARS, no weight decay
    -> controlled via weight_decay_filter / lars_adaptation_filter

Reference: Zbontar et al. (2021) arXiv:2103.03230
           You et al.   (2017) arXiv:1708.03888  (original LARS paper)
"""

import torch
import torch.optim as optim


def exclude_bias_and_norm(p: torch.Tensor) -> bool:
    """Return True for 1-D parameters (biases and BatchNorm/LayerNorm scalars)."""
    return p.ndim == 1


class LARS(optim.Optimizer):
    """
    LARS (Layer-wise Adaptive Rate Scaling) optimizer.

    Implements the LARS update rule:
        v_{t+1} = momentum * v_t + lr * (eta * ||w|| / ||g|| * g + wd * w)
        w_{t+1} = w_t - v_{t+1}

    Parameters
    ----------
    params : iterable
        Iterable of parameters or parameter groups.
    lr : float
        Base learning rate (typically set to 0 and adjusted per-step externally).
    weight_decay : float
        L2 regularisation coefficient applied before LARS scaling.
    momentum : float
        SGD momentum coefficient.
    eta : float
        LARS trust ratio (typically 0.001).
    weight_decay_filter : callable, optional
        If provided and returns True for a parameter, weight decay is skipped.
    lars_adaptation_filter : callable, optional
        If provided and returns True for a parameter, LARS scaling is skipped
        (plain SGD used instead).
    """

    def __init__(
        self,
        params,
        lr: float = 0.0,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue

                # Apply weight decay (skipped for bias / BN params)
                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                # LARS trust-ratio scaling (skipped for bias / BN params)
                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm  = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0.0,
                            g["eta"] * param_norm / update_norm,
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                # SGD momentum update
                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])
