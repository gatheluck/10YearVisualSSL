"""SIGReg: the Gaussian regularizer at the heart of LeJEPA (arXiv:2511.08544).

Ported verbatim from the lab's own LeJEPA code. SIGReg pushes the distribution of
the projected features toward an isotropic Gaussian, measured by an Epps-Pulley
statistic: the empirical characteristic function of random 1-D slices of the batch
is compared against the standard-normal characteristic function ``exp(-t^2/2)`` on
a trapezoidal quadrature grid. The random slice directions are drawn from a
per-step seeded generator, so a run is reproducible.

The lab wrapper averages the characteristic function across DDP ranks
(``differentiable_mean_across_ranks`` + an ``all_reduce`` of the step counter);
this single-process port keeps the same arithmetic with a world size of one, so
the statistic is identical to a one-rank distributed run. The recipe is the
minimal Epps-Pulley quadrature and random-slicing scheme, not an imported package.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def differentiable_mean_across_ranks(x: torch.Tensor) -> torch.Tensor:
    """The mean across DDP ranks. Single-process: the local value is the mean."""
    return x


class SIGReg(nn.Module):
    def __init__(self, t_max: float, knots: int, num_slices: int, seed: int):
        super().__init__()
        if knots < 3:
            raise ValueError("SIGReg requires at least 3 quadrature knots")
        t = torch.linspace(0, float(t_max), int(knots), dtype=torch.float32)
        dt = float(t_max) / max(int(knots) - 1, 1)
        weights = torch.full((int(knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt                       # trapezoidal end weights
        phi = torch.exp(-0.5 * t.square())          # standard-normal char. fn.
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.num_slices = int(num_slices)
        self.seed = int(seed)

    def _projection_matrix(self, dim: int, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed + int(self.step.item()))
            a = torch.randn((dim, self.num_slices), device=device,
                            dtype=torch.float32, generator=generator)
            a = F.normalize(a, dim=0, eps=1e-12)
            self.step.add_(1)
        return a

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        if proj.ndim != 3:
            raise ValueError(
                f"Expected projected features [V,N,D] or [N,D], got "
                f"{tuple(proj.shape)}")
        local_n = proj.size(-2)
        world = 1
        x = proj.float()
        a = self._projection_matrix(x.size(-1), x.device)
        sliced = x @ a
        xt = sliced.unsqueeze(-1) * self.t
        cos_mean = differentiable_mean_across_ranks(xt.cos().mean(dim=-3))
        sin_mean = differentiable_mean_across_ranks(xt.sin().mean(dim=-3))
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * local_n * world
        return statistic.mean()
