"""
ResNet-50 backbone + 3-layer MLP projector for Barlow Twins (Step 1).

Strictly follows Zbontar et al. (2021) arXiv:2103.03230 and the official
facebookresearch/barlowtwins repository:

  Backbone  : ResNet-50  (zero_init_residual=True, fc replaced by Identity)
  Projector : Linear(2048, 8192, bias=False) + BN + ReLU
              Linear(8192, 8192, bias=False) + BN + ReLU
              Linear(8192, 8192, bias=False)            [no activation]
  BN-norm   : BN(8192, affine=False) applied to projector output in forward
              before computing cross-correlation matrix

Barlow Twins loss:
  C = Z_a^T Z_b  /  N
  L = sum_i (1 - C_ii)^2  +  lambda * sum_i sum_{j!=i} C_ij^2
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a 1-D view of all off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _health(name: str, x: torch.Tensor) -> str:
    with torch.no_grad():
        finite = torch.isfinite(x)
        bad = int((~finite).sum().item())
        total = x.numel()
        if finite.any():
            max_abs = float(x[finite].abs().max().item())
        else:
            max_abs = float("nan")
    return f"{name}: bad={bad}/{total} max_abs={max_abs:.6g} dtype={x.dtype}"


def _build_projector(in_dim: int, projector_str: str) -> nn.Sequential:
    """
    Build MLP projector from dash-separated size string.
    Example: "8192-8192-8192" with in_dim=2048 builds:
      Linear(2048,8192)+BN+ReLU, Linear(8192,8192)+BN+ReLU, Linear(8192,8192)
    """
    sizes = [in_dim] + [int(s) for s in projector_str.split("-")]
    layers = []
    for i in range(len(sizes) - 2):
        layers += [
            nn.Linear(sizes[i], sizes[i + 1], bias=False),
            nn.BatchNorm1d(sizes[i + 1]),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    return nn.Sequential(*layers)


class BarlowTwinsResNet(nn.Module):
    """
    Barlow Twins with ResNet-50 backbone (Step 1 original settings).

    Follows the official implementation exactly:
      - backbone: ResNet-50 (zero_init_residual=True)
      - projector: 3-layer MLP [8192-8192-8192]
      - self.bn: BN(proj_dim, affine=False) applied to projector output
      - loss via cross-correlation of BN-normalised projections
    """

    def __init__(self, projector: str = "8192-8192-8192", lambd: float = 0.0051):
        super().__init__()
        self.lambd = lambd

        base = tvm.resnet50(weights=None, zero_init_residual=True)
        base.fc = nn.Identity()
        self.backbone = base

        self.projector = _build_projector(2048, projector)

        proj_dim = int(projector.split("-")[-1])
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y1, y2 : (B, 3, H, W) two augmented views
        Returns:
            loss   : scalar Barlow Twins loss
        """
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        batch_size = z1.size(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_batch = batch_size * torch.distributed.get_world_size()
        else:
            global_batch = batch_size

        # The encoder/projector can run under AMP, but the 8192x8192
        # cross-correlation is numerically fragile at the official high LR.
        # Keep the objective identical while forcing the normalization, matrix
        # multiply, all-reduce, and loss reduction to FP32.
        device_type = "cuda" if z1.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            c = self.bn(z1.float()).T @ self.bn(z2.float())
            # Match the official implementation order: scale before all_reduce.
            # With AMP, reducing the unscaled matrix can overflow before division.
            c.div_(global_batch)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(c)

            on_diag  = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss     = on_diag + self.lambd * off_diag
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite Barlow Twins loss in forward; "
                    + "; ".join([
                        _health("z1", z1),
                        _health("z2", z2),
                        _health("c", c),
                        _health("on_diag", on_diag),
                        _health("off_diag", off_diag),
                    ])
                )
        return loss

    def get_encoder(self) -> nn.Module:
        """Return backbone only (dim=2048) for linear probing."""
        return self.backbone


def build_barlow_resnet(
    projector: str = "8192-8192-8192",
    lambd: float = 0.0051,
) -> BarlowTwinsResNet:
    return BarlowTwinsResNet(projector=projector, lambd=lambd)
