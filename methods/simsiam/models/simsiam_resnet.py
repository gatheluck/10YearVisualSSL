"""
SimSiam ResNet-50 encoder for Step 1 (strict original settings).

Strictly follows Chen & He (2020) arXiv:2011.10566:
  Backbone   : ResNet-50 (zero_init_residual=True, weights=None)
  Projector  : 3-layer MLP: [Linear(2048,2048,bias=False)+BN+ReLU] x2 +
               Linear(2048,dim,bias=False) + BN(affine=False)
  Predictor  : 2-layer bottleneck: Linear(dim,pred_dim,bias=False)+BN+ReLU +
               Linear(pred_dim,dim)  [no BN, no activation on output]
  dim        : 2048
  pred_dim   : 512
  Loss       : -0.5 * [cos(p1, sg(z2)) + cos(p2, sg(z1))]

CRITICAL:
  - Stop-gradient is on z (projector output), NOT on p (predictor output).
  - Predictor uses a FIXED learning rate (do not decay with cosine scheduler).
  - No momentum encoder, no negative pairs, no queue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class SimSiamResNet(nn.Module):
    """
    SimSiam with ResNet-50 backbone.

    Components:
      - backbone : ResNet-50 (avgpool output, 2048-dim)
      - projector: 3-layer MLP (bottleneck-style, output dim 2048)
      - predictor: 2-layer bottleneck MLP (512 hidden)
    """

    def __init__(self, dim: int = 2048, pred_dim: int = 512):
        super().__init__()

        # ResNet-50 backbone: everything except the final fc layer
        base = tvm.resnet50(weights=None, zero_init_residual=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)

        # 3-layer projector MLP
        # Matches builder.py from facebookresearch/simsiam exactly:
        #   layer 1: Linear(2048, 2048, bias=False) + BN + ReLU
        #   layer 2: Linear(2048, 2048, bias=False) + BN + ReLU
        #   layer 3: Linear(2048, dim,  bias=False) + BN(affine=False)
        # Note: output BN has affine=False (no learnable scale/shift).
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False),
        )

        # 2-layer predictor MLP (bottleneck)
        # layer 1: Linear(dim, pred_dim, bias=False) + BN + ReLU
        # layer 2: Linear(pred_dim, dim)  — no BN, no activation
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Args:
            x1, x2: two augmented views of the same image, (B, 3, H, W)
        Returns:
            p1, p2: predictor outputs — gradients flow through these
            z1, z2: projector outputs — already .detach()ed (stop-gradient)
        """
        h1 = self.backbone(x1).flatten(1)   # (B, 2048)
        h2 = self.backbone(x2).flatten(1)

        z1 = self.projector(h1)             # (B, dim)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)             # (B, dim)
        p2 = self.predictor(z2)

        # Stop-gradient: applied to z, NOT to p
        return p1, p2, z1.detach(), z2.detach()

    def get_encoder(self) -> nn.Module:
        """Return backbone only for linear probing. Output: (B, 2048, 1, 1)."""
        return self.backbone


def simsiam_loss(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    """
    Negative cosine similarity loss (Eq. 1 in arXiv:2011.10566).

    loss = -0.5 * [cos(p1, sg(z2)) + cos(p2, sg(z1))]

    z1, z2 must already be detached (returned from SimSiamResNet.forward).
    """
    loss = (
        -F.cosine_similarity(p1, z2, dim=1).mean()
        - F.cosine_similarity(p2, z1, dim=1).mean()
    ) * 0.5
    return loss


def build_simsiam_resnet(dim: int = 2048, pred_dim: int = 512) -> SimSiamResNet:
    return SimSiamResNet(dim=dim, pred_dim=pred_dim)
