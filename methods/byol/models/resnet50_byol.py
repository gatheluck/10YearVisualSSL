"""BYOL with a ResNet-50 backbone (Grill et al., 2020; arXiv:2006.07733), ported
from the lab's own paper-faithful implementation.

  Online network: ResNet-50 -> Projection MLP (2048->4096->256)
                            -> Prediction MLP  (256->4096->256)
  Target network: ResNet-50 -> Projection MLP (2048->4096->256)
                  [EMA of the online backbone + projector; no predictor, no grad]

The loss is a symmetric negative cosine similarity between the online prediction
of one view and the (stop-gradient) target projection of the other -- **no
negatives, no queue**. The EMA momentum tau follows a cosine schedule from 0.996
to 1.0. Optimised with LARS.

`encoder.pt` is the online ResNet-50 backbone (`online_encoder.*`); the projector,
predictor and target network are training machinery and are excluded. `encode()`
returns the 2048-d backbone feature for the linear probe. torch/torchvision only;
the capture's ViT variant (`vit_byol.py`, which needs `timm`) is a separate step
and is not ported.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


class ProjectionMLP(nn.Module):
    """Two-layer projection MLP (BYOL Appendix A): Linear-BN-ReLU -> Linear."""

    def __init__(self, input_dim=2048, hidden_dim=4096, output_dim=256):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class PredictionMLP(nn.Module):
    """Two-layer prediction MLP (online network only)."""

    def __init__(self, input_dim=256, hidden_dim=4096, output_dim=256):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class BYOLResNet50(nn.Module):
    """BYOL wrapping online and target networks with a ResNet-50 backbone."""

    def __init__(self, encoder_dim=2048, proj_hidden_dim=4096,
                 proj_output_dim=256, pred_hidden_dim=4096,
                 pred_output_dim=256):
        super().__init__()
        resnet = models.resnet50(weights=None)
        # Strip the final FC; a 2048-d vector after global average pooling.
        self.online_encoder = nn.Sequential(*list(resnet.children())[:-1],
                                            nn.Flatten())
        self.online_projector = ProjectionMLP(encoder_dim, proj_hidden_dim,
                                              proj_output_dim)
        self.predictor = PredictionMLP(proj_output_dim, pred_hidden_dim,
                                       pred_output_dim)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_network(self, tau):
        """EMA: theta_target = tau * theta_target + (1 - tau) * theta_online."""
        for op, tp in zip(self.online_encoder.parameters(),
                          self.target_encoder.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)
        for op, tp in zip(self.online_projector.parameters(),
                          self.target_projector.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    def forward_online(self, x):
        return self.predictor(self.online_projector(self.online_encoder(x)))

    @torch.no_grad()
    def forward_target(self, x):
        return self.target_projector(self.target_encoder(x))

    def forward(self, x1, x2):
        """Returns (p1, p2, z1, z2): online predictions and target projections."""
        p1 = self.forward_online(x1)
        p2 = self.forward_online(x2)
        z1 = self.forward_target(x1)
        z2 = self.forward_target(x2)
        return p1, p2, z1, z2

    def encode(self, x):
        """The online backbone feature (2048-d), for linear evaluation."""
        return self.online_encoder(x)


class BYOLLoss(nn.Module):
    """Symmetric negative cosine similarity: the target is stop-gradient."""

    def forward(self, p1, p2, z1, z2):
        p1 = F.normalize(p1, dim=-1, p=2)
        p2 = F.normalize(p2, dim=-1, p=2)
        z1 = F.normalize(z1.detach(), dim=-1, p=2)
        z2 = F.normalize(z2.detach(), dim=-1, p=2)
        return -0.5 * ((p1 * z2).sum(dim=-1).mean()
                       + (p2 * z1).sum(dim=-1).mean())


class LARS(optim.Optimizer):
    """LARS optimizer (You et al., 2017). Weight tensors (ndim > 1) get the
    trust-ratio update; bias / BN params (ndim <= 1) get plain SGD (no weight
    decay, no trust scaling)."""

    def __init__(self, params, lr=0.2, momentum=0.9, weight_decay=1.5e-6,
                 trust_coefficient=0.001, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            tc = group["trust_coefficient"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                dp = p.grad
                if p.ndim > 1:
                    if wd != 0.0:
                        dp = dp.add(p, alpha=wd)
                    p_norm = p.norm(2)
                    dp_norm = dp.norm(2)
                    if p_norm > 0 and dp_norm > 0:
                        local_lr = lr * tc * p_norm / (dp_norm + eps)
                    else:
                        local_lr = lr
                else:
                    local_lr = lr
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(dp, alpha=local_lr)
                p.data.sub_(buf)
        return loss


def build_byol_resnet50(config):
    """Build BYOLResNet50 from a model config dict."""
    return BYOLResNet50(
        encoder_dim=config.get("encoder_dim", 2048),
        proj_hidden_dim=config.get("proj_hidden_dim", 4096),
        proj_output_dim=config.get("proj_output_dim", 256),
        pred_hidden_dim=config.get("pred_hidden_dim", 4096),
        pred_output_dim=config.get("pred_output_dim", 256))


def build_lars_optimizer(model, config):
    """Build LARS with linear lr scaling from a config dict."""
    lr = config["training"]["learning_rate"]
    if config["training"].get("lr_scale_by_batch", False):
        base_batch = config["training"].get("lr_scale_base", 256)
        lr = lr * config["training"]["batch_size"] / base_batch
    return LARS(params=model.parameters(), lr=lr,
                momentum=config["training"].get("momentum", 0.9),
                weight_decay=config["training"].get("weight_decay", 1.5e-6),
                trust_coefficient=config["training"].get("trust_coefficient",
                                                         0.001))


def compute_ema_tau(epoch, total_epochs, tau_base=0.996, tau_final=1.0):
    """Cosine schedule for the EMA momentum, from tau_base (t=0) to tau_final."""
    progress = epoch / total_epochs
    return tau_final - (tau_final - tau_base) * (math.cos(math.pi * progress)
                                                 + 1) / 2


class LinearClassifier(nn.Module):
    """Linear probe classifier for evaluation."""

    def __init__(self, input_dim=2048, num_classes=1000):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
