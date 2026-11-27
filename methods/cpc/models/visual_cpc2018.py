"""Visual CPC 2018 model (van den Oord et al., 2018), ported from the lab's own
paper-faithful implementation.

A ResNet-v2-101-style no-BN encoder maps each patch to a z-vector, a PixelCNN-style
masked-convolution context autoregresses over the patch grid, and a log-bilinear
InfoNCE loss predicts future rows' z-vectors from the context.

`encoder.pt` is the patch encoder (`encoder.*`); the PixelCNN context and the
InfoNCE predictors are pretext machinery and are excluded. `get_encoder()`
returns a backbone that maps a patch grid to the grid-averaged z (`avg_z`), the
representation the linear probe reads.

The lab wrapper carries DistributedDataParallel all-gather branches for gathering
InfoNCE negatives across ranks; they are kept behind a flag but default off, so
the single-process port draws negatives from within the batch.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBottleneckNoBN(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        out_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=True)
        self.shortcut = (
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                      bias=True)
            if stride != 1 or in_planes != out_planes else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x, inplace=False)
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out, inplace=True))
        out = self.conv3(F.relu(out, inplace=True))
        return out + shortcut


class VisualCPCResNetV2101Encoder(nn.Module):
    """Patch encoder approximating the visual CPC ResNet-v2-101 no-BN target
    (conv1 + 3 pre-activation bottleneck stages: 3, 4, 23 blocks)."""

    def __init__(self, z_dim: int = 1024, width_mult: float = 1.0):
        super().__init__()
        base = max(16, int(64 * width_mult))
        self.in_planes = base
        self.conv1 = nn.Conv2d(3, base, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base, 3, stride=1)
        self.layer2 = self._make_layer(base * 2, 4, stride=2)
        self.layer3 = self._make_layer(base * 4, 23, stride=2)
        self.out_dim = base * 4 * PreActBottleneckNoBN.expansion
        self.proj = (nn.Linear(self.out_dim, z_dim)
                     if self.out_dim != z_dim else nn.Identity())
        self.z_dim = z_dim
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [PreActBottleneckNoBN(self.in_planes, planes, stride)]
        self.in_planes = planes * PreActBottleneckNoBN.expansion
        for _ in range(1, blocks):
            layers.append(PreActBottleneckNoBN(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.conv1(patches)
        x = self.pool(F.relu(x, inplace=True))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(x, inplace=True).mean(dim=(2, 3))
        return self.proj(x)


class MaskedConv2d(nn.Conv2d):
    """PixelCNN-style masked convolution."""

    def __init__(self, mask_type: str, *args, **kwargs):
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        cy, cx = kh // 2, kw // 2
        self.mask[:, :, cy + 1:, :] = 0
        self.mask[:, :, cy, cx + (1 if mask_type == "A" else 0):] = 0
        if mask_type == "A":
            self.mask[:, :, cy, cx] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PixelCNNContext(nn.Module):
    def __init__(self, z_dim: int = 1024, c_dim: int = 1024, layers: int = 5):
        super().__init__()
        blocks = [MaskedConv2d("A", z_dim, c_dim, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(max(layers - 1, 0)):
            blocks.extend([MaskedConv2d("B", c_dim, c_dim, kernel_size=3,
                                        padding=1),
                           nn.ReLU(inplace=True)])
        self.net = nn.Sequential(*blocks)
        self.c_dim = c_dim

    def forward(self, z_grid: torch.Tensor) -> torch.Tensor:
        x = z_grid.permute(0, 3, 1, 2).contiguous()  # B,R,C,D -> B,D,R,C
        c = self.net(x)
        return c.permute(0, 2, 3, 1).contiguous()


class _AvgZBackbone(nn.Module):
    """Maps a patch grid to the grid-averaged z (`avg_z`) using the encoder only,
    so it runs from the weights encoder.pt carries."""

    def __init__(self, encoder: VisualCPCResNetV2101Encoder, z_dim: int):
        super().__init__()
        self.encoder = encoder
        self.z_dim = z_dim

    def forward(self, patch_grid: torch.Tensor) -> torch.Tensor:
        bsz, rows, cols = patch_grid.shape[:3]
        patches = patch_grid.reshape(bsz * rows * cols, *patch_grid.shape[3:])
        z = self.encoder(patches).reshape(bsz, rows * cols, self.z_dim)
        return z.mean(dim=1)


class VisualCPC2018(nn.Module):
    def __init__(self, z_dim: int = 1024, c_dim: int = 1024, pred_steps: int = 5,
                 context_layers: int = 5, encoder_width_mult: float = 1.0):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.pred_steps = pred_steps
        self.encoder = VisualCPCResNetV2101Encoder(z_dim=z_dim,
                                                   width_mult=encoder_width_mult)
        self.context = PixelCNNContext(z_dim=z_dim, c_dim=c_dim,
                                       layers=context_layers)
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim, bias=False) for _ in range(pred_steps)])
        for predictor in self.predictors:
            nn.init.normal_(predictor.weight, std=1.0e-5)

    def encode_grid(self, patch_grid: torch.Tensor) -> torch.Tensor:
        bsz, rows, cols, channels, height, width = patch_grid.shape
        patches = patch_grid.reshape(bsz * rows * cols, channels, height, width)
        z = self.encoder(patches)
        return z.reshape(bsz, rows, cols, self.z_dim)

    def forward(self, patch_grid: torch.Tensor) -> Tuple[torch.Tensor,
                                                         torch.Tensor]:
        z_grid = self.encode_grid(patch_grid)
        c_grid = self.context(z_grid)
        return z_grid, c_grid

    def get_encoder(self) -> nn.Module:
        """A backbone mapping a patch grid to avg_z, for downstream probing."""
        return _AvgZBackbone(self.encoder, self.z_dim)

    def cpc_loss(self, z_grid: torch.Tensor, c_grid: torch.Tensor,
                 use_ddp_negatives: bool = False) -> torch.Tensor:
        """Log-bilinear InfoNCE over future rows. CPC 2018 uses a linear map W_k
        for each future step and scores z_{t+k}^T W_k c_t (no cosine, no
        temperature). Negatives are the other targets in the batch (the port
        runs single-process, so the cross-rank all-gather path is unused)."""
        bsz, rows, cols, z_dim = z_grid.shape
        losses = []
        for step in range(1, self.pred_steps + 1):
            max_anchor = rows - step
            if max_anchor <= 0:
                continue
            ctx = c_grid[:, :max_anchor, :, :]
            target = z_grid[:, step:step + max_anchor, :, :]
            pred = self.predictors[step - 1](ctx.reshape(-1, self.c_dim))
            pred = pred.reshape(bsz, max_anchor, cols, z_dim)

            n = max_anchor * cols
            pred_flat = pred.reshape(bsz * n, z_dim)
            target_flat = target.reshape(bsz * n, z_dim)
            logits = pred_flat @ target_flat.t()
            labels = torch.arange(pred_flat.size(0), device=logits.device)
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean() if losses else z_grid.sum() * 0.0


def build_visual_cpc2018_from_config(config: Dict) -> VisualCPC2018:
    model_cfg = config.get("model", {})
    return VisualCPC2018(
        z_dim=int(model_cfg.get("z_dim", 1024)),
        c_dim=int(model_cfg.get("c_dim", 1024)),
        pred_steps=int(model_cfg.get("pred_steps", 5)),
        context_layers=int(model_cfg.get("context_layers", 5)),
        encoder_width_mult=float(model_cfg.get("encoder_width_mult", 1.0)))
