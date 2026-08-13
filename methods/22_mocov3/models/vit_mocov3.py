"""MoCo v3 with the official PyTorch ViT (Chen et al., 2021; arXiv:2104.02057),
ported from the lab's own implementation (following facebookresearch/moco-v3).

- VisionTransformerMoCo with fixed 2D sin-cos positional embeddings, subclassing
  timm's VisionTransformer (this is why timm is a pretrain dependency).
- Base + momentum encoders (the momentum encoder is an EMA copy, no gradient);
  each ViT's ``head`` is replaced by a 3-layer MLP projector; a 2-layer MLP
  predictor sits on the base encoder. A symmetric InfoNCE loss.
- The ViT is built from scratch (no pretrained weights) -- the run is hermetic.

`encoder.pt` is the base ViT trunk (`base_encoder.*` minus the projector
`base_encoder.head.*`); the projector, predictor and momentum encoder are
excluded. `get_backbone()` returns the CLS feature (embed_dim) for the probe.

The port threads ``img_size`` (the capture hard-coded timm's 224 default) so the
same code runs a small hermetic CPU smoke; the DDP all-gather is kept but inert
single-process.
"""

from __future__ import annotations

import math
from functools import partial, reduce
from operator import mul
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, _cfg

try:
    from timm.layers import PatchEmbed
except ImportError:  # older timm
    from timm.models.layers import PatchEmbed


def _num_prefix_tokens(model: VisionTransformer) -> int:
    return int(getattr(model, "num_tokens",
                       getattr(model, "num_prefix_tokens", 1)))


class VisionTransformerMoCo(VisionTransformer):
    """Official MoCo v3 ViT wrapper (fixed 2D sin-cos pos-embed, official init)."""

    def __init__(self, stop_grad_conv1: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.build_2d_sincos_position_embedding()

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "qkv" in name:
                    val = math.sqrt(
                        6.0 / float(module.weight.shape[0] // 3
                                    + module.weight.shape[1]))
                    nn.init.uniform_(module.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1)
                            + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)
            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                if self.patch_embed.proj.bias is not None:
                    self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature: float = 10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert self.embed_dim % 4 == 0, \
            "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w),
             torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        assert _num_prefix_tokens(self) == 1, "Assuming one and only one token, [cls]"
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


_ARCH_BUILDERS = {
    "vit_small": vit_small,
    "vit_small_patch16_224": vit_small,
    "vit_base": vit_base,
    "vit_base_patch16_224": vit_base,
}


def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    """Official MoCo v3 MLP helper."""
    mlp = []
    for layer_idx in range(num_layers):
        in_dim = input_dim if layer_idx == 0 else mlp_dim
        out_dim = output_dim if layer_idx == num_layers - 1 else mlp_dim
        mlp.append(nn.Linear(in_dim, out_dim, bias=False))
        if layer_idx < num_layers - 1:
            mlp.append(nn.BatchNorm1d(out_dim))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            mlp.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*mlp)


def build_vit_backbone(arch: str, head_dim: int, stop_grad_conv1: bool = False,
                       img_size: int = 224) -> VisionTransformerMoCo:
    """Build an official MoCo v3 ViT with a temporary linear head.

    ``img_size`` is threaded (the capture hard-coded timm's 224 default) so a
    small hermetic CPU smoke can run at a lower resolution."""
    if arch not in _ARCH_BUILDERS:
        raise ValueError(f"Unsupported MoCo v3 ViT arch: {arch!r}")
    return _ARCH_BUILDERS[arch](num_classes=head_dim,
                                stop_grad_conv1=stop_grad_conv1,
                                img_size=img_size)


def replace_vit_head_with_projector(model, proj_dim, mlp_dim):
    hidden_dim = model.head.weight.shape[1]
    del model.head
    model.head = _build_mlp(3, hidden_dim, mlp_dim, proj_dim, last_bn=True)


class ViTFeatureExtractor(nn.Module):
    """The frozen CLS features before the projector head (for the probe)."""

    def __init__(self, vit: VisionTransformerMoCo):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit.forward_features(x)
        if hasattr(self.vit, "forward_head"):
            return self.vit.forward_head(features, pre_logits=True)
        if features.ndim == 3:
            return features[:, 0]
        return features


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class MoCoV3(nn.Module):
    """Official-layout MoCo v3 model for ViT backbones."""

    def __init__(self, arch: str = "vit_base", proj_dim: int = 256,
                 mlp_dim: int = 4096, temperature: float = 0.2,
                 momentum: float = 0.99, stop_grad_conv1: bool = True,
                 img_size: int = 224):
        super().__init__()
        self.T = temperature
        self.m = momentum

        self.base_encoder = build_vit_backbone(arch, head_dim=mlp_dim,
                                               stop_grad_conv1=stop_grad_conv1,
                                               img_size=img_size)
        self.momentum_encoder = build_vit_backbone(arch, head_dim=mlp_dim,
                                                   stop_grad_conv1=stop_grad_conv1,
                                                   img_size=img_size)
        replace_vit_head_with_projector(self.base_encoder, proj_dim, mlp_dim)
        replace_vit_head_with_projector(self.momentum_encoder, proj_dim, mlp_dim)

        self.predictor = _build_mlp(2, proj_dim, mlp_dim, proj_dim, last_bn=True)

        for param_q, param_k in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self, momentum: Optional[float] = None):
        m = self.m if momentum is None else momentum
        for param_q, param_k in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    def _contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        k_all = concat_all_gather(k)
        logits = torch.einsum("nc,mc->nm", [q, k_all]) / self.T
        batch_size = logits.shape[0]
        labels = (torch.arange(batch_size, device=q.device, dtype=torch.long)
                  + batch_size * _get_rank())
        return F.cross_entropy(logits, labels) * (2.0 * self.T)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                momentum: Optional[float] = None) -> torch.Tensor:
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        with torch.no_grad():
            self._momentum_update(momentum)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
        return self._contrastive_loss(q1, k2) + self._contrastive_loss(q2, k1)

    def get_backbone(self) -> nn.Module:
        return ViTFeatureExtractor(self.base_encoder)


def build_mocov3_vit(arch: str = "vit_base", proj_dim: int = 256,
                     mlp_dim: int = 4096, temperature: float = 0.2,
                     momentum: float = 0.99, stop_grad_conv1: bool = True,
                     img_size: int = 224) -> MoCoV3:
    return MoCoV3(arch=arch, proj_dim=proj_dim, mlp_dim=mlp_dim,
                  temperature=temperature, momentum=momentum,
                  stop_grad_conv1=stop_grad_conv1, img_size=img_size)
