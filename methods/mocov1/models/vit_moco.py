"""ViT Step-2 MoCo v1 model: a unified ViT-B/16 query/key encoder + a momentum
queue.

Faithful to the capture's `models/vit_moco.py`: a timm `VisionTransformer` (from
scratch) reads the image, its CLS token feeds a single `Linear(768, feature_dim)`
projection (NO MLP -- that is v2), L2-normalised. A momentum key encoder (an EMA
copy, no gradient) and a FIFO queue of K past keys give the InfoNCE negatives;
``forward`` returns ``(loss, logits, labels)`` exactly as the ResNet path does.
Restructured to this port's convention: the ViT trunk lives under
``encoder_q.backbone`` (num_classes=0), so `encoder.pt` keeps only
``encoder_q.backbone.*`` and ``get_encoder()`` returns that trunk (its CLS
feature) for the linear probe. timm is imported lazily (only on `arch: vit`), and
the ViT dimensions are configurable so a tiny model can run a CPU smoke.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTEncoder(nn.Module):
    """ViT trunk (CLS token) with a single linear projection (MoCo v1 style)."""

    def __init__(self, feature_dim: int = 128, image_size: int = 224,
                 patch_size: int = 16, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        self.backbone = VisionTransformer(
            img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            num_classes=0, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            qkv_bias=True, norm_layer=nn.LayerNorm)
        self.proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)         # (B, embed_dim) -- CLS token
        feat = self.proj(feat)          # (B, feature_dim)
        return F.normalize(feat, dim=1)


class MoCoViT(nn.Module):
    """MoCo v1 with a ViT-B/16 encoder: momentum encoder, FIFO queue, InfoNCE."""

    def __init__(self, feature_dim: int = 128, queue_size: int = 65536,
                 momentum: float = 0.999, temperature: float = 0.07,
                 image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0) -> None:
        super().__init__()
        self.K = queue_size
        self.m = momentum
        self.T = temperature

        vit_kwargs = dict(feature_dim=feature_dim, image_size=image_size,
                          patch_size=patch_size, embed_dim=embed_dim,
                          depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.encoder_q = ViTEncoder(**vit_kwargs)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update(self) -> None:
        for pq, pk in zip(self.encoder_q.parameters(),
                          self.encoder_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, (
            f"Queue size ({self.K}) must be divisible by batch size "
            f"({batch_size}).")
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        q = self.encoder_q(im_q)
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(im_k)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=logits.device)
        loss = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(k)
        return loss, logits, labels

    def get_encoder(self) -> nn.Module:
        """The query ViT trunk (CLS feature), for the linear probe."""
        return self.encoder_q.backbone


def build_moco_vit(feature_dim: int = 128, queue_size: int = 65536,
                   momentum: float = 0.999, temperature: float = 0.07,
                   image_size: int = 224, patch_size: int = 16,
                   embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                   mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                   attn_drop_rate: float = 0.0) -> MoCoViT:
    return MoCoViT(feature_dim=feature_dim, queue_size=queue_size,
                   momentum=momentum, temperature=temperature,
                   image_size=image_size, patch_size=patch_size,
                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                   mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                   attn_drop_rate=attn_drop_rate)
