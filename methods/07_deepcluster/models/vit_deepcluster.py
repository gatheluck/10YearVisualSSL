"""ViT Step-2 DeepCluster model: a unified ViT-B/16 + a reset-each-epoch head.

Faithful to the capture's `models/vit_deepcluster.py`: a timm `VisionTransformer`
(from scratch) maps the image to its CLS token, and a `Linear(embed_dim, k)`
`top_layer` produces the logits the network learns to predict from the k-means
pseudo-labels. As in DeepCluster, the `top_layer` is **reset each epoch** (unlike
SeLa, where it is trained continuously). This port's convention: the ViT trunk
lives under ``self.backbone`` (num_classes=0), so `encoder.pt` keeps only
``backbone.*`` (the reset-each-epoch head is training machinery, excluded) and
``get_features()`` returns the CLS feature -- both for the per-epoch clustering
and for the linear probe (the eval sizes its head to it dynamically).
``get_features`` takes the AlexNet path's ``before_final_relu`` flag and ignores
it (a ViT CLS token has no final ReLU), so the shared
`extract_features_for_clustering` reuses it unchanged. timm is imported lazily
(only on `arch: vit`); the ViT dimensions are configurable so a tiny model can
run a smoke.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViTDeepCluster(nn.Module):
    def __init__(self, num_classes: int = 1000, image_size: int = 224,
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
        self.feature_dim = embed_dim
        self.top_layer = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.top_layer.weight, 0, 0.01)
        nn.init.zeros_(self.top_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if self.top_layer is not None:
            x = self.top_layer(x)
        return x

    def get_features(self, x: torch.Tensor,
                     before_final_relu: bool = False) -> torch.Tensor:
        """The CLS backbone feature, no gradient -- for clustering and probing.

        ``before_final_relu`` is accepted for signature parity with the AlexNet
        path (so `extract_features_for_clustering` is shared) and ignored: a ViT
        CLS token has no final ReLU to take the features before.
        """
        with torch.no_grad():
            return self.backbone(x)

    def reset_top_layer(self, num_classes: int, device: "torch.device",
                        seed: "int | None" = None):
        """Reinitialise the top_layer weights in place (same ``seed`` -> same
        weights). k is fixed across epochs, so the head keeps its shape."""
        assert num_classes == self.top_layer.out_features, (
            f"num_classes {num_classes} != top_layer.out_features "
            f"{self.top_layer.out_features}")
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.normal_(self.top_layer.weight.data, 0, 0.01)
        nn.init.constant_(self.top_layer.bias.data, 0)


def build_vit_deepcluster(num_classes: int = 1000, image_size: int = 224,
                          patch_size: int = 16, embed_dim: int = 768,
                          depth: int = 12, num_heads: int = 12,
                          mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                          attn_drop_rate: float = 0.0) -> ViTDeepCluster:
    return ViTDeepCluster(num_classes=num_classes, image_size=image_size,
                          patch_size=patch_size, embed_dim=embed_dim,
                          depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
