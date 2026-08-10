"""LeJEPA model: a timm ViT backbone + a projection MLP (Balestriero & LeCun,
2025; arXiv:2511.08544).

Ported from the lab's own LeJEPA code. Each image is seen as several augmented
views; the backbone encodes every view to a feature, the projection MLP maps it to
the space where SIGReg (see ``sigreg.py``) and the cross-view invariance loss act.
The backbone is the trained representation: ``encoder.pt`` is the backbone alone
(the projector is training machinery and is excluded), read back through
``LeJEPABackbone`` for linear probing (one num_features vector per image).

The ViT is timm's, built from scratch (``pretrained=False``), so the run stays
hermetic. ``img_size`` and the ViT dims are threaded so a small hermetic CPU smoke
can run a tiny ViT at a lower resolution.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_backbone(model_name: str, img_size: int, drop_path_rate: float = 0.0,
                   pretrained: bool = False) -> nn.Module:
    """A timm feature backbone (num_classes=0). ``img_size`` is passed when the
    model accepts it (ViTs do); models that do not are built at their default."""
    kwargs = {"pretrained": bool(pretrained), "num_classes": 0,
              "drop_path_rate": float(drop_path_rate)}
    try:
        return timm.create_model(model_name, img_size=int(img_size), **kwargs)
    except TypeError:
        return timm.create_model(model_name, **kwargs)


def _encode(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """One feature vector per image, however the backbone returns its output."""
    feats = backbone(images)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 3:
        feats = feats[:, 0]                     # a token grid -> the CLS token
    elif feats.ndim > 2:
        feats = feats.flatten(2).mean(dim=-1)   # a feature map -> global pool
    return feats


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int,
                 final_bn: bool):
        super().__init__()
        layers = max(2, int(layers))
        modules = []
        dim = in_dim
        for _ in range(layers - 1):
            modules.extend([nn.Linear(dim, hidden_dim, bias=False),
                            nn.BatchNorm1d(hidden_dim), nn.GELU()])
            dim = hidden_dim
        modules.append(nn.Linear(dim, out_dim, bias=not final_bn))
        if final_bn:
            modules.append(nn.BatchNorm1d(out_dim, affine=False))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeJEPABackbone(nn.Module):
    """The frozen backbone as a linear-probe feature extractor: one feature vector
    per image (num_features)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(getattr(backbone, "num_features", 0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _encode(self.backbone, images)


class LeJEPAEncoder(nn.Module):
    """Backbone + projection MLP. ``forward`` takes views [N,V,C,H,W] and returns
    (features [N,V,D], proj [V,N,proj_dim]) -- the shapes SIGReg and the
    invariance loss consume."""

    def __init__(self, model_name: str, img_size: int = 224,
                 drop_path_rate: float = 0.0, proj_hidden_dim: int = 2048,
                 proj_dim: int = 512, proj_layers: int = 3, final_bn: bool = True,
                 pretrained: bool = False):
        super().__init__()
        self.backbone = build_backbone(model_name, img_size, drop_path_rate,
                                       pretrained)
        self.feature_dim = int(getattr(self.backbone, "num_features", 0))
        if self.feature_dim <= 0:
            raise RuntimeError(
                f"Could not infer feature dimension for timm model {model_name}")
        self.projector = ProjectionMLP(
            in_dim=self.feature_dim, hidden_dim=proj_hidden_dim, out_dim=proj_dim,
            layers=proj_layers, final_bn=final_bn)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return _encode(self.backbone, images)

    def forward(self, views: torch.Tensor):
        if views.ndim != 5:
            raise ValueError(
                f"Expected views as [N,V,C,H,W], got shape {tuple(views.shape)}")
        n, v = views.shape[:2]
        features = self.encode(views.flatten(0, 1))
        proj = self.projector(features)
        features = features.reshape(n, v, -1)
        proj = proj.reshape(n, v, -1).transpose(0, 1).contiguous()
        return features, proj

    def get_encoder(self) -> "LeJEPABackbone":
        return LeJEPABackbone(self.backbone)


def build_lejepa(model_name: str = "vit_base_patch16_224", img_size: int = 224,
                 drop_path_rate: float = 0.0, proj_hidden_dim: int = 2048,
                 proj_dim: int = 512, proj_layers: int = 3, final_bn: bool = True,
                 pretrained: bool = False) -> LeJEPAEncoder:
    return LeJEPAEncoder(model_name=model_name, img_size=img_size,
                         drop_path_rate=drop_path_rate,
                         proj_hidden_dim=proj_hidden_dim, proj_dim=proj_dim,
                         proj_layers=proj_layers, final_bn=final_bn,
                         pretrained=pretrained)
