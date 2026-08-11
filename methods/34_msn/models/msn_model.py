"""MSN model construction (Assran et al., 2022; arXiv:2204.07141).

The ViT is the official `facebookresearch/msn` `deit` VisionTransformer, imported
from the pinned submodule under `third_party/msn` (never copied). `build_msn_model`
replicates the upstream `init_model`: a ViT trunk (patch embed, cls token, blocks,
norm) plus a projection head `fc` (Linear -> BN -> GELU -> Linear -> BN -> GELU ->
Linear), config-driven so a small hermetic CPU smoke can run a tiny ViT (the
upstream `init_model` only builds the fixed deit_* variants at 224px).

`build_msn_backbone` rebuilds the bare trunk for linear evaluation: with the
projection head left as the default Identity, `forward_features` returns the CLS
token at embed_dim (the trunk feature the probe reads). `encoder.pt` is that trunk;
the projection head `fc.*` is training machinery and is excluded.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

# The pinned facebookresearch/msn upstream: third_party/msn at the repo root
# (methods/34_msn/models/msn_model.py -> parents[3] is the repo root).
_MSN_SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "msn"


def _import_deit():
    # Make `src` resolve to THIS submodule only. Another submodule port (35_vjepa)
    # also exposes a top-level `src`, and `src` is a PEP 420 namespace package that
    # would otherwise merge both submodules' `src/` dirs. So drop any cached `src*`,
    # remove every other third_party root from sys.path, and put third_party/msn
    # first.
    for key in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[key]
    tp = str(_MSN_SUBMODULE.parent) + os.sep       # <repo>/third_party/
    sys.path[:] = [q for q in sys.path if not q.startswith(tp)]
    sys.path.insert(0, str(_MSN_SUBMODULE))
    try:
        from src.deit import VisionTransformer
        from src.utils import trunc_normal_
    except ImportError as e:
        raise ImportError(
            "the facebookresearch/msn code is required (the ViT lives there). It "
            "is the pinned submodule at third_party/msn; run `git submodule update "
            "--init third_party/msn`.") from e
    return VisionTransformer, trunc_normal_


def _attach_head(encoder, trunc_normal_, embed_dim: int, use_bn: bool,
                 hidden_dim: int, output_dim: int) -> None:
    """The upstream init_model projection head + init (replicated verbatim)."""
    encoder.fc = None
    fc = OrderedDict([])
    fc["fc1"] = nn.Linear(embed_dim, hidden_dim)
    if use_bn:
        fc["bn1"] = nn.BatchNorm1d(hidden_dim)
    fc["gelu1"] = nn.GELU()
    fc["fc2"] = nn.Linear(hidden_dim, hidden_dim)
    if use_bn:
        fc["bn2"] = nn.BatchNorm1d(hidden_dim)
    fc["gelu2"] = nn.GELU()
    fc["fc3"] = nn.Linear(hidden_dim, output_dim)
    encoder.fc = nn.Sequential(fc)
    for m in encoder.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def build_msn_model(img_size: int = 224, patch_size: int = 16,
                    embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                    mlp_ratio: float = 4.0, use_bn: bool = True,
                    hidden_dim: int = 2048, output_dim: int = 256,
                    drop_path_rate: float = 0.0):
    """The MSN encoder: ViT trunk + projection head (as upstream init_model)."""
    VisionTransformer, trunc_normal_ = _import_deit()
    encoder = VisionTransformer(
        img_size=[int(img_size)], patch_size=int(patch_size),
        embed_dim=int(embed_dim), depth=int(depth), num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=float(drop_path_rate))
    _attach_head(encoder, trunc_normal_, int(embed_dim), bool(use_bn),
                 int(hidden_dim), int(output_dim))
    return encoder


def build_msn_backbone(img_size: int = 224, patch_size: int = 16,
                       embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                       mlp_ratio: float = 4.0, drop_path_rate: float = 0.0):
    """The bare ViT trunk for linear eval: forward_features -> CLS at embed_dim
    (the projection head is left as the default Identity)."""
    VisionTransformer, _ = _import_deit()
    return VisionTransformer(
        img_size=[int(img_size)], patch_size=int(patch_size),
        embed_dim=int(embed_dim), depth=int(depth), num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=float(drop_path_rate))
