"""The CPC models, in one place. The paper-faithful visual_cpc2018 patch encoder
is the native path; the capture's unified ViT-B/16 Step 2 (CPCViT) is ported
additively. The capture's deprecated local baseline (cpc_resnet) is excluded.

build_cpc_vit is a lazy accessor: it needs timm (imported inside CPCViT), so the
native visual_cpc2018 path never imports timm."""

from __future__ import annotations

from .visual_cpc2018 import VisualCPC2018, build_visual_cpc2018_from_config

__all__ = ["VisualCPC2018", "build_visual_cpc2018_from_config", "CPCViT",
           "build_cpc_vit"]


def __getattr__(name: str):
    if name in ("CPCViT", "build_cpc_vit"):
        from . import vit_cpc
        return getattr(vit_cpc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
