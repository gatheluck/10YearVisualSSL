"""The CPC model, in one place. Only the paper-faithful visual_cpc2018 step-1
model is brought across; the capture's deprecated local baseline (cpc_resnet) and
its ViT (step 2) are excluded from this port."""

from __future__ import annotations

from .visual_cpc2018 import VisualCPC2018, build_visual_cpc2018_from_config

__all__ = ["VisualCPC2018", "build_visual_cpc2018_from_config"]
