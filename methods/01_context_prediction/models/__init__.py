"""Models for Context Prediction, official-style track.

Rewritten during the port: the captured original also re-exported
`alexnet_context` and a legacy `vit_context`, which were not brought across.
The unified ViT-B/16 Step-2 model is added as `vit_context.py`, imported lazily
(only on the arch: vit path). See `../provenance.json` and docs/STEP2_VIT_PORTING.md.
"""

from .alexnet_context_official import (      # noqa: F401
    OfficialContextPredictionAlexNet,
    build_official_context_alexnet,
)

__all__ = ["OfficialContextPredictionAlexNet", "build_official_context_alexnet"]
