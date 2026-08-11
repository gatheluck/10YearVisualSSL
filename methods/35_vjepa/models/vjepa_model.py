"""V-JEPA model construction (Bardes et al., 2024; V-JEPA, arXiv:2404.08471).

The model is the official `facebookresearch/jepa` video ViT + predictor, imported
from the pinned submodule under `third_party/jepa` (never copied) via
`app.vjepa.utils.init_video_model`. Run at `num_frames=1` / `tubelet_size=1` the
backbone is an image ViT (the capture's step-2 unified-comparison setting).

`init_video_model` is imported **lazily** inside the build functions (not at module
import): the upstream package's top level is a generic `src` name shared with other
submodule ports, so importing it only when a model is actually built keeps the
in-process test suite collision-free (the device/ast tests load this module without
triggering the `src` import).

`build_vjepa` returns (encoder, predictor) for training; `build_vjepa_encoder`
returns the encoder alone for linear evaluation (its `num_mask_tokens` only shapes
the discarded predictor, so it is fixed). `encoder.pt` is the EMA target encoder;
there is no separate projection head to exclude.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The pinned facebookresearch/jepa upstream: third_party/jepa at the repo root
# (methods/35_vjepa/models/vjepa_model.py -> parents[3] is the repo root).
_JEPA_SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "jepa"


def _prepare_jepa_path() -> None:
    """Make `src`/`app` resolve to THIS submodule only. Another submodule port also
    exposes a top-level `src`, and `src` is a PEP 420 namespace package that would
    otherwise merge both submodules' `src/` dirs (so a sibling's `src/utils.py`
    module could shadow this one's `src/utils/` package). So: drop any cached
    `src*`/`app*`, remove every other third_party root from sys.path, and put
    third_party/jepa first."""
    for key in [k for k in sys.modules
                if k in ("src", "app") or k.startswith(("src.", "app."))]:
        del sys.modules[key]
    tp = str(_JEPA_SUBMODULE.parent) + os.sep       # <repo>/third_party/
    sys.path[:] = [q for q in sys.path if not q.startswith(tp)]
    sys.path.insert(0, str(_JEPA_SUBMODULE))


def _init_video_model():
    _prepare_jepa_path()
    try:
        from app.vjepa.utils import init_video_model
    except ImportError as e:
        raise ImportError(
            "the facebookresearch/jepa code is required (the V-JEPA ViT lives "
            "there). It is the pinned submodule at third_party/jepa; run `git "
            "submodule update --init third_party/jepa`.") from e
    return init_video_model


def _model_kwargs(m: dict, num_mask_tokens: int, device) -> dict:
    return {"uniform_power": bool(m.get("uniform_power", False)),
            "use_mask_tokens": bool(m.get("use_mask_tokens", True)),
            "num_mask_tokens": int(num_mask_tokens),
            "zero_init_mask_tokens": bool(m.get("zero_init_mask_tokens", True)),
            "device": device,
            "patch_size": int(m["patch_size"]),
            "num_frames": int(m["num_frames"]),
            "tubelet_size": int(m["tubelet_size"]),
            "model_name": str(m["model_name"]),
            "crop_size": int(m["crop_size"]),
            "pred_depth": int(m["pred_depth"]),
            "pred_embed_dim": int(m["pred_embed_dim"]),
            "use_sdpa": bool(m.get("use_sdpa", True))}


def build_vjepa(model_cfg: dict, num_mask_tokens: int, device):
    """The V-JEPA context encoder + predictor (for training)."""
    init_video_model = _init_video_model()
    encoder, predictor = init_video_model(
        **_model_kwargs(model_cfg, num_mask_tokens, device))
    return encoder, predictor


def build_vjepa_encoder(model_cfg: dict, device, num_mask_tokens: int = 2):
    """The V-JEPA encoder alone (for linear eval). num_mask_tokens only shapes the
    predictor, which is discarded here, so it is fixed."""
    init_video_model = _init_video_model()
    encoder, _predictor = init_video_model(
        **_model_kwargs(model_cfg, num_mask_tokens, device))
    return encoder
