"""AIM as the ``aim_vit`` frozen spatial backbone for the ARSSL harness.

Item A3:aim of the Step-3 plan (docs/STEP3_PORTING_PLAN.md). Phase A3 wires the
lineage backbones already ported as Step-1&2 methods into the A1 ARSSL harness
(`downstream/arssl.py`) so their Step-3 numbers reproduce here.

**This file lives in the method's own directory on purpose.** The shared
downstream layer (`downstream/spatial_backbones.py`) *discovers* backbone
providers -- it names no method -- and a method that offers a downstream backbone
kind declares it here, next to the model it wraps. The provider contract is
structural: a module-level ``KIND`` string and a ``build(spec)`` function. It
reuses AIM's own model, loaded **by file path under a unique module name** (never
via ``sys.path``), so it cannot collide with another method's top-level ``models``
package -- the cross-method import bug this repo has hit before.

**Why AIM needs a provider, not a config** (measured, not assumed): AIM's encoder
is a hand-written, from-scratch, non-timm ViT -- its patch embedding is a
``nn.Linear`` over unfolded pixels (not a Conv2d), all linears are bias-free, there
is no CLS token, and the position embedding is a sincos buffer. Its ``encoder.pt``
keys therefore cannot load into the shared ``vit`` (timm) kind the way LeJEPA's do.

**How the map is read.** AIM's ``forward(x, prefix_len)`` returns a training tuple
``(loss, pred, target)`` under a prefix-LM causal mask -- not tokens. Clean tokens
come only from ``AIMViT.forward_features(x, layer_ids)``, which runs the trunk
**bidirectionally** and averages the chosen layers. The method's own linear probe
reads the last ``num_feature_layers`` blocks averaged, then patch-mean-pools
(`methods/30_aim/evaluate_linear_aim.py`, the ``unified`` recipe). This provider
reproduces exactly that read and reshapes the per-position tokens to a
``[B, C, h, w]`` map, so global-average-pooling the map equals AIM's own probe
feature (one representation, two readers).

**Two things the shared ViT backbone schema has no slot for, absorbed here so the
four task runners stay unchanged** (the pattern iGPT established):

* **num_feature_layers** -- the size of the last-N layer window AIM averages. The
  backbone schema rejects unknown keys, so it is fixed to AIM's protocol value 6
  (`methods/30_aim/configs/linear_eval_vit.yaml`). It is clamped to the model depth
  exactly as the probe clamps it, so a shallower model (the hermetic smoke) reads
  all its layers rather than indexing off the end.
* **the prediction head** -- ``encoder.pt`` excludes ``predictor.*`` (the method's
  adapter drops it), and the head is never used to read features. A minimal head is
  built (it is discarded by the strict load, which tolerates its missing keys), so
  the wrapper carries only the trunk. A real trunk weight missing, or any alien
  key, is refused rather than half-loaded.

The optional timm-named schema keys map onto AIM's architecture
(``embed_dim``, ``depth``, ``num_heads``); ``mlp_ratio`` is the standard 4.0.
Like I-JEPA, AIM's position embedding is fixed to ``img_size`` and is not
interpolated, so ``aim_vit`` serves the fixed-size dense tasks; a different
resolution raises loudly inside ``forward_features`` rather than being resized.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

KIND = "aim_vit"

# AIM's protocol: average the last 6 transformer blocks for evaluation features
# (methods/30_aim/configs/linear_eval_vit.yaml, num_feature_layers). The backbone
# schema has no slot for this, so it is absorbed here (the four runners stay
# unchanged); it is clamped to the model depth, as the probe clamps it.
DEFAULT_NUM_FEATURE_LAYERS = 6

# The prediction head is excluded from encoder.pt and never used to read features.
_PREDICTOR_PREFIX = "predictor."

# AIM's self-contained ViT sits next to this provider, in this method's own dir;
# the path is taken from __file__ so no method name is hard-coded anywhere.
_AIM_PATH = Path(__file__).resolve().parent / "models" / "aim_vit.py"
_UNIQUE_NAME = "_downstream_aim_vit_model"
_AIM_MOD = None


def load_aim_module():
    """Import this method's self-contained AIM model module by file path, under a
    unique name so it can never collide with another method's ``models`` package.
    Cached after the first load."""
    global _AIM_MOD
    if _AIM_MOD is not None:
        return _AIM_MOD
    if not _AIM_PATH.is_file():
        raise RuntimeError(
            f"the AIM model module is missing at {_AIM_PATH}; aim_vit reuses this "
            "method's own model and cannot be built without it")
    spec = importlib.util.spec_from_file_location(_UNIQUE_NAME, _AIM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AIM_MOD = module
    return module


def _build_aim(spec: dict):
    """Build an ``AIMViT`` at the dimensions the shared backbone schema names. The
    optional timm-named keys (embed_dim/depth/num_heads) drive the trunk directly;
    ``mlp_ratio`` is the standard 4.0. The prediction head is built minimally: it
    is excluded from ``encoder.pt`` and unused for features, so its size is
    irrelevant and a full head would only waste memory."""
    mod = load_aim_module()
    embed_dim = int(spec.get("embed_dim", 768))
    depth = int(spec.get("depth", 12))
    num_heads = int(spec.get("num_heads", 12))
    return mod.AIMViT(
        img_size=int(spec["img_size"]),
        patch_size=int(spec["patch_size"]),
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.0,
        head_depth=1, head_dim=embed_dim)


class FrozenAIMSpatialBackbone(nn.Module):
    """Wrap AIM's ``AIMViT`` so its evaluation tokens become a ``[B, C, h, w]`` map.

    ``forward_features`` reads the last ``num_feature_layers`` blocks averaged (the
    trunk run bidirectionally, AIM's own ``forward_features``), then reshapes the
    raster-order tokens to the grid. AIM has no CLS token, so there is nothing to
    drop; global-average-pooling the map equals AIM's own probe feature."""

    def __init__(self, model: nn.Module, patch_size: int, out_channels: int,
                 num_feature_layers: int):
        super().__init__()
        self.model = model
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.num_feature_layers = int(num_feature_layers)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):        # stays frozen; never trains
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        n_blocks = len(self.model.blocks)
        # Clamp the last-N window to the model depth, exactly as the probe does, so
        # a shallow model reads all its layers rather than indexing off the end.
        k = min(self.num_feature_layers, n_blocks)
        layer_ids = list(range(n_blocks - k, n_blocks))
        tokens = self.model.forward_features(x, layer_ids=layer_ids)  # [B, N, D]
        if tokens.ndim != 3:
            raise RuntimeError(
                f"expected AIM tokens [B, N, D], got {tuple(tokens.shape)}")
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        if tokens.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"token grid mismatch: {tokens.shape[1]} patch tokens but input "
                f"{tuple(x.shape[-2:])} at patch {self.patch_size} implies "
                f"{grid_h}x{grid_w} (AIM's position embedding is fixed to img_size "
                "and is not interpolated)")
        b, _, d = tokens.shape
        return tokens.transpose(1, 2).reshape(b, d, grid_h, grid_w).contiguous()


def build(spec: dict) -> FrozenAIMSpatialBackbone:
    """Build a frozen AIM spatial backbone from the backbone ``spec``.

    A real run names an ``encoder`` (AIM's encoder.pt, the trunk with the prediction
    head excluded); the hermetic smoke leaves it empty and a tiny random AIM is
    built, so CI downloads and trains nothing. A checkpoint that is not this encoder
    -- alien keys, or a missing trunk weight -- is refused rather than half-loaded;
    the head's ``predictor.*`` keys are expected to be missing and are tolerated."""
    model = _build_aim(spec)
    encoder_path = spec.get("encoder") or ""
    if encoder_path:
        state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                "encoder.pt carries keys this AIM encoder does not have: "
                f"{result.unexpected_keys[:5]}")
        # The prediction head is excluded from encoder.pt, so predictor.* is
        # expected to be missing; any other missing weight means the checkpoint is
        # not this trunk (or the config's dimensions disagree with it).
        absent = [k for k in result.missing_keys
                  if not k.startswith(_PREDICTOR_PREFIX)]
        if absent:
            raise RuntimeError(
                f"encoder.pt is missing trunk weights: {absent[:5]}. The head "
                "is expected to be missing; the representation is not")
    return FrozenAIMSpatialBackbone(
        model, patch_size=int(spec["patch_size"]),
        out_channels=int(model.embed_dim),
        num_feature_layers=DEFAULT_NUM_FEATURE_LAYERS)
