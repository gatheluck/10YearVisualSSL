"""I-JEPA as the ``ijepa_vit`` frozen spatial backbone for the ARSSL harness.

Item A3:ijepa of the Step-3 plan (docs/STEP3_PORTING_PLAN.md). Phase A3 wires the
lineage backbones already ported as Step-1&2 methods into the A1 ARSSL harness
(`downstream/arssl.py`) so their Step-3 numbers reproduce here.

**This file lives in the method's own directory on purpose.** The shared
downstream layer (`downstream/spatial_backbones.py`) *discovers* backbone
providers -- it names no method -- and a method that offers a downstream
backbone kind declares it here, next to the model it wraps. The provider
contract is structural: a module-level ``KIND`` string and a ``build(spec)``
function. See `tests/test_no_hard_coded_methods.py` for why shared machinery
must discover rather than name; a sibling ``downstream_backbone.py`` under
another method's directory was the first to follow this pattern.

**It reuses I-JEPA's own model, it does not re-implement it.** I-JEPA ships its
own self-contained ``VisionTransformer`` (torch-only, no timm). Its
``forward(x)`` already returns every patch token in raster order with the final
norm applied and **no masking / no shuffling** (the mask branch runs only when
``mask_ids`` is given), so -- unlike MAE, whose ``forward_encoder`` shuffles even
at ``mask_ratio=0`` -- the encoder forward is reused verbatim, with no hook and no
non-shuffling clone. I-JEPA has **no CLS token** (features are the mean of the
patch tokens), so there is no prefix to drop, and its position embedding is a
stored parameter that ships inside ``encoder.pt``.

Two facts keep the reuse safe and faithful:

* **Load the model by file path, under a unique module name.** ``vision_transformer.py``
  is imported with ``importlib`` from this directory -- never by inserting the
  directory onto ``sys.path``. Many methods ship a top-level ``models`` package;
  a bare ``import models`` after a path insert has collided across methods in
  this repo before. A path-keyed unique name cannot collide.
* **The checkpoint is exact.** ``encoder.pt`` is the target encoder with the
  ``target_encoder.`` prefix stripped at save time (methods/29_ijepa/adapter),
  so its keys are bare ``VisionTransformer`` keys and must match exactly: any
  unexpected key, or any missing weight, is refused rather than half-loaded.

``ijepa_vit`` is a drop-in for the shared downstream backbone schema
(``kind, encoder, arch, img_size, patch_size`` + optional
``embed_dim, depth, num_heads``): the timm-named optional keys map onto I-JEPA's
``embed_dim/depth/num_heads`` directly, and ``mlp_ratio`` is the standard 4.0.

**One limitation, stated rather than hidden:** I-JEPA's position embedding is a
fixed-size parameter tied to ``img_size`` (the port does not interpolate it), so
``ijepa_vit`` serves the fixed-size dense tasks (ADE20K, NYUv2, SSv2). A
different resolution raises a shape error loudly inside the encoder's forward; it
is not silently resized.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

KIND = "ijepa_vit"

# I-JEPA's self-contained ViT sits next to this provider, in this method's own
# dir; the path is taken from __file__ so no method name is hard-coded anywhere.
_VIT_PATH = Path(__file__).resolve().parent / "models" / "vision_transformer.py"
_UNIQUE_NAME = "_downstream_ijepa_vit_model"
_VIT_MOD = None


def load_vit_module():
    """Import this method's self-contained ViT model module by file path, under a
    unique name so it can never collide with another method's ``models``
    package. Cached after the first load."""
    global _VIT_MOD
    if _VIT_MOD is not None:
        return _VIT_MOD
    if not _VIT_PATH.is_file():
        raise RuntimeError(
            f"the I-JEPA model module is missing at {_VIT_PATH}; ijepa_vit reuses "
            "this method's own model and cannot be built without it")
    spec = importlib.util.spec_from_file_location(_UNIQUE_NAME, _VIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _VIT_MOD = module
    return module


def _build_ijepa(spec: dict):
    """Build an I-JEPA ``VisionTransformer`` at the dimensions the shared backbone
    schema names. The optional timm-named keys (embed_dim/depth/num_heads) drive
    the encoder directly; ``mlp_ratio`` is the standard 4.0."""
    mod = load_vit_module()
    embed_dim = int(spec.get("embed_dim", 768))
    depth = int(spec.get("depth", 12))
    num_heads = int(spec.get("num_heads", 12))
    return mod.VisionTransformer(
        img_size=int(spec["img_size"]),
        patch_size=int(spec["patch_size"]),
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=4.0)


class FrozenIJEPASpatialBackbone(nn.Module):
    """Wrap I-JEPA's ``VisionTransformer`` so its patch tokens become a
    ``[B, C, h, w]`` map. The encoder's own forward (patch embed + pos + blocks +
    norm) is reused unchanged -- it returns raster-order tokens with no masking,
    and I-JEPA has no CLS token, so there is nothing to hook or drop."""

    def __init__(self, encoder: nn.Module, patch_size: int, out_channels: int):
        super().__init__()
        self.encoder = encoder
        self.patch_size = int(patch_size)
        self.num_prefix_tokens = 0                          # I-JEPA has no CLS
        self.out_channels = int(out_channels)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):        # stays frozen; never trains
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)                        # [B, N, D], raster order
        if tokens.ndim != 3:
            raise RuntimeError(
                f"expected I-JEPA tokens [B, N, D], got {tuple(tokens.shape)}")
        patches = tokens[:, self.num_prefix_tokens:, :]
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        if patches.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"token grid mismatch: {patches.shape[1]} patch tokens but input "
                f"{tuple(x.shape[-2:])} at patch {self.patch_size} implies "
                f"{grid_h}x{grid_w} (I-JEPA's position embedding is fixed to "
                "img_size and is not interpolated)")
        b, _, d = patches.shape
        return patches.transpose(1, 2).reshape(b, d, grid_h, grid_w).contiguous()


def build(spec: dict) -> FrozenIJEPASpatialBackbone:
    """Build a frozen I-JEPA spatial backbone from the backbone ``spec``.

    A real run names an ``encoder`` (I-JEPA's encoder.pt, the target encoder with
    its prefix stripped); the hermetic smoke leaves it empty and a tiny random
    I-JEPA is built, so CI downloads and trains nothing. A checkpoint that is not
    this encoder -- alien keys, or missing weights -- is refused rather than
    half-loaded."""
    model = _build_ijepa(spec)
    encoder_path = spec.get("encoder") or ""
    if encoder_path:
        state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                "encoder.pt carries keys this I-JEPA encoder does not have: "
                f"{result.unexpected_keys[:5]}")
        if result.missing_keys:
            raise RuntimeError(
                "encoder.pt is missing encoder weights: "
                f"{result.missing_keys[:5]}")
    return FrozenIJEPASpatialBackbone(
        model, patch_size=int(spec["patch_size"]),
        out_channels=int(model.embed_dim))
