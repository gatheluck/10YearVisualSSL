"""MAE as the ``mae_vit`` frozen spatial backbone for the ARSSL harness.

Item A3:mae of the Step-3 plan (docs/STEP3_PORTING_PLAN.md). Phase A3 wires the
lineage backbones already ported as Step-1&2 methods into the A1 ARSSL harness
(`downstream/arssl.py`) so their Step-3 numbers reproduce here.

**This file lives in the method's own directory on purpose.** The shared
downstream layer (`downstream/spatial_backbones.py`) *discovers* backbone
providers -- it names no method -- and a method that offers a downstream
backbone kind declares it here, next to the model it wraps. The provider
contract is structural: a module-level ``KIND`` string and a ``build(spec)``
function. See `tests/test_no_hard_coded_methods.py` for why shared machinery
must discover rather than name.

**It reuses MAE's own model, it does not re-implement it.** MAE's ``encoder.pt``
is not timm-loadable: its keys are ``enc_blocks.*``/``enc_norm.*`` (not
``blocks.*``/``norm.*``), and its 2-D sincos position embedding is a buffer that
is regenerated at build time and **not** stored in the checkpoint. MAE's
``forward_encoder`` also shuffles patches even at ``mask_ratio=0``. So only MAE's
own non-shuffling ``MAEEncoder`` yields a faithful, correctly ordered patch grid
-- reproducing it against a timm ViT would be a second implementation that could
silently drift. One implementation, invoked (CLAUDE.md).

Two mechanisms keep that reuse safe and faithful:

* **Load the model by file path, under a unique module name.** ``mae_vit.py`` is
  imported with ``importlib`` from this directory -- never by inserting the
  directory onto ``sys.path``. Many methods ship a top-level ``models`` package;
  a bare ``import models`` after a path insert has collided across methods in
  this repo before. A path-keyed unique name cannot collide.
* **Read the grid with a forward hook.** ``MAEEncoder.forward`` assembles
  ``cls + patch + pos``, runs the blocks and applies ``enc_norm`` before pooling.
  A forward hook on ``enc_norm`` captures its output -- the full ``[B, 1+N, D]``
  token sequence in raster order -- so the exact assembly is reused verbatim
  rather than copied here.

``mae_vit`` is a drop-in for the shared downstream backbone schema
(``kind, encoder, arch, img_size, patch_size`` + optional
``embed_dim, depth, num_heads``): the timm-named optional keys map onto MAE's
``enc_embed_dim/enc_depth/enc_num_heads``. The decoder is built but unused for
features, and ``mlp_ratio`` is the standard 4.0.

**One limitation, stated rather than hidden:** MAE's position embedding is fixed
to ``img_size`` (the port does not interpolate it), so ``mae_vit`` serves the
fixed-size dense tasks (ADE20K, NYUv2, SSv2). A variable-size detection input
(COCO) at a different resolution raises a shape error loudly; it is not silently
resized.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

KIND = "mae_vit"

# The encoder-side weights MAE hands over in encoder.pt (this method's
# ENCODER_PREFIXES). enc_pos_embed is a non-persistent sincos buffer, so it is
# regenerated at build time and never appears in the checkpoint.
ENC_PREFIXES = ("patch_embed.", "cls_token", "enc_blocks.", "enc_norm.")

# The MAE model module sits next to this provider, in this method's own dir; the
# path is taken from __file__ so no method name is hard-coded anywhere.
_MAE_VIT_PATH = Path(__file__).resolve().parent / "models" / "mae_vit.py"
_UNIQUE_NAME = "_downstream_mae_vit_model"
_MAE_VIT_MOD = None


def load_mae_vit_module():
    """Import this method's self-contained MAE model module by file path, under a
    unique name so it can never collide with another method's ``models``
    package. Cached after the first load."""
    global _MAE_VIT_MOD
    if _MAE_VIT_MOD is not None:
        return _MAE_VIT_MOD
    if not _MAE_VIT_PATH.is_file():
        raise RuntimeError(
            f"the MAE model module is missing at {_MAE_VIT_PATH}; mae_vit reuses "
            "this method's own model and cannot be built without it")
    spec = importlib.util.spec_from_file_location(_UNIQUE_NAME, _MAE_VIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MAE_VIT_MOD = module
    return module


def _build_mae(spec: dict):
    """Build a MAE at the encoder dimensions the shared backbone schema names.

    The optional timm-named keys (embed_dim/depth/num_heads) drive MAE's encoder;
    the decoder is unused for features, so it is built small. ``mlp_ratio`` is the
    standard 4.0 that produced the released encoders."""
    mod = load_mae_vit_module()
    embed_dim = int(spec.get("embed_dim", 768))
    depth = int(spec.get("depth", 12))
    num_heads = int(spec.get("num_heads", 12))
    return mod.MaskedAutoencoder(
        img_size=int(spec["img_size"]),
        patch_size=int(spec["patch_size"]),
        enc_embed_dim=embed_dim, enc_depth=depth, enc_num_heads=num_heads,
        # unused for the spatial features; kept minimal and valid.
        dec_embed_dim=embed_dim, dec_depth=1, dec_num_heads=num_heads,
        mlp_ratio=4.0)


class FrozenMAESpatialBackbone(nn.Module):
    """Wrap MAE's ``MAEEncoder`` so its patch tokens become a ``[B, C, h, w]`` map.

    The tokens are read with a forward hook on the encoder's final norm, so MAE's
    own forward (assembly + blocks + norm) is reused unchanged."""

    def __init__(self, encoder: nn.Module, patch_size: int, out_channels: int):
        super().__init__()
        self.encoder = encoder
        self.patch_size = int(patch_size)
        self.num_prefix_tokens = 1                          # MAE prepends one CLS
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
        captured: dict = {}
        handle = self.encoder.enc_norm.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("tokens", out))
        try:
            self.encoder(x)
        finally:
            handle.remove()
        tokens = captured.get("tokens")
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError(
                "the MAE encoder did not apply its final norm; the token grid "
                "could not be read")
        patches = tokens[:, self.num_prefix_tokens:, :]
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        if patches.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"token grid mismatch: {patches.shape[1]} patch tokens but input "
                f"{tuple(x.shape[-2:])} at patch {self.patch_size} implies "
                f"{grid_h}x{grid_w} (MAE's position embedding is fixed to "
                "img_size and is not interpolated)")
        b, _, d = patches.shape
        return patches.transpose(1, 2).reshape(b, d, grid_h, grid_w).contiguous()


def build(spec: dict) -> FrozenMAESpatialBackbone:
    """Build a frozen MAE spatial backbone from the backbone ``spec``.

    A real run names an ``encoder`` (MAE's encoder.pt); the hermetic smoke leaves
    it empty and a tiny random MAE is built, so CI downloads and trains nothing.
    A checkpoint that is not this MAE -- alien keys, or missing encoder weights --
    is refused rather than half-loaded."""
    mae = _build_mae(spec)
    encoder_path = spec.get("encoder") or ""
    if encoder_path:
        state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        result = mae.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                "encoder.pt carries keys this MAE does not have: "
                f"{result.unexpected_keys[:5]}")
        absent = [k for k in result.missing_keys if k.startswith(ENC_PREFIXES)]
        if absent:
            raise RuntimeError(
                f"encoder.pt is missing encoder weights: {absent[:5]}. The "
                "decoder is expected to be missing; the encoder is not")
    encoder = mae.get_encoder(pool="cls")
    return FrozenMAESpatialBackbone(
        encoder, patch_size=int(spec["patch_size"]),
        out_channels=int(mae.enc_embed_dim))
