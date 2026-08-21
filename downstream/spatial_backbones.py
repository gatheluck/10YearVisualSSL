"""Frozen spatial backbones for downstream dense tasks.

A downstream task head reads a spatial feature map, so every backbone this package
builds exposes the same two-symbol interface the capture harness uses:

    backbone.forward_features(x) -> Tensor[B, C, h, w]
    backbone.out_channels: int

The backbone is always frozen (eval, `requires_grad = False`). A real run loads a
method's trained `encoder.pt`; the hermetic smoke leaves `encoder` empty and builds
a random tiny backbone, so CI downloads and trains nothing on the backbone.

For Step 2 every method's backbone is the unified ViT-B/16, so one ViT adapter
serves them all (the capture's `_load_step2_vit`); Step-1's diverse backbones and
CLIP's own `VisionTransformer` (no `patch_embed.proj`) get their own kinds as those
tasks are ported. Only `vit` is implemented in this pilot.
"""

from __future__ import annotations

import torch
import torch.nn as nn

VIT = "vit"
KINDS = (VIT,)


class FrozenViTSpatialBackbone(nn.Module):
    """Wrap a timm ViT so its patch tokens become a spatial [B, C, h, w] map."""

    def __init__(self, vit: nn.Module, patch_size: int, num_prefix_tokens: int,
                 out_channels: int):
        super().__init__()
        self.vit = vit
        self.patch_size = int(patch_size)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.out_channels = int(out_channels)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):        # stays frozen; never trains
        return super().train(False)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)          # [B, prefix + h*w, D]
        if tokens.ndim != 3:
            raise RuntimeError(
                f"expected ViT tokens [B, N, D], got {tuple(tokens.shape)}")
        patches = tokens[:, self.num_prefix_tokens:, :]
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        if patches.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"token grid mismatch: {patches.shape[1]} patch tokens but "
                f"input {tuple(x.shape[-2:])} at patch {self.patch_size} implies "
                f"{grid_h}x{grid_w}")
        b, _, d = patches.shape
        return patches.transpose(1, 2).reshape(b, d, grid_h, grid_w).contiguous()


def _build_vit(spec: dict) -> FrozenViTSpatialBackbone:
    import timm
    arch = spec.get("arch", "vit_base_patch16_224")
    patch_size = int(spec.get("patch_size", 16))
    kwargs = {"pretrained": False, "num_classes": 0,
              "img_size": int(spec["img_size"]), "patch_size": patch_size}
    for key in ("embed_dim", "depth", "num_heads"):
        if key in spec:
            kwargs[key] = int(spec[key])
    vit = timm.create_model(arch, **kwargs)
    encoder = spec.get("encoder") or ""
    if encoder:
        state = torch.load(encoder, map_location="cpu", weights_only=True)
        missing, unexpected = vit.load_state_dict(state, strict=False)
        # The classifier head is dropped (num_classes=0), so head.* is expected to
        # be unexpected; anything else unexpected means the encoder is not this ViT.
        unexpected = [k for k in unexpected if not k.startswith("head.")]
        if unexpected:
            raise RuntimeError(
                f"encoder.pt carries keys this ViT does not have: {unexpected[:5]}")
    return FrozenViTSpatialBackbone(
        vit, patch_size=patch_size,
        num_prefix_tokens=int(getattr(vit, "num_prefix_tokens", 1)),
        out_channels=int(vit.embed_dim))


def build_frozen_backbone(spec: dict, device: "torch.device") -> nn.Module:
    """Build the frozen spatial backbone named by `spec['kind']`."""
    kind = spec.get("kind")
    if kind == VIT:
        model = _build_vit(spec)
    elif kind in ("resnet50", "clip_vit"):
        raise NotImplementedError(
            f"backbone kind {kind!r} is not ported yet; this pilot implements "
            f"{VIT!r} (the unified Step-2 ViT). See docs/DOWNSTREAM.md.")
    else:
        raise ValueError(f"unknown backbone kind {kind!r}; known: {', '.join(KINDS)}")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
