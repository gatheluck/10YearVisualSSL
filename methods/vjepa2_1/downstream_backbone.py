"""Pinned V-JEPA 2.1 image/video encoder with explicit differentiable access.

Geometry follows the captured Basic5 wrapper: pad to a patch boundary, use
T=1 for images, preserve native spatiotemporal video tokens. Inputs must already
be normalized by the task. This is not the separate native-384 extraction
profile and does not establish complete protocol or score reproduction.
"""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from provider_support import prepare_upstream

KIND = "vjepa2_1"
UPSTREAM = "https://github.com/facebookresearch/vjepa2"
TRAINABLE = True
CAPTURE_PYRAMID = True
COMPONENT_ONLY = True
_VARIANTS = {"vit_large": "ema_encoder", "vit_giant_xformers": "target_encoder"}


def _encoder(spec):
    arch = spec["arch"]
    if arch not in _VARIANTS:
        raise ValueError(f"unsupported encoder architecture {arch!r}")
    if int(spec["patch_size"]) != 16 or int(spec["img_size"]) != 384:
        raise ValueError("pretrained encoder requires patch_size=16 and img_size=384; task geometry is separate")
    root = Path(__file__).resolve().parents[2] / "third_party" / UPSTREAM.rsplit("/", 1)[-1]
    prepare_upstream(root, ("src", "app"))
    from app.vjepa_2_1.models import vision_transformer as vit
    kwargs = dict(patch_size=16, img_size=(384, 384), num_frames=64,
                  tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True,
                  uniform_power=False, use_rope=True, img_temporal_dim_size=1,
                  interpolate_rope=True, modality_embedding=True)
    fixture = {k: int(spec[k]) for k in ("embed_dim", "depth", "num_heads") if k in spec}
    if fixture:
        if set(fixture) != {"embed_dim", "depth", "num_heads"}:
            raise ValueError("fixture dimensions must specify embed_dim, depth and num_heads together")
        model = vit.VisionTransformer(**kwargs, **fixture)
    else:
        model = getattr(vit, arch)(**kwargs)
    if spec.get("encoder"):
        blob = torch.load(spec["encoder"], map_location="cpu", weights_only=False)
        key = _VARIANTS[arch]
        if not isinstance(blob, dict) or key not in blob:
            raise ValueError(f"checkpoint requires {key}; no encoder substitution is allowed")
        state = {}
        for name, value in blob[key].items():
            clean = name.replace("module.", "").replace("backbone.", "")
            if clean in state:
                raise ValueError("checkpoint prefix removal creates duplicate keys")
            state[clean] = value
        model.load_state_dict(state, strict=True)
    return model


class Backbone(nn.Module):
    def __init__(self, encoder, *, trainable=False):
        super().__init__()
        self.encoder = encoder
        self.out_channels = encoder.embed_dim
        self.trainable = trainable
        self.requires_grad_(trainable)
        self.train(trainable)

    def train(self, mode=True):
        return super().train(mode if self.trainable else False)

    def _tokens(self, clip):
        param = next(self.encoder.parameters())
        with nullcontext() if self.trainable else torch.no_grad():
            return self.encoder(clip.to(device=param.device, dtype=param.dtype)).float()

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError("images must be nonempty [B, 3, H, W]")
        h, w = images.shape[-2:]
        images = F.pad(images, (0, (-w) % 16, 0, (-h) % 16))
        tokens = self._tokens(images.unsqueeze(2))
        gh, gw = images.shape[-2] // 16, images.shape[-1] // 16
        if tokens.ndim != 3 or tokens.shape[1] != gh * gw:
            raise RuntimeError("encoder tokens do not match the image patch grid")
        return tokens.transpose(1, 2).reshape(images.shape[0], self.out_channels, gh, gw)

    def video_tokens(self, clip):
        if (clip.ndim != 5 or clip.shape[2] != 3 or clip.shape[1] < 2
                or clip.shape[1] % 2 or min(clip.shape[-2:]) < 1):
            raise ValueError("video requires [B, frames, 3, H, W] with positive even frames >= 2")
        h, w = clip.shape[-2:]
        clip = F.pad(clip, (0, (-w) % 16, 0, (-h) % 16))
        return self._tokens(clip.permute(0, 2, 1, 3, 4).contiguous())

    def forward(self, images):
        return self.forward_features(images)


def build(spec):
    return Backbone(_encoder(spec))


def build_trainable(spec):
    return Backbone(_encoder(spec), trainable=True)
