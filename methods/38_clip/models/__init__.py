"""Model construction for 38_clip, built on the pinned openai/CLIP submodule."""

from __future__ import annotations

from .clip_backbone import (
    CLIP_MEAN,
    CLIP_STD,
    OFFICIAL_COMMIT,
    OFFICIAL_VIT_B32_SHA256,
    STEP2_PROTOCOL,
    build_clip,
    build_clip_visual,
    load_official_clip_package,
    load_official_imagenet_metadata,
    load_official_vit_b32,
    sha256_file,
    tokenize_prompts,
)

__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "OFFICIAL_COMMIT",
    "OFFICIAL_VIT_B32_SHA256",
    "STEP2_PROTOCOL",
    "build_clip",
    "build_clip_visual",
    "load_official_clip_package",
    "load_official_imagenet_metadata",
    "load_official_vit_b32",
    "sha256_file",
    "tokenize_prompts",
]
