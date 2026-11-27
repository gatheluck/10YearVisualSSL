"""Frozen scaling evaluation with source-specific encoder and input geometry."""

from downstream.frozen_vit import load_backbone

KIND = "rotation_scaling"
COMPONENT_ONLY = True
IMAGE_CLASSIFICATION = True
IMAGENET_PROBE_INTERPOLATION = "bicubic"
IMAGENET_EXECUTION = {
    "tail_policy": "discard",
    "rank_seed_stride": 1000,
    "distributed": False,
}
VIDEO_RECIPE = {
    "interpolation": "bicubic",
    "temporal_rng": "python",
    "augmentation": "plain",
    "batch_mixing": "none",
}
SUPPORTED_ADAPTATIONS = ("frozen",)
NATIVE_DETECTION = True
PRESETS = {
    "vit_base_patch16_224": (768, 12, 12),
    "vit_large_patch16_224": (1024, 24, 16),
}
NATIVE_PREFIXES = ("vit.",)

NATIVE_DISCARDED_PREFIXES = ("vit.head.",)


def build_encoder(arch):
    if arch not in PRESETS:
        raise ValueError("supported native ViT preset required")
    width, depth, heads = PRESETS[arch]
    from timm.models.vision_transformer import VisionTransformer
    from torch import nn

    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=width,
        depth=depth,
        num_heads=heads,
        num_classes=0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
    )


def build(spec):
    return load_backbone(spec, kind=KIND, factory=build_encoder, transform=None)
