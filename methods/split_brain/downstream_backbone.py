"""Frozen scaling evaluation with source-specific encoder and input geometry."""

from downstream.frozen_vit import lab_inputs, load_backbone

KIND = "split_brain_scaling"
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
NATIVE_PREFIXES = ("net1.encoder.", "net2.encoder.")

NATIVE_DISCARDED_PREFIXES = ("net1.decoder.", "net2.decoder.")


def build_encoder(arch):
    if arch not in PRESETS:
        raise ValueError("supported native ViT preset required")
    width, depth, heads = PRESETS[arch]
    import timm
    from torch import nn

    return nn.ModuleList(
        [
            timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0,
                img_size=224,
                in_chans=channels,
                embed_dim=width // 2,
                depth=depth,
                num_heads=heads // 2,
                mlp_ratio=4.0,
            )
            for channels in (1, 2)
        ]
    )


def build(spec):
    return load_backbone(spec, kind=KIND, factory=build_encoder, transform=lab_inputs)
