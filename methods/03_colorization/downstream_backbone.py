"""Frozen scaling evaluation with source-specific encoder and input geometry."""

from downstream.frozen_vit import lightness_input, load_backbone

KIND = "colorization_scaling"
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
NATIVE_PREFIXES = ("",)

NATIVE_DISCARDED_PREFIXES = ("decoder.",)


def build_encoder(arch):
    if arch not in PRESETS:
        raise ValueError("supported native ViT preset required")
    width, depth, heads = PRESETS[arch]
    from importlib.machinery import ModuleSpec, SourceFileLoader
    from importlib.util import module_from_spec
    from pathlib import Path

    path = Path(__file__).parent / "models/vit_colorization.py"
    loader = SourceFileLoader("_scaling_color_trunk", str(path))
    module = module_from_spec(ModuleSpec(loader.name, loader, origin=str(path)))
    loader.exec_module(module)
    return module._ViTTrunk(embed_dim=width, depth=depth, num_heads=heads)


def build(spec):
    return load_backbone(
        spec, kind=KIND, factory=build_encoder, transform=lightness_input
    )
