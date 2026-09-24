"""Explicit local released-vision component provider; legacy adapter is unchanged."""
from downstream.hf_vision import build_vision

KIND = 'siglip2_g'
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
COMPONENT_ONLY = True
# Detection pads after normalization; the shared transform must be reconciled.
CAPTURE_PYRAMID = False
SUPPORTED_ADAPTATIONS = ("frozen", "finetune")


def build(spec):
    return build_vision(spec, family='map_pool', expected={'hidden_size': 1536, 'num_hidden_layers': 40, 'num_attention_heads': 16, 'patch_size': 16, 'image_size': 384})


def build_trainable(spec):
    return build_vision(spec, family='map_pool', expected={'hidden_size': 1536, 'num_hidden_layers': 40, 'num_attention_heads': 16, 'patch_size': 16, 'image_size': 384}, trainable=True)
