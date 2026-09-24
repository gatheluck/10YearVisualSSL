"""Explicit local released-vision component provider; legacy adapter is unchanged."""
from downstream.hf_vision import build_vision

KIND = 'clip_hf'
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
COMPONENT_ONLY = True
CAPTURE_PYRAMID = False
SUPPORTED_ADAPTATIONS = ("frozen", "finetune")


def build(spec):
    return build_vision(spec, family='projected_cls', expected={'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16, 'patch_size': 14, 'image_size': 336, 'projection_dim': 768})


def build_trainable(spec):
    return build_vision(spec, family='projected_cls', expected={'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16, 'patch_size': 14, 'image_size': 336, 'projection_dim': 768}, trainable=True)
