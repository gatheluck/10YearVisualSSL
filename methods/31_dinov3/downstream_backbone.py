"""Explicit local released-vision component provider; legacy adapter is unchanged."""
from downstream.hf_vision import build_vision

KIND = 'dinov3_hf'
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
COMPONENT_ONLY = True
CAPTURE_PYRAMID = True
SUPPORTED_ADAPTATIONS = ("frozen", "finetune")


def build(spec):
    return build_vision(spec, family='register_cls', expected={'hidden_size': 4096, 'num_hidden_layers': 40, 'num_attention_heads': 32, 'patch_size': 16, 'num_register_tokens': 4})


def build_trainable(spec):
    return build_vision(spec, family='register_cls', expected={'hidden_size': 4096, 'num_hidden_layers': 40, 'num_attention_heads': 32, 'patch_size': 16, 'num_register_tokens': 4}, trainable=True)
