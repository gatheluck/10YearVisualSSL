"""Explicit local released-vision component provider; legacy adapter is unchanged."""
from downstream.hf_vision import build_vision

KIND = 'clip_hf'
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
IMAGENET_FT_RECIPE = "captured_bicubic_random_python_v1"
IMAGENET_PROBE_INTERPOLATION = "bicubic"
IMAGENET_EXECUTION = dict(tail_policy="discard", rank_seed_stride=1000, distributed=True)
DENSE_EXECUTION = {"tail_policy": "discard", "clip_frozen_detection": False, "distributed": True, "rank_seed_stride": 1000}
VIDEO_RECIPE = {'interpolation': 'bicubic', 'temporal_rng': 'python', 'augmentation': 'shared_random', 'batch_mixing': 'python'}
COMPONENT_ONLY = True
CAPTURE_PYRAMID = False
SUPPORTED_ADAPTATIONS = ("frozen", "attentive", "finetune")
ATTENTIVE_PROFILE = "captured_cross_self_v1"
NATIVE_DETECTION = True


def build(spec):
    return build_vision(spec, family='projected_cls', expected={'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16, 'patch_size': 14, 'image_size': 336, 'projection_dim': 768})


def build_trainable(spec):
    return build_vision(spec, family='projected_cls', expected={'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16, 'patch_size': 14, 'image_size': 336, 'projection_dim': 768}, trainable=True)
