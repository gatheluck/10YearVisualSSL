"""Explicit local released-vision component provider; legacy adapter is unchanged."""
from downstream.hf_vision import build_vision

KIND = 'dinov3_hf'
EXTENDED_IMAGE_READER = "captured_single_block_v1"
EXTENDED_VIDEO_READER = "captured_single_block_v1"
EXTENDED_DENSE_READER = "captured_single_block_v1"
EXTENDED_ACCUMULATION_TAILS = {"frozen": "discard", "attentive": "flush_scaled"}
TRAINABLE = True
ACTIVATION_CHECKPOINTING = ('hf', 'model')
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
IMAGENET_FT_RECIPE = "captured_bilinear_plain_torch_v1"
IMAGENET_PROBE_INTERPOLATION = "bilinear"
IMAGENET_EXECUTION = dict(tail_policy="discard", rank_seed_stride=17, distributed=True)
DENSE_EXECUTION = {"tail_policy": "discard", "clip_frozen_detection": False}
VIDEO_RECIPE = {'interpolation': 'bilinear', 'temporal_rng': 'torch', 'augmentation': 'plain', 'batch_mixing': 'torch'}
COMPONENT_ONLY = True
CAPTURE_PYRAMID = True
SUPPORTED_ADAPTATIONS = ("frozen", "attentive", "finetune")
ATTENTIVE_PROFILE = "captured_single_block_v1"
NATIVE_DETECTION = True


def build(spec, *, load_dtype=None):
    return build_vision(spec, family='register_cls', expected={'hidden_size': 4096, 'num_hidden_layers': 40, 'num_attention_heads': 32, 'patch_size': 16, 'num_register_tokens': 4}, load_dtype=load_dtype)


def build_revised_cuda(spec):
    import torch
    backbone = build(spec, load_dtype=torch.bfloat16)
    # The loader constructs the rotary frequencies in float32.
    backbone.preserve_revised_storage = True
    return backbone


def build_trainable(spec):
    return build_vision(spec, family='register_cls', expected={'hidden_size': 4096, 'num_hidden_layers': 40, 'num_attention_heads': 32, 'patch_size': 16, 'num_register_tokens': 4}, trainable=True)


def revised_schedule(spec, task, track, name):
    """Explicit author-reported compute-limited schedules for the 7B release."""
    schedules = {
        ('imagenet', 'lora', 'B5X_LORA_LP_50EP_v2'): 50,
        ('ssv2', 'lora', 'B5X_LORA_LP_24EP_v2'): 24,
    }
    if spec.get('arch') != 'released' or not spec.get('encoder'):
        raise ValueError('short schedule requires the released checkpoint profile')
    key = (task, track, name)
    if key not in schedules:
        raise ValueError('unconfirmed task, track or named schedule')
    return {'epochs': schedules[key], 'protocol_id': name}
