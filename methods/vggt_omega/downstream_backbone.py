"""Pinned aggregator-only image/native-video components for the Omega profile."""
from contextlib import nullcontext
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from provider_support import prepare_upstream
from downstream.hf_vision import vision_no_decay
from downstream.contract import sha256_file

KIND = "vggt_omega"
EXTENDED_IMAGE_READER = "captured_cross_self_v1"
EXTENDED_VIDEO_READER = "captured_cross_self_v1"
EXTENDED_DENSE_READER = "captured_cross_self_v1"
EXTENDED_ACCUMULATION_TAILS = {"frozen": "discard", "attentive": "discard"}
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
IMAGENET_FT_RECIPE = "captured_bicubic_unit_mixup_v1"
IMAGENET_PROBE_INTERPOLATION = "bicubic"
IMAGENET_EXECUTION = dict(tail_policy="discard", rank_seed_stride=1, distributed=False)
VIDEO_RECIPE = {'interpolation': 'bicubic', 'temporal_rng': 'python', 'augmentation': 'shared_unit', 'batch_mixing': 'mixup'}
COMPONENT_ONLY = True
CAPTURE_PYRAMID = False
NATIVE_DETECTION = True
DETECTION_LABEL_SPACE = "contiguous"
SUPPORTED_ADAPTATIONS = ("frozen", "attentive", "finetune")
ATTENTIVE_PROFILE = "captured_cross_self_v1"
RELEASED_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"


def _encoder(spec):
    if spec.get("arch") not in ("released", "fixture"):
        raise ValueError("arch must explicitly select released or fixture")
    if spec.get("patch_size") != 16 or spec.get("img_size") != 224:
        raise ValueError("aggregator requires patch_size=16 and img_size=224")
    path = Path(spec.get("encoder", ""))
    if not spec.get("encoder") or not path.is_file():
        raise ValueError("encoder must be a complete local aggregator checkpoint")
    dims = {k: spec[k] for k in ("embed_dim", "depth", "num_heads") if k in spec}
    if spec["arch"] == "released" and dims:
        raise ValueError("released architecture cannot override dimensions")
    if spec["arch"] == "released" and sha256_file(path) != RELEASED_SHA256:
        raise ValueError("checkpoint identity differs from the selected 512 visual release")
    if spec["arch"] == "fixture" and (
        set(dims) != {"embed_dim", "depth", "num_heads"}
        or any(type(v) is not int or v < 1 for v in dims.values())
        or dims["embed_dim"] % 64 or dims["embed_dim"] % (4 * dims["num_heads"])
    ):
        raise ValueError("fixture requires complete dimensions and valid RoPE head widths")
    root = Path(__file__).resolve().parents[2] / "third_party" / "vggt-omega"
    prepare_upstream(root, ("vggt_omega",))
    from vggt_omega.models.aggregator import Aggregator
    kwargs = (dict(**dims, register_attention_block_indices=[0],
                   cached_layer_indices=tuple(range(dims["depth"]))) if dims else {})
    model = Aggregator(**kwargs)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict) or not any(k.startswith("aggregator.") for k in state):
        raise ValueError("checkpoint requires explicit aggregator weights")
    if any(not k.startswith(("aggregator.", "camera_head.", "dense_head.", "text_alignment_head.")) for k in state):
        raise ValueError("checkpoint has unexpected non-aggregator weights")
    model.load_state_dict({k.removeprefix("aggregator."):v for k,v in state.items()
                           if k.startswith("aggregator.")}, strict=True)
    if model.patch_token_start != 17:
        raise ValueError("aggregator requires one camera and sixteen register tokens")
    return model


class Backbone(nn.Module):
    reader_profile = ATTENTIVE_PROFILE
    classifier_init_std = .01
    patch_size = 16
    native_pyramid_style = "transposed"
    detection_num_classes = 81

    def __init__(self, aggregator, *, trainable=False):
        super().__init__()
        self.aggregator = aggregator
        self.out_channels = self.global_channels = 2 * aggregator.camera_token.shape[-1]
        self.trainable = trainable
        self.requires_grad_(trainable)
        self.train(trainable)

    def train(self, mode=True):
        return super().train(mode if self.trainable else False)

    def _patches(self, images, *, raw=False):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if (images.ndim != 5 or images.shape[1] < 1 or images.shape[2] != 3
                or min(images.shape[-2:]) < 16 or any(n % 16 for n in images.shape[-2:])):
            raise ValueError("aggregator requires nonempty [B, frames, 3, H, W] with patch-aligned geometry")
        if not raw:
            mean = images.new_tensor((.485, .456, .406))[None, None, :, None, None]
            std = images.new_tensor((.229, .224, .225))[None, None, :, None, None]
            images = images * std + mean
        param = next(self.aggregator.parameters())
        with nullcontext() if self.trainable else torch.no_grad():
            cached, start = self.aggregator(images.to(device=param.device, dtype=param.dtype))
            last = next((x for x in reversed(cached) if x is not None), None)
            if last is None or start != 17:
                raise ValueError("aggregator must return cached layers and the expected prefix")
            patches = last[:, :, start:, :].float()
        expected = (images.shape[0], images.shape[1],
                    (images.shape[-2] // 16) * (images.shape[-1] // 16), self.out_channels)
        if patches.shape != expected:
            raise ValueError("aggregator patches do not match the actual image/frame grid")
        return patches

    def forward_features(self, images):
        if images.ndim != 4:
            raise ValueError("spatial features require single images [B, 3, H, W]")
        patches = self._patches(images)[:, 0]
        return patches.transpose(1, 2).reshape(images.shape[0], self.out_channels,
                                              images.shape[-2] // 16, images.shape[-1] // 16)

    def video_tokens(self, clip):
        if clip.ndim != 5:
            raise ValueError("native video requires [B, frames, 3, H, W]")
        return self._patches(clip).flatten(1, 2)

    def detection_normalization(self):
        return (0., 0., 0.), (1., 1., 1.)

    def forward_detection_features(self, images):
        patches = self._patches(images, raw=True)[:, 0]
        return patches.transpose(1, 2).reshape(images.shape[0], self.out_channels,
                                              images.shape[-2] // 16, images.shape[-1] // 16)

    def classification_features(self, images, *, adaptation, video=False):
        if adaptation not in ("frozen", "finetune"):
            raise ValueError("classification requires frozen or finetune readout")
        if video and images.ndim != 5:
            raise ValueError("native video requires [B, frames, 3, H, W]")
        feat = self._patches(images).mean((1, 2))
        return F.normalize(feat, dim=-1) if adaptation == "frozen" else feat

    def structured_global_features(self, images):
        """Preserve pooled magnitude in the captured structured-task readout."""
        return self.classification_features(images, adaptation="finetune")

    def forward(self, images):
        return self.forward_features(images)

    def finetune_group_policy(self):
        depth = len(self.aggregator.frame_blocks)
        entries, seen = {}, {"frame_blocks":set(), "inter_frame_blocks":set()}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            block = re.match(r"aggregator\.(frame_blocks|inter_frame_blocks)\.(\d+)\.", name)
            layer = 0
            if block:
                index = int(block[2]); seen[block[1]].add(index); layer = index + 1
            lower = name.lower()
            no_decay = vision_no_decay(name) or any(k in lower for k in ("camera_token", "register_token", "rope"))
            entries[name] = (layer, no_decay)
        if any(indices != set(range(depth)) for indices in seen.values()):
            raise ValueError("layer groups require complete paired aggregator blocks")
        return depth, entries


def build(spec):
    return Backbone(_encoder(spec))


def build_trainable(spec):
    return Backbone(_encoder(spec), trainable=True)
