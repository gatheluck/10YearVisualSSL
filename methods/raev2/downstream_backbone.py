"""Captured K7 tokenizer: seven normalized DINOv3-L layers plus a global term.

This is distinct from the historical last-seven sum and final-layer-only RAE
profiles. The decoder and diffusion transformer are never instantiated.
"""
from contextlib import nullcontext
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from provider_support import prepare_upstream

KIND = "raev2_k7"
EXTENDED_IMAGE_READER = "captured_single_block_v1"
EXTENDED_VIDEO_READER = "captured_single_block_v1"
EXTENDED_DENSE_READER = "captured_single_block_v1"
EXTENDED_ACCUMULATION_TAILS = {"frozen": "discard", "attentive": "flush_scaled"}
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
IMAGENET_FT_RECIPE = "captured_bilinear_raw_zero_torch_v1"
COMPONENT_ONLY = True
CAPTURE_PYRAMID = False
NATIVE_DETECTION = True
SUPPORTED_ADAPTATIONS = ("frozen", "attentive", "finetune")
ATTENTIVE_PROFILE = "captured_single_block_v1"
K7_LAYERS = (11, 13, 15, 17, 19, 21, 23)


def _encoder(spec):
    if spec.get("arch") not in ("released", "fixture"):
        raise ValueError("arch must explicitly select released or fixture")
    if spec.get("patch_size") != 16 or spec.get("img_size") != 224:
        raise ValueError("K7 requires patch_size=16 and pretrained img_size=224")
    path = Path(spec.get("encoder", ""))
    if not spec.get("encoder") or not path.is_file():
        raise ValueError("encoder must be a complete local DINOv3-L checkpoint")
    dims = {k: spec[k] for k in ("embed_dim", "depth", "num_heads") if k in spec}
    if spec["arch"] == "released" and dims:
        raise ValueError("released architecture cannot override dimensions")
    if spec["arch"] == "fixture" and (
        set(dims) != {"embed_dim", "depth", "num_heads"} or dims["depth"] != 24
        or any(type(v) is not int or v < 1 for v in dims.values())
        or dims["embed_dim"] % (4 * dims["num_heads"])
    ):
        raise ValueError("fixture requires 24 layers and valid complete dimensions")
    root = Path(__file__).resolve().parents[2] / "third_party" / "dinov3"
    prepare_upstream(root, ("dinov3",))
    if spec["arch"] == "released":
        from dinov3.hub.backbones import dinov3_vitl16
        model = dinov3_vitl16(pretrained=False)
    else:
        from dinov3.models.vision_transformer import DinoVisionTransformer
        model = DinoVisionTransformer(img_size=224, patch_size=16, **dims,
                                     n_storage_tokens=4, pos_embed_rope_rescale_coords=2.)
        model.init_weights()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(k.startswith("blocks.") for k in state):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    # Check the complete original state first; replacing this earlier would
    # hide missing learned norm weights in an incomplete checkpoint.
    model.norm = nn.LayerNorm(model.embed_dim, elementwise_affine=False)
    return model


class Backbone(nn.Module):
    reader_profile = ATTENTIVE_PROFILE
    native_pyramid_style = "bilinear"
    classifier_init_std = 0.
    patch_size = 16

    def __init__(self, model, *, trainable=False):
        super().__init__()
        self.model = model
        self.out_channels = self.global_channels = model.embed_dim
        self.trainable = trainable
        self.requires_grad_(trainable)
        self.train(trainable)

    def train(self, mode=True):
        super().train(mode if self.trainable else False)
        # Preserve evaluation geometry even when the complete encoder trains.
        self.model.rope_embed.eval()
        return self

    def _tokens(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError("images require nonempty [B, 3, H, W]")
        pixels = F.pad(images, (0, -images.shape[-1] % 16, 0, -images.shape[-2] % 16))
        param = next(self.model.parameters())
        with nullcontext() if self.trainable else torch.no_grad():
            layers = self.model.get_intermediate_layers(
                pixels.to(device=param.device, dtype=param.dtype), n=list(K7_LAYERS),
                reshape=False, return_class_token=False, norm=True)
            if len(layers) != 7:
                raise ValueError("encoder did not return all seven selected layers")
            tokens = torch.stack(layers).mean(0) + layers[-1].mean(1, keepdim=True)
        return tokens.float(), (pixels.shape[-2] // 16, pixels.shape[-1] // 16)

    def forward_features(self, images):
        tokens, (height, width) = self._tokens(images)
        if tokens.shape != (images.shape[0], height * width, self.out_channels):
            raise ValueError("K7 patch tokens do not match the actual spatial grid")
        return tokens.transpose(1, 2).reshape(images.shape[0], self.out_channels, height, width)

    def video_tokens(self, clip):
        if clip.ndim != 5 or clip.shape[1] < 1 or clip.shape[2] != 3:
            raise ValueError("video requires [B, frames, 3, H, W]")
        tokens, _ = self._tokens(clip.flatten(0, 1))
        return tokens.mean(1).reshape(clip.shape[0], clip.shape[1], self.out_channels)

    def classification_features(self, images, *, adaptation, video=False):
        if adaptation not in ("frozen", "finetune"):
            raise ValueError("classification requires frozen or finetune readout")
        if video:
            return F.normalize(self.video_tokens(images), dim=-1).mean(1)
        tokens, _ = self._tokens(images)
        return F.normalize(tokens.mean(1), dim=-1)

    def detection_normalization(self):
        return (0., 0., 0.), (1., 1., 1.)

    def forward_detection_features(self, images):
        mean = images.new_tensor((.485, .456, .406))[None, :, None, None]
        std = images.new_tensor((.229, .224, .225))[None, :, None, None]
        return self.forward_features((images - mean) / std)

    def forward(self, images):
        return self.forward_features(images)

    def finetune_group_policy(self):
        depth = len(self.model.blocks)
        entries, blocks = {}, set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            block = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
            layer = 0
            if block:
                index = int(block[1]); blocks.add(index); layer = index + 1
            elif re.search(r"(?:^|\.)norm\.(weight|bias)$", name):
                layer = depth
            lower = name.lower()
            no_decay = (lower.endswith("bias") or any(k in lower for k in (
                "norm", "cls_token", "storage_tokens", "register_tokens", "mask_token", "layer_scale"))
                or lower.endswith(("ls1", "ls2")) or ".ls1." in lower or ".ls2." in lower)
            entries[name] = (layer, no_decay)
        if blocks != set(range(depth)) or depth != 24:
            raise ValueError("K7 finetune policy requires all 24 encoder blocks")
        return depth + 1, entries


def build(spec):
    return Backbone(_encoder(spec))


def build_trainable(spec):
    return Backbone(_encoder(spec), trainable=True)
