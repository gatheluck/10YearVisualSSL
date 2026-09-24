"""Local vision-only transformer providers with explicit captured readouts.

Task inputs use ImageNet normalization. Conversion to a model's normalization
happens exactly once here. Checkpoints are never fetched or silently replaced.
These are component integrations, not complete canonical experiment recipes.
"""
from contextlib import nullcontext
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

_IM_MEAN = (.485, .456, .406)
_IM_STD = (.229, .224, .225)


def build_vision(spec, *, family, expected, trainable=False):
    """Read a complete local HF snapshot; reduced fixtures need arch=fixture."""
    from transformers import (AutoConfig, CLIPVisionModelWithProjection,
                              SiglipVisionModel, DINOv3ViTModel)
    path = Path(spec.get("encoder", ""))
    if not spec.get("encoder") or not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("encoder must be a local checkpoint directory containing config.json")
    outer = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    config = getattr(outer, "vision_config", outer)
    if family == "projected_cls" and config is not outer:
        # The multimodal checkpoint owns this projection outside vision_config.
        config.projection_dim = outer.projection_dim
    classes = {"projected_cls": CLIPVisionModelWithProjection,
               "map_pool": SiglipVisionModel, "register_cls": DINOv3ViTModel}
    cls = classes[family]
    if not isinstance(config, cls.config_class):
        raise ValueError("checkpoint config does not match the requested vision family")
    if spec.get("arch") not in ("released", "fixture"):
        raise ValueError("arch must explicitly select released or fixture")
    if spec["arch"] == "released" and any(getattr(config, k, None) != v for k, v in expected.items()):
        raise ValueError("checkpoint architecture differs from the released model profile")
    if config.patch_size != spec["patch_size"] or config.image_size != spec["img_size"]:
        raise ValueError("checkpoint geometry differs from backbone patch_size/img_size")
    if family == "register_cls":
        config.pos_embed_shift = config.pos_embed_jitter = config.pos_embed_rescale = None
    model, info = cls.from_pretrained(path, config=config, local_files_only=True,
                                     trust_remote_code=False, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("incomplete checkpoint: missing or incompatible vision weights")
    # Full multimodal snapshots may also carry a text tower; never instantiate it.
    ignored = ("text_model.", "text_projection.", "logit_scale", "logit_bias")
    if any(not key.startswith(ignored) for key in info.get("unexpected_keys", [])):
        raise ValueError("checkpoint has unexpected non-text weights")
    return VisionBackbone(model, family=family, trainable=trainable)


class VisionBackbone(nn.Module):
    def __init__(self, model, *, family, trainable=False):
        super().__init__()
        self.model, self.family, self.trainable = model, family, trainable
        self.out_channels = int(model.config.hidden_size)
        self.global_channels = int(model.config.projection_dim) if family == "projected_cls" else self.out_channels
        self.classifier_init_std = 0. if family == "register_cls" else .01
        self.patch_size = int(model.config.patch_size)
        self.requires_grad_(trainable)
        self.train(trainable)

    def train(self, mode=True):
        super().train(mode if self.trainable else False)
        if self.family == "register_cls" and hasattr(self.model, "rope_embeddings"):
            self.model.rope_embeddings.eval()
        return self

    def _pixels(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError("images require nonempty [B, 3, H, W]")
        if self.family == "register_cls":
            return F.pad(images, (0, -images.shape[-1] % self.patch_size,
                                  0, -images.shape[-2] % self.patch_size))
        mean, std = ((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)) if self.family == "projected_cls" else ((.5,)*3, (.5,)*3)
        def channel(values):
            return images.new_tensor(values)[None, :, None, None]
        return (images * channel(_IM_STD) + channel(_IM_MEAN) - channel(mean)) / channel(std)

    def _forward(self, images):
        pixel = self._pixels(images)
        param = next(self.model.parameters())
        kwargs = {} if self.family == "register_cls" else {"interpolate_pos_encoding": True}
        with nullcontext() if self.trainable else torch.no_grad():
            out = self.model(pixel_values=pixel.to(device=param.device, dtype=param.dtype), **kwargs)
        return out, pixel.shape[-2:]

    def forward_features(self, images):
        out, (height, width) = self._forward(images)
        prefix = (1 + self.model.config.num_register_tokens if self.family == "register_cls"
                  else 1 if self.family == "projected_cls" else 0)
        tokens = out.last_hidden_state[:, prefix:].float()
        gh, gw = height // self.patch_size, width // self.patch_size
        if tokens.ndim != 3 or tokens.shape[1:] != (gh * gw, self.out_channels) or min(gh, gw) < 1:
            raise ValueError("vision tokens do not match the real spatial grid")
        return tokens.transpose(1, 2).reshape(images.shape[0], self.out_channels, gh, gw)

    def classification_features(self, images, *, adaptation, video=False):
        if adaptation not in ("frozen", "finetune"):
            raise ValueError("attentive reader recipe is not reconciled for this provider")
        if video:
            if images.ndim != 5 or images.shape[1] < 1:
                raise ValueError("video requires [B, frames, 3, H, W]")
            batch, frames = images.shape[:2]
            images = images.flatten(0, 1)
        out, _ = self._forward(images)
        pooled = out.image_embeds if self.family == "projected_cls" else out.pooler_output
        if pooled is None:
            raise ValueError("official global pooling output is missing")
        feat = pooled.float()
        # The captured CLS family normalizes each frame; MAP/projected families
        # normalize only the frozen image probe, never temporal means or FT.
        if self.family == "register_cls" or (not video and adaptation == "frozen"):
            feat = F.normalize(feat, dim=-1)
        return feat.reshape(batch, frames, -1).mean(1) if video else feat

    def forward(self, images):
        return self.forward_features(images)

    def finetune_group_policy(self):
        depth = int(self.model.config.num_hidden_layers)
        endpoint = depth + 1 if self.family == "register_cls" else depth
        entries, blocks = {}, set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            pattern = r"(?:^|\.)layer\.(\d+)\." if self.family == "register_cls" else r"encoder\.layers\.(\d+)\."
            match = re.search(pattern, name)
            layer = 0
            if match:
                index = int(match[1]); blocks.add(index); layer = index + 1
            elif self.family == "register_cls":
                if re.search(r"(?:^|\.)norm\.(weight|bias)$", name):
                    layer = depth
            elif any(key in name for key in ("post_layernorm", ".head.", "visual_projection")):
                layer = depth
            lower = name.lower()
            if self.family == "register_cls":
                no_decay = (lower.endswith("bias") or any(k in lower for k in (
                    "norm", "cls_token", "register_tokens", "mask_token", "layer_scale")) or lower.endswith("lambda1"))
            else:
                no_decay = any(k in lower for k in ("bias", "norm", "ln", "position_embedding", "pos_embed", "positional", "class_token", "cls_token"))
                if self.family == "projected_cls":
                    no_decay = no_decay or "class_embedding" in lower
            entries[name] = (layer, no_decay)
        if blocks != set(range(depth)):
            raise ValueError("finetune policy requires all encoder blocks without gaps")
        return endpoint, entries
