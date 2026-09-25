"""Basic5 image-trunk components, independent of legacy feature extraction."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from downstream.patch_vision import PatchVisionBackbone, load_local

KIND = "sam3_trunk"
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
COMPONENT_ONLY = True
CAPTURE_PYRAMID = False
SUPPORTED_ADAPTATIONS = ("frozen", "finetune")


class TrunkBackbone(PatchVisionBackbone):
    def _forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError("images require nonempty [B, 3, H, W]")
        pixel = F.pad(
            images,
            (
                0,
                -images.shape[-1] % self.patch_size,
                0,
                -images.shape[-2] % self.patch_size,
            ),
        )
        param = next(self.model.parameters())
        pixel = pixel.to(device=param.device, dtype=param.dtype)
        hp, wp = pixel.shape[-2] // self.patch_size, pixel.shape[-1] // self.patch_size
        from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding

        scale = self.model.config.window_size / hp
        for layer in self.model.layers:
            if layer.window_size != 0:
                continue
            emb = layer.rotary_emb
            if (emb.end_x, emb.end_y, emb.scale) != (wp, hp, scale):
                table = Sam3ViTRotaryEmbedding(self.model.config, wp, hp, scale=scale)
                emb.end_x, emb.end_y, emb.scale = wp, hp, scale
                emb.rope_embeddings_cos = table.rope_embeddings_cos.to(
                    emb.rope_embeddings_cos
                )
                emb.rope_embeddings_sin = table.rope_embeddings_sin.to(
                    emb.rope_embeddings_sin
                )
        with nullcontext() if self.trainable else torch.no_grad():
            tokens = self.model(pixel_values=pixel).last_hidden_state.float()
        if tokens.shape != (images.shape[0], hp * wp, self.out_channels):
            raise ValueError("trunk tokens do not match the spatial grid")
        return (
            SimpleNamespace(last_hidden_state=tokens, pooler_output=tokens.mean(1)),
            pixel.shape[-2:],
        )


def _build(spec, trainable):
    from transformers import Sam3ViTModel

    expected = dict(
        hidden_size=1024,
        num_hidden_layers=32,
        num_attention_heads=16,
        patch_size=14,
        pretrain_image_size=336,
        window_size=24,
    )
    if spec.get("encoder") and Path(spec["encoder"]).is_file():
        if spec.get("arch") not in ("released", "fixture"):
            raise ValueError("arch must explicitly select released or fixture")
        # Match the existing providers' isolated sibling imports. The
        # methods tree is not an installable dependency of a method environment.
        module_spec = importlib.util.spec_from_file_location(
            "_basic5_sam3_trunk", Path(__file__).with_name("sam3_trunk.py")
        )
        converter = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(converter)
        model = converter.load_official_trunk(spec["encoder"], img_size=spec["img_size"])
        if spec["arch"] == "released" and any(
            getattr(model.config, k, None) != v for k, v in expected.items()
        ):
            raise ValueError(
                "checkpoint architecture differs from the released profile"
            )
        if model.config.patch_size != spec["patch_size"]:
            raise ValueError("checkpoint patch_size differs from backbone geometry")
    else:
        model = load_local(spec, Sam3ViTModel, expected)
    if model.config.image_size != spec["img_size"]:
        raise ValueError("checkpoint image_size differs from backbone geometry")
    return TrunkBackbone(
        model,
        trainable=trainable,
        channels=model.config.hidden_size,
        stride=model.config.patch_size,
        depth=model.config.num_hidden_layers,
        block_name="layers",
    )


def build(spec):
    return _build(spec, False)


def build_trainable(spec):
    return _build(spec, True)
