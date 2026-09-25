"""Basic5 final-merger readout; legacy patch-token extraction stays separate."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from downstream.patch_vision import PatchVisionBackbone, load_local

KIND = "cosmos3_super_vm"
TRAINABLE = True
FINETUNE_GROUPS = True
IMAGE_CLASSIFICATION = True
COMPONENT_ONLY = True
CAPTURE_PYRAMID = False
SUPPORTED_ADAPTATIONS = ("frozen", "finetune")


class MergerBackbone(PatchVisionBackbone):
    def _forward(self, images):
        # Convert task ImageNet normalization once; zero pad in Qwen space.
        # _pixels also validates the shared image layout.
        pixel = self._pixels(images)
        pixel = F.pad(
            pixel,
            (
                0,
                -pixel.shape[-1] % self.patch_size,
                0,
                -pixel.shape[-2] % self.patch_size,
            ),
        )
        param = next(self.model.parameters())
        pixel = pixel.to(device=param.device, dtype=param.dtype)
        cfg = self.model.config
        b, c, h, w = pixel.shape
        ps, m, t = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size
        gh, gw = h // ps, w // ps
        patches = pixel.reshape(b, c, gh // m, m, ps, gw // m, m, ps).permute(
            0, 2, 5, 3, 6, 1, 4, 7
        )
        patches = (
            patches.unsqueeze(6)
            .expand(-1, -1, -1, -1, -1, -1, t, -1, -1)
            .reshape(-1, c * t * ps * ps)
        )
        grid = torch.tensor([[1, gh, gw]] * b, device=pixel.device, dtype=torch.long)
        with nullcontext() if self.trainable else torch.no_grad():
            out = self.model(hidden_states=patches, grid_thw=grid)
        tokens = out.pooler_output
        if tokens is None or tokens.shape != (
            b * (gh // m) * (gw // m),
            self.out_channels,
        ):
            raise ValueError("final merger output does not match the spatial grid")
        tokens = tokens.reshape(b, (gh // m) * (gw // m), self.out_channels).float()
        return SimpleNamespace(
            last_hidden_state=tokens, pooler_output=tokens.mean(1)
        ), (h, w)


def _build(spec, trainable):
    from transformers import Qwen3VLVisionModel

    expected = dict(
        hidden_size=1152,
        out_hidden_size=5120,
        depth=27,
        num_heads=16,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        deepstack_visual_indexes=[8, 16, 24],
    )
    model = load_local(spec, Qwen3VLVisionModel, expected)
    cfg = model.config
    return MergerBackbone(
        model,
        trainable=trainable,
        channels=cfg.out_hidden_size,
        stride=cfg.patch_size * cfg.spatial_merge_size,
        depth=cfg.depth,
        block_name="blocks",
    )


def build(spec):
    return _build(spec, False)


def build_trainable(spec):
    return _build(spec, True)
