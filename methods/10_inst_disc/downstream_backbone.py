"""Frozen BasicFive readouts for the portable IDv2 scaling export."""
import importlib.util
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

KIND = "idv2_scaling"
COMPONENT_ONLY = True
IMAGE_CLASSIFICATION = True
IMAGENET_PROBE_INTERPOLATION = "bicubic"
IMAGENET_EXECUTION = {"tail_policy": "discard", "rank_seed_stride": 1000, "distributed": False}
VIDEO_RECIPE = {"interpolation": "bicubic", "temporal_rng": "python",
                    "augmentation": "plain", "batch_mixing": "none"}
SUPPORTED_ADAPTATIONS = ("frozen",)
NATIVE_DETECTION = True


def encoder_module():
    # No sys.path mutation or bare models import: other methods coexist here.
    path = Path(__file__).parent / "models" / "scaling_encoder.py"
    loader = SourceFileLoader("_idv2_scaling_encoder", str(path))
    spec = ModuleSpec(loader.name, loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class ScalingBackbone(nn.Module):
    classifier_init_std = .01
    native_pyramid_style = "transposed"

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.out_channels = self.global_channels = encoder.embed_dim
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def tokens(self, images):
        if (images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3
                or min(images.shape[-2:]) < 16
                or any(size % 16 for size in images.shape[-2:])):
            raise ValueError("scaling images require nonempty [B,3,H,W] on a stride-16 grid")
        param = next(self.encoder.parameters())
        return self.encoder.forward_features(images.to(device=param.device, dtype=param.dtype)).float()

    def forward(self, images):
        return self.forward_features(images)

    def forward_features(self, images):
        patches = self.tokens(images)[:, 1:]
        return patches.transpose(1, 2).reshape(images.shape[0], self.out_channels,
                                               images.shape[-2] // 16, images.shape[-1] // 16)

    def detection_normalization(self):
        return (.485, .456, .406), (.229, .224, .225)

    def forward_detection_features(self, images):
        return self.forward_features(images)

    def classification_features(self, images, *, adaptation, video=False):
        if adaptation != "frozen":
            raise ValueError("scaling evaluation verifies frozen LINEAR only")
        if video:
            if images.ndim != 5 or images.shape[1] == 0:
                raise ValueError("scaling video requires nonempty [B,T,3,H,W]")
            cls = self.tokens(images.flatten(0, 1))[:, 0]
            return cls.reshape(images.shape[0], images.shape[1], -1).mean(1)
        return F.normalize(self.tokens(images)[:, 0], dim=-1)


def build(spec):
    if (spec.get("img_size") != 224 or spec.get("patch_size") != 16
            or any(key in spec for key in ("embed_dim", "depth", "num_heads"))):
        raise ValueError("scaling exports require native 224/16 preset geometry without overrides")
    if not spec.get("encoder") or not Path(spec["encoder"]).is_file():
        raise ValueError("scaling encoder must be a local export file")
    module = encoder_module()
    export = module.read_export(spec["encoder"])
    if spec.get("arch") != export["model"]:
        raise ValueError("scaling preset differs from export identity")
    encoder = module.build_encoder(export["model"])
    encoder.load_state_dict(export["state_dict"], strict=True)
    return ScalingBackbone(encoder)
