"""Frozen native teacher exports from DINO design/scaling experiments."""
import importlib.util
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

KIND = "dinov3_ablation"
COMPONENT_ONLY = True
IMAGE_CLASSIFICATION = True
IMAGENET_PROBE_INTERPOLATION = "bicubic"
IMAGENET_EXECUTION = {"tail_policy": "discard", "rank_seed_stride": 1000, "distributed": False}
SUPPORTED_ADAPTATIONS = ("frozen",)
NATIVE_DETECTION = True
VIDEO_RECIPE = {"interpolation": "bicubic", "temporal_rng": "python",
                "augmentation": "plain", "batch_mixing": "none"}
PRESETS = {"vit_base_patch16": (768, 12, 12), "vit_large_patch16": (1024, 24, 16)}


def build_encoder(arch):
    if arch not in PRESETS:
        raise ValueError("native teacher requires explicit base/large patch-16 preset")
    path = Path(__file__).parent / "models" / "vision_transformer.py"
    loader = SourceFileLoader("_dino_ablation_vit", str(path))
    spec = ModuleSpec(loader.name, loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    dim, depth, heads = PRESETS[arch]
    return module.VisionTransformer(embed_dim=dim, depth=depth, num_heads=heads,
                                   patch_size=16, n_register_tokens=4)


class TeacherBackbone(nn.Module):
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
                or min(images.shape[-2:]) < 16 or any(s % 16 for s in images.shape[-2:])):
            raise ValueError("teacher images require nonempty [B,3,H,W] on a stride-16 grid")
        param = next(self.encoder.parameters())
        cls, patches = self.encoder(images.to(device=param.device, dtype=param.dtype), is_global=True)
        return cls.float(), patches.float()

    def forward(self, images):
        return self.forward_features(images)

    def forward_features(self, images):
        _, patches = self.tokens(images)
        return patches.transpose(1, 2).reshape(images.shape[0], self.out_channels,
                                             images.shape[-2] // 16, images.shape[-1] // 16)

    def forward_detection_features(self, images):
        return self.forward_features(images)

    def detection_normalization(self):
        return (.485, .456, .406), (.229, .224, .225)

    def classification_features(self, images, *, adaptation, video=False):
        if adaptation != "frozen":
            raise ValueError("native ablation evaluation supports frozen LINEAR only")
        if video:
            if images.ndim != 5 or images.shape[1] == 0:
                raise ValueError("teacher video requires nonempty [B,T,3,H,W]")
            _, patches = self.tokens(images.flatten(0, 1))
            return patches.mean(1).reshape(images.shape[0], images.shape[1], -1).mean(1)
        cls, _ = self.tokens(images)
        return F.normalize(cls, dim=-1)


def build(spec):
    if (spec.get("arch") not in PRESETS or spec.get("img_size") != 224
            or spec.get("patch_size") != 16
            or any(key in spec for key in ("embed_dim", "depth", "num_heads"))):
        raise ValueError("native teacher requires explicit 224/16 preset without dimension overrides")
    if not spec.get("encoder") or not Path(spec["encoder"]).is_file():
        raise ValueError("native teacher requires a local encoder export")
    state = torch.load(spec["encoder"], map_location="cpu", weights_only=True)
    encoder = build_encoder(spec["arch"])
    encoder.load_state_dict(state, strict=True)
    return TeacherBackbone(encoder)
