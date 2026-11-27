"""Frozen native teacher exports from DINO design/scaling experiments."""
import importlib.util
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from downstream.contract import sha256_file

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


def import_native(*, source, sha256, arch, epoch, out):
    """Export a verified native teacher without carrying private training metadata."""
    import json

    out = Path(out)
    if out.exists():
        raise FileExistsError("output already exists; choose a new directory")
    if arch not in PRESETS or type(epoch) is not int or epoch < 1:
        raise ValueError("explicit supported architecture and positive native epoch required")
    if sha256_file(source) != sha256:
        raise ValueError("source sha256 mismatch")
    blob = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(blob, dict) or type(blob.get("epoch")) is not int
            or blob["epoch"] != epoch):
        raise ValueError("native checkpoint epoch does not match the explicit epoch")
    cfg = blob.get("cfg")
    model = cfg.get("model") if isinstance(cfg, dict) else None
    if not isinstance(model, dict) or model.get("arch") != arch:
        raise ValueError("native checkpoint must declare the matching model architecture")
    dim, depth, heads = PRESETS[arch]
    expected_config = {"embed_dim": dim, "depth": depth, "num_heads": heads, "patch_size": 16,
                       "mlp_ratio": 4., "n_register_tokens": 4, "use_rope": True,
                       "rope_base": 100., "rope_rescale_coords": 2.}
    if any(model.get(key, value) != value for key, value in expected_config.items()):
        raise ValueError("native model configuration differs from the evaluation preset")
    raw = blob.get("teacher_state_dict")
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("native teacher must be a string-keyed mapping")
    head_names = ("dino_head", "ibot_head", "prototypes", "dino_prototypes",
                  "ibot_prototypes", "shared_mlp", "token_affine_cls",
                  "token_affine_patch", "gram_head")
    state = {}
    for key, value in raw.items():
        if key.startswith("backbone."):
            state[key[len("backbone."):]] = value
        elif not any(key == head or key.startswith(head + ".") for head in head_names):
            raise ValueError("unknown non-backbone teacher namespace")
    encoder = build_encoder(arch)
    expected = encoder.state_dict()
    if set(state) != set(expected):
        raise ValueError("native teacher keys do not exactly match the preset")
    for key, value in state.items():
        target = expected[key]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or not torch.isfinite(value).all()):
            raise ValueError("native teacher tensor shape, dtype or finiteness mismatch")
    encoder.load_state_dict(state, strict=True)
    if sha256_file(source) != sha256:
        raise ValueError("source changed during import")
    out.mkdir(parents=True, exist_ok=False)
    dest = out / "encoder.pt"
    torch.save(encoder.state_dict(), dest)
    record = {"source_sha256": sha256, "encoder_sha256": sha256_file(dest), "arch": arch,
              "epoch": epoch, "teacher_key": "teacher_state_dict", "tensor_count": len(state),
              "validation": "strict teacher export; historical score attribution unverified"}
    (out / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Export native DINO teacher weights for frozen evaluation")
    for name in ("source", "sha256", "arch", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epoch", required=True, type=int)
    import_native(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
