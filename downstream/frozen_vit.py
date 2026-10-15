"""Strict frozen export and token plumbing for provider-owned ViT recipes."""

import hashlib
import json
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


def lightness_input(rgb):
    rgb = rgb.clamp(0, 1)
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    y = torch.einsum(
        "nchw,c->nhw", linear, linear.new_tensor((0.2126729, 0.7151522, 0.0721750))
    ).clamp_min(0)
    fy = torch.where(y > 0.008856, y ** (1 / 3), 7.787 * y + 16 / 116)
    return ((116 * fy - 16) / 100).unsqueeze(1)


def lab_inputs(rgb):
    rgb = rgb.clamp(0, 1)
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    matrix = linear.new_tensor(
        (
            (0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041),
        )
    )
    xyz = torch.einsum("nchw,dc->ndhw", linear, matrix)
    xyz = xyz / xyz.new_tensor((0.95047, 1.0, 1.08883))[None, :, None, None]
    f = torch.where(xyz > 216 / 24389, xyz ** (1 / 3), ((24389 / 27) * xyz + 16) / 116)
    l = 116 * f[:, 1] - 16
    return ((l - 50) / 50).unsqueeze(1), torch.stack(
        (500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])), 1
    ) / 128


def tokens(encoder, pixels):
    from timm.layers import resample_abs_pos_embed

    grid = tuple(size // 16 for size in pixels.shape[-2:])
    # Projection directly avoids mutating timm's strict_img_size during forward.
    x = encoder.patch_embed.proj(pixels).flatten(2).transpose(1, 2)
    if hasattr(encoder.patch_embed, "norm"):
        x = encoder.patch_embed.norm(x)
    x = torch.cat((encoder.cls_token.expand(x.shape[0], -1, -1), x), 1)
    pos = resample_abs_pos_embed(
        encoder.pos_embed,
        grid,
        old_size=(14, 14),
        num_prefix_tokens=1,
        interpolation="bicubic",
        antialias=True,
    )
    x = encoder.pos_drop(x + pos)
    for name in ("patch_drop", "norm_pre"):
        if hasattr(encoder, name):
            x = getattr(encoder, name)(x)
    for block in encoder.blocks:
        x = block(x)
    return encoder.norm(x)


class FrozenViT(nn.Module):
    classifier_init_std = 0.01
    native_pyramid_style = "transposed"

    def __init__(self, encoder, transform=None):
        super().__init__()
        self.encoder = encoder
        self.transform = transform
        branches = list(encoder) if isinstance(encoder, nn.ModuleList) else [encoder]
        self.out_channels = self.global_channels = sum(
            cast(torch.Tensor, m.cls_token).shape[-1] for m in branches
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def _tokens(self, images, *, detection=False):
        if (
            images.ndim != 4
            or images.shape[0] == 0
            or images.shape[1] != 3
            or min(images.shape[-2:]) < 16
            or any(s % 16 for s in images.shape[-2:])
        ):
            raise ValueError("expected nonempty RGB batch on a stride-16 grid")
        if self.transform is not None:
            if not detection:
                images = (
                    images
                    * images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
                    + images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
                )
            inputs = self.transform(images.float())
        else:
            inputs = images
        branches = (
            list(self.encoder)
            if isinstance(self.encoder, nn.ModuleList)
            else [self.encoder]
        )
        inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        values = []
        for branch, pixels in zip(branches, inputs, strict=True):
            param = next(branch.parameters())
            values.append(
                tokens(
                    branch, pixels.to(device=param.device, dtype=param.dtype)
                ).float()
            )
        return torch.cat(values, -1)

    def _spatial(self, images, *, detection=False):
        patches = self._tokens(images, detection=detection)[:, 1:]
        return patches.transpose(1, 2).reshape(
            images.shape[0],
            self.out_channels,
            images.shape[-2] // 16,
            images.shape[-1] // 16,
        )

    def forward(self, images):
        return self.forward_features(images)

    def forward_features(self, images):
        return self._spatial(images)

    def forward_detection_features(self, images):
        return self._spatial(images, detection=True)

    def detection_normalization(self):
        return (
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            if self.transform is not None
            else ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    def classification_features(self, images, *, adaptation, video=False):
        if adaptation != "frozen":
            raise ValueError("only frozen evaluation is verified")
        if video:
            if images.ndim != 5 or images.shape[1] == 0:
                raise ValueError("video requires nonempty [B,T,3,H,W]")
            cls = self._tokens(images.flatten(0, 1))[:, 0]
            return cls.reshape(images.shape[0], images.shape[1], -1).mean(1)
        return F.normalize(self._tokens(images)[:, 0], dim=-1)


def validate_state(encoder, state):
    expected = encoder.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("encoder keys must exactly match the declared preset")
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("encoder tensor shape, dtype or finiteness mismatch")
    encoder.load_state_dict(state, strict=True)


def load_backbone(spec, *, kind, factory, transform=None):
    if (
        spec.get("img_size") != 224
        or spec.get("patch_size") != 16
        or any(k in spec for k in ("embed_dim", "depth", "num_heads"))
    ):
        raise ValueError("native 224/16 preset required without geometry overrides")
    if not spec.get("encoder") or not Path(spec["encoder"]).is_file():
        raise ValueError("local versioned encoder export required")
    blob = torch.load(spec["encoder"], map_location="cpu", weights_only=True)
    if (
        not isinstance(blob, dict)
        or type(blob.get("frozen_encoder_version")) is not int
        or blob["frozen_encoder_version"] != 1
        or blob.get("kind") != kind
        or blob.get("arch") != spec.get("arch")
    ):
        raise ValueError("encoder export identity mismatch")
    encoder = factory(spec.get("arch"))
    validate_state(encoder, blob.get("state_dict"))
    return FrozenViT(encoder, transform)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_native(
    *,
    provider,
    source,
    sha256,
    arch,
    state_key,
    epoch_key,
    epoch_value,
    module_prefix,
    out,
):
    """Explicit literal epoch identity; never infer completed epochs from filenames."""
    out = Path(out)
    if out.exists():
        raise FileExistsError("choose a new output directory")
    if (
        state_key not in ("state_dict", "model")
        or epoch_key not in ("epoch", "completed_epoch")
        or type(epoch_value) is not int
        or epoch_value < 0
        or type(module_prefix) is not bool
    ):
        raise ValueError(
            "explicit state key, literal nonnegative epoch and prefix required"
        )
    if digest(source) != sha256:
        raise ValueError("source sha256 mismatch")
    from downstream.native_checkpoint import load_native

    blob = load_native(source)
    if (
        not isinstance(blob, dict)
        or type(blob.get(epoch_key)) is not int
        or blob[epoch_key] != epoch_value
    ):
        raise ValueError("native literal epoch identity mismatch")
    raw = blob.get(state_key)
    if not isinstance(raw, dict) or any(not isinstance(k, str) for k in raw):
        raise ValueError("native state mapping required")
    encoder = provider.build_encoder(arch)
    expected = encoder.state_dict()
    state = {}
    selected = set()
    for key in expected:
        if isinstance(encoder, nn.ModuleList):
            index, rest = key.split(".", 1)
            native = provider.NATIVE_PREFIXES[int(index)] + rest
        else:
            native = provider.NATIVE_PREFIXES[0] + key
        native = ("module." if module_prefix else "") + native
        if native not in raw:
            raise ValueError("native encoder tensor missing")
        selected.add(native)
        state[key] = raw[native]
    discarded = tuple(
        ("module." if module_prefix else "") + prefix
        for prefix in provider.NATIVE_DISCARDED_PREFIXES
    )
    if any(key not in selected and not key.startswith(discarded) for key in raw):
        raise ValueError("unexpected native encoder parameters")
    validate_state(encoder, state)
    if digest(source) != sha256:
        raise ValueError("source changed during import")
    out.mkdir(parents=True, exist_ok=False)
    dest = out / "encoder.pt"
    torch.save(
        {
            "frozen_encoder_version": 1,
            "kind": provider.KIND,
            "arch": arch,
            "state_dict": state,
        },
        dest,
    )
    record = {
        "kind": provider.KIND,
        "arch": arch,
        "source_sha256": sha256,
        "encoder_sha256": digest(dest),
        "epoch_key": epoch_key,
        "epoch_value": epoch_value,
        "state_key": state_key,
        "module_prefix": module_prefix,
        "tensor_count": len(state),
        "validation": "encoder import only; historical score attribution unverified",
    }
    (out / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None):
    import argparse

    from downstream import spatial_backbones as sb

    parser = argparse.ArgumentParser(
        description="Import a provider's native frozen encoder"
    )
    for name in ("kind", "source", "sha256", "arch", "state-key", "epoch-key", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epoch-value", required=True, type=int)
    parser.add_argument("--module-prefix", action="store_true")
    args = vars(parser.parse_args(argv))
    provider = sb._load_provider(sb.discover_providers()[args.pop("kind")])
    import_native(provider=provider, **args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
