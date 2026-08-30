"""Frozen spatial backbones for downstream dense tasks.

A downstream task head reads a spatial feature map, so every backbone this package
builds exposes the same two-symbol interface the capture harness uses:

    backbone.forward_features(x) -> Tensor[B, C, h, w]
    backbone.out_channels: int

The backbone is always frozen (eval, `requires_grad = False`). A real run loads a
method's trained `encoder.pt`; the hermetic smoke leaves `encoder` empty and builds
a random tiny backbone, so CI downloads and trains nothing on the backbone.

For Step 2 every method's backbone is the unified ViT-B/16, so one ViT adapter
serves them all (the capture's `_load_step2_vit`); Step-1's diverse backbones and
CLIP's own `VisionTransformer` (no `patch_embed.proj`) get their own kinds as those
tasks are ported. `vit` (timm) is built here; `resnet50` / `clip_vit` are not yet.

**Lineage backbones are discovered, not named** (CLAUDE.md; the anti-pattern
`tests/test_no_hard_coded_methods.py` guards). A Step-1&2 method whose own model
must be reused as a frozen backbone -- MAE is the first, Step-3 item A3:mae --
declares a provider in *its own* directory (`methods/<m>/downstream_backbone.py`,
a module-level `KIND` string and a `build(spec)` function). This shared layer
discovers those providers by structure and dispatches to them, so a new lineage
method registers a backbone kind with no edit here.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = ROOT / "methods"

VIT = "vit"

# The provider file every lineage method drops in its own directory to offer a
# frozen backbone kind. Structural, so this layer names no method.
PROVIDER_FILE = "downstream_backbone.py"


def _provider_kind(path: Path) -> "str | None":
    """The module-level `KIND` string of a backbone provider, else None.

    A `methods/*/downstream_backbone.py` qualifies only when it declares, at
    module level, a `KIND = "..."` string constant *and* a `build` function. The
    AST is read structurally, so a `KIND` inside another name or a string, or a
    nested `build`, never matches -- the too-wide-substring mistake this repo
    keeps a list of."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kind = None
    funcs = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == "KIND"
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    kind = node.value.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
    if kind is not None and "build" in funcs:
        return kind
    return None


def discover_providers(methods_dir: Path = METHODS_DIR) -> "dict[str, Path]":
    """Map each discovered backbone `KIND` to its provider file path.

    Discovers by structure (`_provider_kind`), so a new lineage method joins with
    no edit here. Two methods claiming the same kind is a hard error, never a
    silent last-wins overwrite."""
    found: dict[str, Path] = {}
    for path in sorted(Path(methods_dir).glob("*/" + PROVIDER_FILE)):
        kind = _provider_kind(path)
        if kind is None:
            continue
        if kind in found:
            raise RuntimeError(
                f"two methods provide backbone kind {kind!r}: "
                f"{found[kind].parent.name} and {path.parent.name}")
        found[kind] = path
    return found


_PROVIDERS = discover_providers()
KINDS = (VIT,) + tuple(sorted(_PROVIDERS))


def _load_provider(path: Path):
    """Import a provider module by file path, under a unique per-method name so it
    cannot collide with another method's top-level ``models`` package (the
    cross-method import bug this repo has hit before)."""
    name = "_downstream_provider_" + path.parent.name
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FrozenViTSpatialBackbone(nn.Module):
    """Wrap a timm ViT so its patch tokens become a spatial [B, C, h, w] map."""

    def __init__(self, vit: nn.Module, patch_size: int, num_prefix_tokens: int,
                 out_channels: int):
        super().__init__()
        self.vit = vit
        self.patch_size = int(patch_size)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.out_channels = int(out_channels)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):        # stays frozen; never trains
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision detection/segmentation backbones are called as backbone(x);
        # a bare tensor is wrapped into an OrderedDict({"0": ...}) downstream.
        return self.forward_features(x)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)          # [B, prefix + h*w, D]
        if tokens.ndim != 3:
            raise RuntimeError(
                f"expected ViT tokens [B, N, D], got {tuple(tokens.shape)}")
        patches = tokens[:, self.num_prefix_tokens:, :]
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        if patches.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"token grid mismatch: {patches.shape[1]} patch tokens but "
                f"input {tuple(x.shape[-2:])} at patch {self.patch_size} implies "
                f"{grid_h}x{grid_w}")
        b, _, d = patches.shape
        return patches.transpose(1, 2).reshape(b, d, grid_h, grid_w).contiguous()


def _build_vit(spec: dict) -> FrozenViTSpatialBackbone:
    import timm
    arch = spec.get("arch", "vit_base_patch16_224")
    patch_size = int(spec.get("patch_size", 16))
    # dynamic_img_size interpolates the position embedding, so one ViT serves
    # both a fixed-size probe (ADE20K) and the variable, padded inputs a detection
    # transform produces (COCO). The spatial reshape recomputes the grid from the
    # actual input, so it stays correct at any size.
    kwargs = {"pretrained": False, "num_classes": 0, "dynamic_img_size": True,
              "img_size": int(spec["img_size"]), "patch_size": patch_size}
    for key in ("embed_dim", "depth", "num_heads"):
        if key in spec:
            kwargs[key] = int(spec[key])
    vit = timm.create_model(arch, **kwargs)
    encoder = spec.get("encoder") or ""
    if encoder:
        state = torch.load(encoder, map_location="cpu", weights_only=True)
        missing, unexpected = vit.load_state_dict(state, strict=False)
        # The classifier head is dropped (num_classes=0), so head.* is expected to
        # be unexpected; anything else unexpected means the encoder is not this ViT.
        unexpected = [k for k in unexpected if not k.startswith("head.")]
        if unexpected:
            raise RuntimeError(
                f"encoder.pt carries keys this ViT does not have: {unexpected[:5]}")
    return FrozenViTSpatialBackbone(
        vit, patch_size=patch_size,
        num_prefix_tokens=int(getattr(vit, "num_prefix_tokens", 1)),
        out_channels=int(vit.embed_dim))


def build_frozen_backbone(spec: dict, device: "torch.device") -> nn.Module:
    """Build the frozen spatial backbone named by `spec['kind']`."""
    kind = spec.get("kind")
    if kind == VIT:
        model = _build_vit(spec)
    elif kind in _PROVIDERS:
        # A lineage method reuses its own model, loaded by file path from its own
        # directory to avoid a cross-method `models`-package collision; imported
        # lazily so a base `vit` run pulls none of it.
        model = _load_provider(_PROVIDERS[kind]).build(spec)
    elif kind in ("resnet50", "clip_vit"):
        raise NotImplementedError(
            f"backbone kind {kind!r} is not ported yet; this pilot implements "
            f"{VIT!r} (the unified Step-2 ViT). See docs/DOWNSTREAM.md.")
    else:
        raise ValueError(f"unknown backbone kind {kind!r}; known: {', '.join(KINDS)}")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
