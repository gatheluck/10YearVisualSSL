"""iGPT as the ``igpt`` frozen spatial backbone for the ARSSL harness.

Item A3:igpt of the Step-3 plan (docs/STEP3_PORTING_PLAN.md). Phase A3 wires the
lineage backbones already ported as Step-1&2 methods into the A1 ARSSL harness
(`downstream/arssl.py`) so their Step-3 numbers reproduce here.

**This file lives in the method's own directory on purpose.** The shared
downstream layer (`downstream/spatial_backbones.py`) *discovers* backbone
providers -- it names no method -- and a method that offers a downstream backbone
kind declares it here, next to the model it wraps. The provider contract is
structural: a module-level ``KIND`` string and a ``build(spec)`` function. It
reuses iGPT's own model and its own quantiser, loaded **by file path under unique
module names** (never via ``sys.path``), so they cannot collide with another
method's top-level ``models`` package -- the cross-method import bug this repo has
hit before.

iGPT is the first provider whose input is **not an image tensor**. It is a causal
transformer over discrete colour-cluster tokens, so ``forward_features`` does what
the method's own linear probe does before reading features: resize the input to
the model's token grid, quantise pixels to colour tokens with the clusters the
model was trained on, then read a **middle** transformer layer. The linear probe
mean-pools that layer to one vector; a dense task keeps the per-position features
and reshapes them to ``[B, C, h, w]`` (``IGPT.extract_token_features``, which the
probe's ``extract_features`` mean-pools -- one representation, two readers). So
global-average-pooling this backbone's map equals iGPT's own probe feature.

**Two things the shared ViT backbone schema has no slot for, absorbed here so the
four task runners stay unchanged** (the chosen A3:igpt shape):

* **vocab_size** -- the colour vocabulary. For a real encoder it is *inferred*
  from ``encoder.pt`` (the token-embedding row count minus the SOS row), so it
  can never disagree with the trained model. The hermetic smoke (no encoder) uses
  ``DEFAULT_VOCAB_SIZE``.
* **clusters** -- the colour centres pixels are quantised to. For a real encoder
  they are read from ``clusters.npy`` **beside** ``encoder.pt`` (where the adapter
  writes them, always co-located with the encoder of the same pretrain run); a
  missing file is refused, not silently replaced. The smoke generates a
  deterministic set.

The optional timm-named schema keys map onto iGPT's architecture
(``embed_dim -> n_embd``, ``depth -> n_layer``, ``num_heads -> n_head``); ``arch``
and ``patch_size`` are schema-required but informational for iGPT (one token per
grid cell). A real encoder settles ``embed_dim/depth/n_head`` too: a config that
disagrees with the checkpoint is refused by the strict load below.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KIND = "igpt"

# The colour vocabulary used only when there is no encoder to infer it from (the
# hermetic smoke, which builds a random tiny model anyway). iGPT-S's value.
DEFAULT_VOCAB_SIZE = 512

CLUSTERS_FILE = "clusters.npy"

_METHOD_DIR = Path(__file__).resolve().parent
_IGPT_PATH = _METHOD_DIR / "models" / "igpt.py"
_QUANTIZE_PATH = _METHOD_DIR / "quantize.py"

# iGPT-S defaults, used only when a config omits the (optional) architecture keys.
_DEFAULTS = {"embed_dim": 512, "depth": 24, "num_heads": 8}

_MODULES: dict = {}


def _load_by_path(path: Path, unique_name: str):
    """Import a self-contained module by file path, under a unique name so it can
    never collide with another method's top-level package. Cached."""
    if unique_name in _MODULES:
        return _MODULES[unique_name]
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing; the igpt backbone reuses this method's own code "
            "and cannot be built without it")
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULES[unique_name] = module
    return module


def load_igpt_module():
    """The method's self-contained iGPT model module (the ``IGPT`` class)."""
    return _load_by_path(_IGPT_PATH, "_downstream_igpt_model")


def load_quantize_module():
    """The method's self-contained colour quantiser (``quantize_images``)."""
    return _load_by_path(_QUANTIZE_PATH, "_downstream_igpt_quantize")


def _build_igpt(spec: dict, vocab_size: int):
    """Build an ``IGPT`` at the colour vocabulary and the dimensions the shared
    backbone schema names (its timm-named optional keys map onto iGPT directly)."""
    mod = load_igpt_module()
    return mod.IGPT(
        vocab_size=int(vocab_size),
        img_size=int(spec["img_size"]),
        n_layer=int(spec.get("depth", _DEFAULTS["depth"])),
        n_head=int(spec.get("num_heads", _DEFAULTS["num_heads"])),
        n_embd=int(spec.get("embed_dim", _DEFAULTS["embed_dim"])))


def _smoke_clusters(n: int) -> np.ndarray:
    """A deterministic set of ``n`` colour centres in the [0, 1] range ToTensor
    produces. Used only by the hermetic smoke; seeded so a run is reproducible."""
    return np.random.RandomState(0).uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)


class FrozenIGPTSpatialBackbone(nn.Module):
    """Wrap iGPT so its middle-layer tokens become a ``[B, C, h, w]`` map.

    ``forward_features`` resizes the input to the model's token grid, quantises it
    to colour tokens with the model's clusters, reads the middle transformer layer
    per position, and reshapes the raster-order tokens back to that grid."""

    def __init__(self, model: nn.Module, clusters: np.ndarray, img_size: int,
                 out_channels: int, quantize):
        super().__init__()
        self.model = model
        self.clusters = np.asarray(clusters, dtype=np.float32)
        self.img_size = int(img_size)
        self.out_channels = int(out_channels)
        self._quantize = quantize
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):        # stays frozen; never trains
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        g = self.img_size
        if x.shape[-2:] != (g, g):
            # iGPT's position embedding is fixed to img_size^2, so the input is
            # resized to the token grid (as the probe's transform does).
            x = F.interpolate(x, size=(g, g), mode="bilinear", align_corners=False)
        tokens = self._quantize(x, self.clusters).to(x.device)   # [B, g*g]
        h = self.model.extract_token_features(tokens)            # [B, g*g, D]
        if h.ndim != 3 or h.shape[1] != g * g:
            raise RuntimeError(
                f"expected iGPT tokens [B, {g * g}, D], got {tuple(h.shape)}")
        b, _, d = h.shape
        return h.transpose(1, 2).reshape(b, d, g, g).contiguous()


def build(spec: dict) -> FrozenIGPTSpatialBackbone:
    """Build a frozen iGPT spatial backbone from the backbone ``spec``.

    A real run names an ``encoder`` (iGPT's encoder.pt); the colour vocabulary is
    inferred from it and the clusters are read from ``clusters.npy`` beside it. The
    hermetic smoke leaves ``encoder`` empty: a tiny random iGPT is built and a
    deterministic cluster set generated, so CI downloads and trains nothing. A
    checkpoint that is not this encoder -- alien keys, or missing weights -- is
    refused rather than half-loaded."""
    encoder_path = spec.get("encoder") or ""
    if encoder_path:
        state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        if "token_embed.weight" not in state:
            raise RuntimeError(
                "encoder.pt has no token_embed.weight; the colour vocabulary "
                "cannot be inferred and this is not an iGPT encoder")
        # token_embed is Embedding(vocab_size + 1, n_embd): +1 for the SOS row.
        vocab_size = int(state["token_embed.weight"].shape[0]) - 1
        clusters = _load_clusters_beside(Path(encoder_path), vocab_size)
    else:
        vocab_size = DEFAULT_VOCAB_SIZE
        clusters = _smoke_clusters(vocab_size)

    model = _build_igpt(spec, vocab_size)
    if encoder_path:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(
                "encoder.pt carries keys this iGPT does not have: "
                f"{unexpected[:5]}")
        # The generative head is excluded from encoder.pt, so head.* is expected
        # to be missing; any other missing weight means the encoder is not this
        # model (or the config's dimensions disagree with the checkpoint).
        absent = [k for k in missing if not k.startswith("head.")]
        if absent:
            raise RuntimeError(
                f"encoder.pt is missing encoder weights: {absent[:5]}. The head "
                "is expected to be missing; the representation is not")

    return FrozenIGPTSpatialBackbone(
        model, clusters=clusters, img_size=int(spec["img_size"]),
        out_channels=int(model.token_embed.embedding_dim),
        quantize=load_quantize_module().quantize_images)


def _load_clusters_beside(encoder: Path, vocab_size: int) -> np.ndarray:
    """Read the colour clusters the model was trained on, co-located with its
    encoder (the adapter writes ``clusters.npy`` beside ``encoder.pt``). A missing
    file, or a set whose size does not match the vocabulary, is refused."""
    path = encoder.parent / CLUSTERS_FILE
    if not path.is_file():
        raise RuntimeError(
            f"no {CLUSTERS_FILE} beside {encoder.name} (looked at {path}); the "
            "probe must quantise with the clusters the model was trained on")
    clusters = np.load(path)
    if clusters.ndim != 2 or clusters.shape[1] != 3:
        raise RuntimeError(
            f"{path} is not a [n, 3] cluster table; got shape {clusters.shape}")
    if clusters.shape[0] != vocab_size:
        raise RuntimeError(
            f"{path} has {clusters.shape[0]} clusters but encoder.pt's vocabulary "
            f"is {vocab_size}; they are from different runs")
    return clusters
