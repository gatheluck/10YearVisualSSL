"""The BEiT visual tokenizer that produces the MIM targets.

Two implementations behind ``build_tokenizer``:

  - ``DALLETokenizer`` (a real run): the frozen OpenAI DALL-E dVAE encoder. The
    encoder.pkl is a pickled ``dall_e`` ``nn.Module``, so the ``dall_e`` package
    is imported **lazily** (only when a real checkpoint is loaded) to unpickle
    it. That package is the pinned upstream under ``third_party/dall_e``
    (imported through PYTHONPATH, never copied or installed -- provenance.json ->
    upstream); the weights are a hash-pinned download (provenance.json ->
    tokenizer_artifact). A 112x112 image becomes 14x14 = 196 discrete tokens in
    [0, vocab_size).

  - ``RandomDVAETokenizer`` (the hermetic smoke): a fixed random conv that maps a
    token image to token indices with the same shape. Torch-only -- no ``dall_e``
    and no download -- so CI stays hermetic; the target tokens are meaningless,
    only the pipeline is exercised. Frozen; deterministic under a seeded run.

The tokenizer only produces the MIM *targets*; the representation this port ships
(encoder.pt) is the trained BEiT backbone, not the tokenizer.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ENCODER_URL = "https://cdn.openai.com/dall-e/encoder.pkl"
_ENCODER_CACHE = os.path.expanduser("~/.cache/dall_e/encoder.pkl")
# The pinned openai/DALL-E upstream: third_party/dall_e at the repository root
# (methods/24_beit/models/dvae_tokenizer.py -> parents[3] is the repo root).
_DALLE_SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "dall_e"


def _map_pixels(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] pixel values to DALL-E's input range [0.1, 0.9]."""
    return 0.8 * x + 0.1


def _load_dalle_encoder(ckpt: str, device: torch.device) -> nn.Module:
    """Unpickle the OpenAI DALL-E encoder from a local .pkl (the hash-pinned
    download). Imports ``dall_e`` lazily from the pinned ``third_party/dall_e``
    submodule -- required only for a real run."""
    if _DALLE_SUBMODULE.is_dir() and str(_DALLE_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(_DALLE_SUBMODULE))
    try:
        import dall_e.encoder  # noqa: F401  # required for the trusted pickle load
    except ImportError as e:
        raise ImportError(
            "the DALL-E package is required to load the dVAE tokenizer for a "
            "real run. It is the pinned submodule at third_party/dall_e and has "
            "its own dependencies (e.g. requests, attr): run `git submodule "
            "update --init third_party/dall_e` and install its requirements. The "
            "hermetic smoke uses a random tokenizer and needs neither the "
            "submodule nor the download.") from e

    def _trusted_load(path):
        # OpenAI's encoder.pkl is a pickled nn.Module; torch>=2.6 defaults to
        # weights_only=True, which cannot load it.
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    with open(ckpt, "rb") as f:
        return _trusted_load(io.BytesIO(f.read()))


class DALLETokenizer(nn.Module):
    """The frozen DALL-E dVAE encoder: a token image -> (B, N) int64 tokens."""

    def __init__(self, ckpt: str, device: torch.device = torch.device("cpu"),
                 input_is_mapped: bool = False):
        super().__init__()
        self._device = device
        self.input_is_mapped = input_is_mapped
        self._encoder = _load_dalle_encoder(ckpt, device)
        for p in self._encoder.parameters():
            p.requires_grad = False
        self._encoder.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._device, dtype=torch.float32)
        x_mapped = x if self.input_is_mapped else _map_pixels(x)
        z_logits = self._encoder(x_mapped)             # (B, vocab, h, w)
        tokens = z_logits.argmax(dim=1)                # (B, h, w)
        return tokens.view(tokens.size(0), -1).long()  # (B, N)


class RandomDVAETokenizer(nn.Module):
    """A random frozen tokenizer for the hermetic smoke (no dall_e, no download).
    A stride-8 conv maps the token image to (B, N) token indices in
    [0, vocab_size), matching the real dVAE's shape (112 -> 14x14 = 196)."""

    def __init__(self, vocab_size: int, stride: int = 8,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self._device = device
        self.proj = nn.Conv2d(3, vocab_size, kernel_size=stride, stride=stride)
        self.to(device)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_logits = self.proj(x.to(self._device, dtype=torch.float32))
        tokens = z_logits.argmax(dim=1)
        return tokens.view(tokens.size(0), -1).long()


def build_tokenizer(vocab_size: int, ckpt: str = "", stride: int = 8,
                    device: torch.device = torch.device("cpu"),
                    input_is_mapped: bool = True):
    """The real DALL-E dVAE when ``ckpt`` is given; a random tokenizer (the
    hermetic smoke) when it is empty."""
    if ckpt:
        return DALLETokenizer(ckpt, device=device, input_is_mapped=input_is_mapped)
    return RandomDVAETokenizer(vocab_size, stride=stride, device=device)
