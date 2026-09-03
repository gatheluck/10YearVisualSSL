"""VAR as the ``var_vqvae`` frozen spatial backbone for the ARSSL harness.

Item A3:var of the Step-3 plan (docs/STEP3_PORTING_PLAN.md). Phase A3 wires the
lineage backbones already ported as Step-1&2 methods into the A1 ARSSL harness
(`downstream/arssl.py`) so their Step-3 numbers reproduce here.

**This file lives in the method's own directory on purpose.** The shared
downstream layer (`downstream/spatial_backbones.py`) *discovers* backbone
providers -- it names no method -- and a method that offers a downstream backbone
kind declares it here, next to the model it wraps. The provider contract is
structural: a module-level ``KIND`` string and a ``build(spec)`` function. It
reuses VAR's own tokeniser builder, loaded **by file path under a unique module
name** (never via ``sys.path``), so it cannot collide with another method's
top-level ``models`` package -- the cross-method import bug this repo has hit
before (the VAR upstream defines a ``models`` package; its own
``train_pretrain_var._load_upstream`` resolves that collision-safely).

**What VAR's representation is, measured -- not the name** (docs/EVAL_DOWNLOAD.md
section 2, and the method's own ``evaluate_linear_var.py``): the probed feature is
the **VQVAE tokeniser's encoder** output, global-average-pooled -- *not* the VAR
transformer that step 1 trains, and *not* ``encoder.pt``. So for VAR the backbone
spec's ``encoder`` names the **VQVAE tokeniser checkpoint** (the pinned download
``vae_ch160v4096z32.pth``), not a trained encoder.

**How the map is read.** ``vae.encoder(x)`` already returns a
``[B, Cvae, H/16, W/16]`` spatial map -- the VQGAN encoder is fully convolutional
and downsamples by 16 -- so, unlike the ViT providers, there is no token grid to
reshape: the provider returns that map directly. The method's own linear probe
reads exactly this, global-average-pooled (``evaluate_linear_var.encode``), so
pooling this map reproduces VAR's own probe feature (one representation, two
readers). Being fully convolutional, it also serves the variable, padded inputs a
detection transform produces -- no fixed ``img_size``.

**What the shared ViT schema has no slot for, absorbed here** (the iGPT pattern):
the VQVAE architecture (its latent width ``Cvae``, vocabulary ``V`` and base width
``ch``) is **inferred from the checkpoint**, so a config can never disagree with
the trained tokeniser. ``arch`` / ``img_size`` / ``patch_size`` are schema-required
but informational for VAR (``patch_size`` 16 merely matches the real stride). The
hermetic smoke leaves ``encoder`` empty and builds a tiny random VQVAE, so CI
downloads and trains nothing. A checkpoint that is not this tokeniser -- a missing
inference weight, a missing tokeniser weight, or an alien key -- is refused rather
than half-loaded.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

KIND = "var_vqvae"

_METHOD_DIR = Path(__file__).resolve().parent
_TRAIN_PATH = _METHOD_DIR / "train_pretrain_var.py"
_UNIQUE_NAME = "_downstream_var_train"
_TRAIN_MOD = None

# The keys the VQVAE architecture is inferred from, and what each fixes. Read from
# the checkpoint (the iGPT pattern) so the rebuilt tokeniser can never disagree
# with the trained one. Their absence means the checkpoint is not this tokeniser.
_CH_KEY = "encoder.conv_in.weight"        # [ch, 3, 3, 3]      -> base width ch
_CVAE_KEY = "quant_conv.weight"           # [Cvae, Cvae, k, k] -> latent width Cvae
_V_KEY = "quantize.embedding.weight"      # [V, Cvae]          -> vocabulary V

# The finest-scale pyramid VAR is trained on (the VAR-d16 recipe,
# methods/var/configs/linear_eval.yaml). It drives the tokeniser's quantiser and
# the (unused) VAR transformer, not the encoder we read, and the VQVAE's shared
# residual blocks do not depend on its length -- so a real checkpoint and the tiny
# smoke both load against it.
_PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

# The VAR transformer `build_vae_var` builds alongside the tokeniser is never read
# here (the representation is the VQVAE encoder), so it is built at minimal depth
# to spend no memory on it. `num_classes` likewise only sizes the unused VAR.
_VAR_DEPTH = 1
_VAR_NUM_CLASSES = 1000

# The tiny random VQVAE the hermetic smoke builds: small enough to build and run
# on a CPU with no download, exercising the pipeline only. ``ch`` must be a
# multiple of 32 -- the VQGAN encoder normalises every stage with a 32-group
# GroupNorm, so a smaller base width cannot be grouped -- so this is the smallest
# admissible width, not an arbitrary one.
_SMOKE_CH = 32
_SMOKE_CVAE = 4
_SMOKE_V = 16


def _load_var_train():
    """Import VAR's own training module by file path, under a unique name so its
    (and the upstream's) ``models`` package cannot collide with another method's.
    It owns ``build_vqvae`` -- the one place that builds the tokeniser through the
    pinned upstream ``build_vae_var`` -- which is reused here. Cached."""
    global _TRAIN_MOD
    if _TRAIN_MOD is not None:
        return _TRAIN_MOD
    if not _TRAIN_PATH.is_file():
        raise RuntimeError(
            f"the VAR training module is missing at {_TRAIN_PATH}; var_vqvae "
            "reuses this method's own tokeniser builder and cannot be built "
            "without it")
    spec = importlib.util.spec_from_file_location(_UNIQUE_NAME, _TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _TRAIN_MOD = module
    return module


def _infer_arch(state: dict) -> dict:
    """The VQVAE architecture, inferred from a tokeniser checkpoint's own weights.

    A checkpoint missing any of these weights is not this tokeniser and is refused
    here (naming the weight), rather than being built against a guessed shape."""
    for key in (_CH_KEY, _CVAE_KEY, _V_KEY):
        if key not in state:
            raise RuntimeError(
                f"the checkpoint has no {key}; it is not a VAR VQVAE tokeniser, so "
                "its architecture cannot be inferred")
    return _arch(ch=int(state[_CH_KEY].shape[0]),
                 cvae=int(state[_CVAE_KEY].shape[0]),
                 vocab=int(state[_V_KEY].shape[0]))


def _arch(ch: int, cvae: int, vocab: int) -> dict:
    """The `train`-style architecture dict `build_vqvae`/`model_kwargs` expect."""
    return {"patch_nums": _PATCH_NUMS, "vocab_size": vocab, "Cvae": cvae,
            "ch": ch, "num_classes": _VAR_NUM_CLASSES, "depth": _VAR_DEPTH,
            "shared_aln": False, "attn_l2_norm": True}


class FrozenVARSpatialBackbone(nn.Module):
    """Wrap VAR's VQVAE so its encoder feature map is the ``[B, C, h, w]`` map.

    ``forward_features`` returns ``vae.encoder(x)`` directly -- already a spatial
    map (the VQGAN encoder is fully convolutional, stride 16). Global-average-
    pooling it equals VAR's own probe feature (``evaluate_linear_var.encode``)."""

    def __init__(self, vae: nn.Module, out_channels: int):
        super().__init__()
        self.vae = vae
        self.out_channels = int(out_channels)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):        # stays frozen; never trains
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vae.encoder(x)             # [B, Cvae, H/16, W/16]
        if feat.ndim != 4 or feat.shape[1] != self.out_channels:
            raise RuntimeError(
                f"expected a [B, {self.out_channels}, h, w] VQVAE feature map, got "
                f"{tuple(feat.shape)}")
        return feat.contiguous()


@contextlib.contextmanager
def _hermetic_build():
    """Restore the global state building through the provider touches -- the
    imports and the RNG -- so a backbone provider stays a pure builder.

    Building the tokeniser has global side effects that would leak into the other
    methods sharing this process (the downstream harness and the test suite hold
    every method at once):

    * **Imports.** ``build_vqvae`` reaches VAR's tokeniser through
      ``train_pretrain_var``'s collision-safe upstream loader, which -- correctly,
      for the adapter's own process -- leaves ``third_party/var`` first on
      ``sys.path`` and rebinds upstream top-level names (``models``, ``utils``,
      ...) under the shared keys, dropping any non-upstream module already cached
      there. Each ViT method imports its *own* ``models`` package
      (``from models import ...``); left changed, a later method would resolve
      VAR's and come out not round-tripping. Newly bound keys are dropped and
      keys the loader replaced are restored to the objects they held before.

    * **The RNG.** The hermetic smoke draws seeded-normal weights for the tiny
      VQVAE, which advances the global torch RNG. A later test that draws an
      unseeded random input would then see a different draw -- enough to tip an
      already RNG-fragile assertion into a NaN it would not otherwise hit.

    The **third** side effect, the upstream's global ``reset_parameters`` no-op,
    is undone by ``build_vqvae`` itself (``train_pretrain_var.restore_default_init``
    wraps the one ``build_vae_var`` call), so the provider inherits that undo and
    does not repeat it here -- one implementation of that rule, shared by the
    pretraining, linear_eval and downstream paths.

    The built VQVAE's forward needs none of this (it is a plain convolutional
    encoder), so both are snapshotted and restored here -- even if the strict load
    raises -- keeping the provider free of observable global side effects."""
    path_before = list(sys.path)
    mods_before = dict(sys.modules)
    rng_before = torch.get_rng_state()
    cuda_rng_before = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        sys.path[:] = path_before
        for name in list(sys.modules):
            if name not in mods_before:
                del sys.modules[name]
        for name, module in mods_before.items():
            sys.modules[name] = module
        torch.set_rng_state(rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)


def build(spec: dict) -> FrozenVARSpatialBackbone:
    """Build a frozen VAR (VQVAE) spatial backbone from the backbone ``spec``.

    A real run names the pretrained VQVAE tokeniser as ``encoder``; its
    architecture is inferred from the checkpoint and its weights strict-loaded. The
    hermetic smoke leaves ``encoder`` empty and a tiny random VQVAE is built, so CI
    downloads and trains nothing. The tokeniser is built through the method's own
    ``build_vqvae`` (the pinned upstream ``build_vae_var``); a checkpoint that is
    not this tokeniser -- a missing or alien weight -- is refused, not half-loaded.
    """
    train = _load_var_train()
    encoder_path = spec.get("encoder") or ""
    if encoder_path:
        state = torch.load(encoder_path, map_location="cpu", weights_only=True)
        arch = _infer_arch(state)
    else:
        state = None
        arch = _arch(ch=_SMOKE_CH, cvae=_SMOKE_CVAE, vocab=_SMOKE_V)

    # build_vqvae builds the tokeniser (and the unused VAR) through the pinned
    # upstream; passing no checkpoint gives seeded random weights (the smoke). The
    # real checkpoint is loaded below so its keys can be checked before it is used.
    # The upstream import is made hermetic: it must not leave VAR's ``models`` on
    # the shared import path for the other methods sharing this process.
    with _hermetic_build():
        vae, _var = train.build_vqvae(arch, None, torch.device("cpu"))

    if state is not None:
        result = vae.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                "the checkpoint carries keys this VQVAE does not have: "
                f"{result.unexpected_keys[:5]}")
        if result.missing_keys:
            raise RuntimeError(
                f"the checkpoint is missing tokeniser weights: "
                f"{result.missing_keys[:5]}; it is not this VQVAE")

    return FrozenVARSpatialBackbone(vae, out_channels=int(vae.Cvae))
