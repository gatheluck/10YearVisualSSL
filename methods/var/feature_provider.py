"""Feature-extraction provider for var.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how VAR turns
an image into a vector stays in one place.

**What VAR's feature is, measured -- not the name** (`evaluate_linear_var.encode`,
and docs/EVAL_DOWNLOAD.md section 2): the probed representation is the **VQVAE
tokeniser's encoder** output, global-average-pooled to `Cvae` dims -- *not* the
VAR transformer this port trains in step 1, and *not* `encoder.pt`. So the
`encoder_path` this provider is handed is the **pretrained VQVAE tokeniser
checkpoint** (the pinned download `vae_ch160v4096z32.pth`), the same file
`linear_eval`'s `vqvae_ckpt` names -- not a trained encoder. The feature
dimension is therefore `Cvae` (32 for the real tokeniser).

The feature is the **raw** encoder output -- `vae.encoder(x).mean([2, 3])` --
*before* the probe's mean-centre + L2-normalise
(`evaluate_linear_var.normalize_features`). Raw features are what the
visualisation asked for.

Two method pieces are reused so VAR's knowledge is not duplicated:

- the VQVAE is built and its checkpoint strict-loaded through this method's own
  ``downstream_backbone.build``. That builder infers the VQVAE architecture from
  the checkpoint (so a config can never disagree with the trained tokeniser) and
  builds **hermetically** -- it snapshots and restores ``sys.path`` /
  ``sys.modules`` / the torch RNG / ``torch.nn``'s ``reset_parameters`` (undone
  even on error), so the VAR upstream's global ``reset_parameters`` no-op cannot
  leak into the shared in-process test suite (see
  ``downstream_backbone._hermetic_build`` and ``train_pretrain_var``'s
  ``restore_default_init``). Reusing it keeps that hermeticity in one place;
- images and pooling go through ``evaluate_linear_var``'s own ``_build_loader``
  and ``extract_features`` (which call ``encode`` -- ``vae.encoder(x).mean([2,3])``),
  so this provider and the linear probe read the **same** feature by the **same**
  code (one representation, two readers).

Method modules are loaded **by file path under unique names**, never as bare
``import downstream_backbone``: six methods ship a module named
``downstream_backbone`` and ``sys.modules`` keeps only the first, so a bare
import would resolve another method's file in the shared test interpreter. The
driver runs each method in an isolated subprocess in production, but the provider
stays collision-safe so its in-process test does not corrupt the suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent
METHOD_NAME = METHOD_DIR.name


def _load_by_path(name: str, filename: str):
    """Import one of this method's modules by file path under a unique, private
    name, so it cannot collide with another method's identically named module."""
    spec = importlib.util.spec_from_file_location(name, METHOD_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, Cvae) raw VQVAE-encoder
    output (global-average-pooled), labels is (N,) ImageFolder class indices, and
    meta describes the run. `encoder_path` names the VQVAE tokeniser checkpoint,
    not a trained encoder (VAR's probed representation is the tokeniser)."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))

    backbone = _load_by_path("_var_provider_downstream_backbone",
                             "downstream_backbone.py")
    ev = _load_by_path("_var_provider_eval", "evaluate_linear_var.py")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    # Build the VQVAE tokeniser from its checkpoint -- hermetically: the builder
    # infers the architecture from the checkpoint, strict-loads it, and restores
    # every global it touches (imports / RNG / reset_parameters). `bb.vae` is the
    # tokeniser; `vae.encoder` is the representation the probe reads.
    bb = backbone.build({"encoder": str(encoder_path)})
    bb.to(device)
    vae = bb.vae
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(vae, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": "var_vqvae",
        "image_size": image_size,
        "feature_source": (
            "VQVAE tokeniser encoder output, global-average-pooled to Cvae dims "
            "(evaluate_linear_var.encode); NOT the VAR transformer, NOT "
            "encoder.pt -- encoder_path is the VQVAE tokeniser checkpoint"),
        "preprocessing": (
            "VAR linear eval: resize + centre crop, ToTensor, normalise to "
            "[-1, 1] (mean/std 0.5)"),
    }
    return feats, labels, meta
