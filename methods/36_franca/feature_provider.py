"""Feature-extraction provider for 36_franca.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how Franca
turns an image into a vector stays in one place.

**What Franca's feature is, measured -- not the name** (`evaluate_linear_franca`,
and this method's `real_run_smoke.json` / docs/EVAL_DOWNLOAD.md): Franca is the
EVAL_DOWNLOAD eval-only shape. The comparable representation is the **official
pretrained Franca ViT-B/14 In21K backbone's CLS token**
(`forward_features(x)["x_norm_clstoken"]`), frozen -- *not* the from-scratch
unified Step-2 encoder this port can also train. So the `encoder_path` this
provider is handed is the **pretrained Franca checkpoint** (the pinned download
`franca_vitb14_In21K.pth`, obtained with `bin/fetch-weights.py`), the same file
`configs/linear_eval.yaml`'s `ckpt`/`${FRANCA_CKPT}` names -- not a trained
`encoder.pt`. The feature is one 768-d (ViT-B embed_dim) vector per image; it is
the **CLS token as is**, never a mean over patch tokens (`extract_cls` mean-pools
only when the chosen `feature_key` output is a token grid, and `x_norm_clstoken`
is already a [B, D] vector).

The feature is the **raw** encoder output (`extract_features`), *before* the
probe's mean-centre + L2-normalise (`normalize_features`). Raw features are what
the visualisation asked for.

Two method pieces are reused so Franca's knowledge is not duplicated:

- the frozen backbone is built from the pinned upstream under
  `third_party/franca` by this method's own ``evaluate_linear_franca.build_model``
  -- the same builder the linear probe uses -- which imports
  ``franca.hub.backbones`` and lets the upstream do its own checkpoint key
  handling (it reads ``state_dict["teacher"]`` and strips the
  ``module.``/``backbone.`` prefixes). `ckpt` empty would build a random backbone
  (the hermetic smoke); here `ckpt` is set to `encoder_path`, the pretrained
  checkpoint;
- images and pooling go through ``evaluate_linear_franca``'s own
  ``_build_loader`` and ``extract_features`` (which call ``extract_cls``), so this
  provider and the linear probe read the **same** feature by the **same** code
  (one representation, two readers).

The eval module is loaded **by file path under a unique, private name** rather
than as a bare ``import evaluate_linear_franca``, so it cannot resolve another
method's identically named module in the shared test interpreter. The driver runs
each method in an isolated subprocess in production, but the provider stays
collision-safe so its in-process test does not corrupt the suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent
METHOD_NAME = METHOD_DIR.name
ROOT = METHOD_DIR.parent.parent
UPSTREAM_DIR = ROOT / "third_party" / "franca"


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
    """Return (features, labels, meta): features is (N, 768) raw encoder output
    (the official Franca ViT-B/14 CLS token), labels is (N,) ImageFolder class
    indices, meta describes the run. `encoder_path` names the pretrained Franca
    checkpoint (the pinned download), not a trained encoder.pt."""
    import torch

    ev = _load_by_path("_franca_provider_eval", "evaluate_linear_franca.py")

    cfg = _load_config()
    train = dict(cfg["train"])
    train["ckpt"] = str(encoder_path)      # probe the pretrained backbone
    resolution = int(train["resolution"])
    dev = torch.device(device)

    # Build the frozen Franca backbone from the pinned upstream, the same builder
    # the linear probe uses; it loads the checkpoint (teacher-format) itself.
    model = ev.build_model(UPSTREAM_DIR, train, dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, resolution, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, train, dev)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("name", "franca_vitb14"),
        "image_size": resolution,
        "feature_source": (
            "official pretrained Franca ViT-B/14 In21K CLS token "
            "(forward_features -> x_norm_clstoken), frozen; the CLS vector as is, "
            "NOT a mean over patch tokens; NOT the from-scratch Step-2 encoder -- "
            "encoder_path is the pretrained Franca checkpoint"),
        "preprocessing": (
            "Franca eval: bicubic resize to resolution/0.875 + centre crop to "
            "resolution, [0,1], ImageNet mean/std; feature is the CLS token"),
    }
    return feats, labels, meta
