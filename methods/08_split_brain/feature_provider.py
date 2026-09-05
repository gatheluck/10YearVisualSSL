"""Feature-extraction provider for 08_split_brain.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how
Split-Brain turns an image into a vector stays in one place:

- the frozen model is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`; that returns the whole two-branch model, and this method's
  eval feature is the model itself (its `extract_features(l, ab)` concatenates
  the two branch encoders' spatially-averaged features), so -- unlike the
  ResNet backbones -- the model is passed **directly** to `extract_features`,
  not a `get_encoder()`. This mirrors the eval main exactly;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `SplitBrainProbeDataset`: resize + centre crop, then
  **RGB -> CIE Lab** split into the L (1-channel) and ab (2-channel) inputs the
  two cross-channel branches read). Split-Brain trains on Lab, not on RGB with
  ImageNet normalisation;
- features are the raw concatenated encoder output (`extract_features`),
  *before* the probe's mean-centre + L2-normalise. Raw features are what the
  visualisation asked for.

Imports are bare module names resolved through this method's directory, as the
adapter itself does. That is safe because the driver runs each method in
isolation; do not rely on this module and another method's `adapter`/`models`
coexisting in one interpreter.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent
METHOD_NAME = METHOD_DIR.name


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, 512) raw concatenated
    branch-encoder output, labels is (N,) ImageFolder class indices, meta
    describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_split_brain")

    cfg = _load_config()
    train = cfg["train"]
    crop_size = int(train["crop_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, crop_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "alexnet"),
        "image_size": crop_size,
        "preprocessing": ("Split-Brain eval: resize + centre crop, then "
                          "RGB -> CIE Lab split into L (1ch) and ab (2ch) "
                          "branch inputs; concatenated branch features, "
                          "no ImageNet normalisation"),
    }
    return feats, labels, meta
