"""Feature-extraction provider for 12_cmc.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how CMC turns
an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, then read through `get_encoder()` -- the two-branch AlexNet's
  layer-6 (fc6) features of the L and ab branches, concatenated (2 x 2048 =
  4096-d);
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `CMCDataset(mode="val")`): resize + centre crop, then the
  **CIE Lab** conversion (`RGB2Lab`: sRGB -> linear -> XYZ(D65) -> Lab) and a
  per-channel Lab normalisation (LAB_MEAN/LAB_STD, the Lab-gamut extremes). CMC
  trains on Lab, not RGB; there is no ImageNet mean/std here;
- features are the raw encoder output (`extract_features`), *before* the
  probe's mean-centre + L2-normalise. Raw features are what the visualisation
  asked for.

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
    """Return (features, labels, meta): features is (N, 4096) raw encoder
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_cmc")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    encoder = model.get_encoder().to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(encoder, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "alexnet"),
        "image_size": image_size,
        "preprocessing": ("CMC eval: resize + centre crop, CIE Lab conversion "
                          "(RGB2Lab) + per-channel Lab normalisation "
                          "(LAB_MEAN/LAB_STD), no ImageNet normalisation; "
                          "feature is the concatenated layer-6 (fc6) output of "
                          "the L and ab branches (2 x 2048)"),
    }
    return feats, labels, meta
