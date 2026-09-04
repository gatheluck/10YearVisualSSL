"""Feature-extraction provider for 11_cpc.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how visual
CPC 2018 turns an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, then read through `get_encoder()` (the patch encoder,
  grid-averaged to one `avg_z`, `z_dim`-d, per image);
- images go through the method's own deterministic val pipeline
  (`_build_loader`: an image is cropped to `source_size`, resized to
  `img_size`, and cut into an overlapping patch grid with no per-patch
  augmentation), driven by the shipped config's patch-grid keys;
- features are the raw encoder output (`extract_features`), *before* the
  probe's mean-centre + L2-normalise. Raw features are what the
  visualisation asked for.

Only the native `cpc2018` path is wrapped here: the shipped `linear_eval.yaml`
carries no `arch` key, so the default `cpc2018` patch encoder is the shipped
representation. The additive `arch: vit` Step-2 path probes plain images at the
CLS token through `_build_vit_loader` and is not what this provider serves.

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
    """Return (features, labels, meta): features is (N, z_dim) raw encoder
    output (grid-averaged z), labels is (N,) ImageFolder class indices, meta
    describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_cpc2018")

    cfg = _load_config()
    train = cfg["train"]
    arch = train.get("arch", "cpc2018")
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    encoder = model.get_encoder().to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, train, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(encoder, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": arch,
        "image_size": image_size,
        "preprocessing": ("visual CPC 2018 val: centre crop to source_size, "
                          "resize to img_size, overlapping patch grid, no "
                          "per-patch augmentation, no ImageNet normalisation"),
    }
    return feats, labels, meta
