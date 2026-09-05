"""Feature-extraction provider for 18_sela.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how SeLa turns
an image into a vector stays in one place:

- the frozen model is rebuilt from `encoder.pt` by the adapter's `load_encoder`
  (a `ResNetSeLa`: the ResNet backbone loads, the prototype heads stay random --
  they are excluded from `encoder.pt` and are not read for features). SeLa's eval
  main hands this whole model to `extract_features`, which reads the 2048-d
  backbone feature through `model.get_features`; the provider mirrors that line;
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: resize + centre crop, [0,1], **with** ImageNet mean/std --
  SeLa trains on ImageNet-normalised inputs);
- features are the raw backbone output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

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
    """Return (features, labels, meta): features is (N, 2048) raw backbone
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_sela")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["image_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "resnetv2"),
        "image_size": image_size,
        "preprocessing": ("SeLa eval: resize + centre crop, [0,1], "
                          "ImageNet normalisation"),
    }
    return feats, labels, meta
