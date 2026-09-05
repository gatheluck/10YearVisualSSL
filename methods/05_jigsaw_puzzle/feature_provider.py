"""Feature-extraction provider for 05_jigsaw_puzzle.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how Jigsaw
turns an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, then read through `get_encoder()` (the 512-d shared CFN
  encoder -- an AlexNet backbone with 1x1-conv "FC" layers);
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: resize to a square, [0,1], **no** ImageNet mean/std --
  this port's Jigsaw probe feeds unnormalised inputs);
- features are the raw encoder output (`extract_features`), *before* the
  probe's mean-centre + L2-normalise. Raw features are what the visualisation
  asked for.

The input size follows the eval's own rule: the ViT Step-2 encoder wants the
reassembled `puzzle_size`, while the native CFN encoder probes at `tile_size`;
the shipped `linear_eval.yaml` is the native path, so `tile_size` is used.

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
    """Return (features, labels, meta): features is (N, 512) raw encoder
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_jigsaw")

    cfg = _load_config()
    train = cfg["train"]
    # The eval's own rule: the ViT Step-2 encoder probes the reassembled
    # puzzle, the native CFN encoder probes at tile_size.
    image_size = int(train["puzzle_size"] if "puzzle_size" in train
                     else train["tile_size"])

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
        "preprocessing": ("Jigsaw eval: resize to a square, [0,1], "
                          "no ImageNet normalisation"),
    }
    return feats, labels, meta
