"""Feature-extraction provider for 01_context_prediction.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how the
official-style Context Prediction encoder turns an image into a vector stays in
one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which for the shipped (non-ViT) config returns the AlexNet
  encoder module itself -- not a model to call `get_encoder()` on. This mirrors
  `evaluate_linear_official.run`, which freezes that encoder and reads features
  straight from it;
- images go through the method's own deterministic eval pipeline, taken from
  `evaluate_linear_official.make_loaders`'s validation transform: resize 256 +
  centre crop to `img_size`, [0,1], **with** ImageNet mean/std -- the official
  linear-probe protocol this port reproduces;
- features are the raw encoder output (`extract_features`), *before* the
  probe's linear layer. Raw features are what the visualisation asked for. For
  a 224x224 ImageNet image the encoder emits a 4096-d fc6 vector (fc6 with an
  adaptive 2x2 pool), measured on this model.

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
    ev = importlib.import_module("evaluate_linear_official")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])
    seed = int(cfg.get("seed", 42))

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    encoder = adapter.load_encoder(state, cfg)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # `make_loaders` builds both the train and the deterministic val loader;
    # the eval main reads val features from the second, so this takes the same.
    _train_loader, val_loader = ev.make_loaders(
        str(data_root), int(batch_size), int(num_workers), image_size,
        seed=seed)
    feats, labels = ev.extract_features(encoder, val_loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "alexnet"),
        "image_size": image_size,
        "preprocessing": ("Context Prediction official linear-eval val: "
                          "resize 256 + centre crop, [0,1], "
                          "ImageNet normalisation"),
    }
    return feats, labels, meta
