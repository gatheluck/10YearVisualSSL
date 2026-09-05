"""Feature-extraction provider for 24_beit.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how BEiT
turns an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, then read through `get_encoder()`. That encoder is the
  BEiT ViT trunk, and its feature is the **mean of the patch tokens with the
  CLS token excluded** -- one 768-d (embed_dim) vector per image;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `val_transform`: bicubic resize to 256, centre crop to
  224, [0,1], **ImageNet** mean/std normalisation, no augmentation);
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
    """Return (features, labels, meta): features is (N, 768) raw encoder output
    (mean of BEiT patch tokens, CLS excluded), labels is (N,) ImageFolder class
    indices, meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_beit")

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
        "arch": train.get("arch", "beit"),
        "image_size": image_size,
        "preprocessing": ("BEiT eval: bicubic resize to 256 + centre crop to "
                          "224, [0,1], ImageNet mean/std; feature is the mean "
                          "of the patch tokens with CLS excluded"),
    }
    return feats, labels, meta
