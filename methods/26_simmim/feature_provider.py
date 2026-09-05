"""Feature-extraction provider for 26_simmim.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how SimMIM
turns an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which for the shipped (native) recipe constructs the bare
  timm Swin encoder and loads the weights straight in. That model is used
  directly as the backbone -- exactly as the eval's `run()` does
  (`backbone = model.to(device)`), with no `get_encoder()` step;
- the feature is the mean of the Swin's `forward_features` tokens
  (embed_dim * 2**(num_stages-1); 1024-d for the shipped Swin-B) for the native
  recipe, or the CLS token for the unified ViT recipe. Which one is chosen is
  the adapter's own `eval_pool(config)`, the same value the eval `run()` passes
  to `extract_features(...)`, so the pooling is not guessed here;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `_val_transform`: bicubic resize to round(image_size *
  256 / 224), centre crop to image_size, [0,1], **ImageNet** mean/std
  normalisation, no augmentation);
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
    """Return (features, labels, meta): features is (N, D) raw encoder output
    (mean of the Swin tokens for the native recipe; the CLS token for the
    unified ViT), labels is (N,) ImageFolder class indices, meta describes the
    run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_simmim")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])
    recipe = train.get("recipe", "native")

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    backbone = model.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    pool = adapter.eval_pool(cfg)
    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(backbone, loader, device, pool)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": "vit" if recipe == "unified" else "swin",
        "image_size": image_size,
        "pooling": pool,
        "preprocessing": ("SimMIM eval: bicubic resize to round(image_size * "
                          "256 / 224) + centre crop to image_size, [0,1], "
                          "ImageNet mean/std; feature is the "
                          + ("CLS token" if pool == "cls"
                             else "mean of the Swin forward_features tokens")),
    }
    return feats, labels, meta
