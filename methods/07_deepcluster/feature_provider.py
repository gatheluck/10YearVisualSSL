"""Feature-extraction provider for 07_deepcluster.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how
DeepCluster turns an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which returns the ready model itself (an AlexNet-BN whose
  `get_features` yields the 4096-d fc7 feature; the fixed Sobel front-end is
  rebuilt on load). Unlike the MoCo-style methods, the eval main feeds this
  model **directly** to `extract_features` -- there is no `get_encoder()` step
  -- and this provider mirrors that exactly;
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: resize + centre crop, [0,1], ImageNet mean/std --
  DeepCluster trained on ImageNet-normalised inputs; the crop size comes from
  the shipped config's `crop_size`, the native AlexNet-BN key);
- features are the raw encoder output (`extract_features`, which calls the
  model's `get_features`), *before* the probe's mean-centre + L2-normalise.
  Raw features are what the visualisation asked for.

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
    """Return (features, labels, meta): features is (N, 4096) raw fc7 encoder
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_deepcluster")

    cfg = _load_config()
    train = cfg["train"]
    arch = train.get("arch", "alexnet")
    # The native AlexNet-BN path sizes the crop by `crop_size`; the ViT path
    # would use `image_size`. The shipped config is the native path.
    image_size = int(train["image_size"]) if arch == "vit" \
        else int(train["crop_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    # The eval main feeds the loaded model directly to extract_features -- it
    # does not call get_encoder(). Mirror that acquisition line exactly.
    model = adapter.load_encoder(state, cfg)
    model = model.to(device)
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
        "arch": arch,
        "image_size": image_size,
        "preprocessing": ("DeepCluster eval: resize + centre crop, [0,1], "
                          "ImageNet normalisation"),
    }
    return feats, labels, meta
