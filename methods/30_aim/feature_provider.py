"""Feature-extraction provider for 30_aim.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how AIM turns
an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which returns the unified Step-2 AIM trunk (`AIMViT`, the
  lab's own from-scratch ViT-B/16). That is the encoder the feature-extraction
  subsystem keys on -- the encoder.pt this port's pretrain writes. AIM's feature
  is the **average of the last `num_feature_layers` transformer-block outputs,
  then mean-pooled over the patch tokens** -- one `embed_dim`-d vector per image
  (768 for the shipped ViT-B/16 config);
- the eval main iterates the loader itself, calling the per-batch
  `extract_feature(model, imgs, train, device)` inside the loader-based
  `extract_features(model, loader, train, device)` wrapper. This provider calls
  that same wrapper, so extraction is bit-for-bit what a `linear_eval` run does;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> bicubic resize to 256, centre crop to 224, [0,1],
  **ImageNet** mean/std normalisation, no augmentation);
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise (`normalize_features`). Raw features are what the
  visualisation asked for.

The Step-1 as-is comparison (the official downloaded AIM-600M) is a separate
path that produces no `encoder.pt`; the subsystem keys on encoder.pt, so this
provider probes the unified Step-2 trunk, matching how the exemplar providers
read `adapter.load_encoder`.

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
    with open(METHOD_DIR / "configs" / "linear_eval_vit.yaml") as f:
        return yaml.safe_load(f)


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (the mean of AIM's last `num_feature_layers` block outputs,
    mean-pooled over patch tokens), labels is (N,) ImageFolder class indices,
    meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_aim")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, train, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": "aim_vit (unified Step 2)",
        "image_size": image_size,
        "num_feature_layers": int(train["num_feature_layers"]),
        "preprocessing": ("AIM unified eval: bicubic resize to 256 + centre "
                          "crop to 224, [0,1], ImageNet mean/std; feature is the "
                          "mean of the last num_feature_layers block outputs, "
                          "mean-pooled over the patch tokens"),
    }
    return feats, labels, meta
