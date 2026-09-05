"""Feature-extraction provider for 34_msn.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how MSN turns
an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which builds the bare MSN anchor ViT trunk from the eval
  config dims and loads the trunk weights into it. `load_encoder` returns that
  ViT model directly (no `get_encoder` step); its feature is the **CLS token at
  embed_dim** -- one 384-d (deit_small embed_dim, shipped `linear_eval.yaml`)
  vector per image, read through `forward_features` via `return_before_head`;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `val_transform`: bicubic resize to 256, centre crop to
  224, [0,1], **ImageNet** mean/std normalisation, no augmentation);
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

The eval main and the adapter both take the model directly (`backbone =
model.to(device)`); this provider mirrors that exact line and the same
`extract_features(backbone, loader, device)` call.

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
    """Return (features, labels, meta): features is (N, 384) raw encoder output
    (the MSN anchor ViT CLS token), labels is (N,) ImageFolder class indices,
    meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_msn")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    backbone = model.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(backbone, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "msn"),
        "image_size": image_size,
        "preprocessing": ("MSN eval: bicubic resize to 256 + centre crop to "
                          "224, [0,1], ImageNet mean/std; feature is the anchor "
                          "ViT CLS token at embed_dim"),
    }
    return feats, labels, meta
