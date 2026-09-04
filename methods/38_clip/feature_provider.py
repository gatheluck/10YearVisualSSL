"""Feature-extraction provider for 38_clip.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how CLIP
turns an image into a vector stays in one place:

- the frozen backbone is the CLIP **image tower** (a `VisionTransformer`),
  rebuilt from `encoder.pt` by the adapter's `load_encoder` (which reads the
  image-tower dimensions from the shipped `linear_eval_vit.yaml`). That is the
  encoder.pt / sha256 shape the driver works in; the eval main uses the tower
  **directly** as the model, and its feature is the **pooled projected image
  embedding** -- `visual(x)`, i.e. `encode_image` -- one `output_dim`-d (512)
  vector per image, not a per-token grid and not the pre-projection width;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> the official CLIP eval transform: **bicubic** resize to
  the resolution, centre crop, convert to RGB, `[0,1]`, then **CLIP** mean/std
  normalisation. Note this normalisation differs from ImageNet's:
  mean `(0.48145466, 0.4578275, 0.40821073)`,
  std `(0.26862954, 0.26130258, 0.27577711)`);
- features are the raw encoder output (`extract_features`, whose per-batch call
  is `extract_cls(model, imgs, train, device)`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

This provider works the encoder.pt path (the driver passes an encoder.pt and
records its sha256), which is the from-scratch ViT-B/16 image tower the unified
Step-2 pretrain produced -- the only tower the driver's frozen-encoder contract
covers. The Step-1 as-is backbone (the official ViT-B/32, loaded through the
checksum-pinned `clip.load`) is a different, download-only path and is not what
`extract-features` drives.

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
    """Return (features, labels, meta): features is (N, 512) raw encoder output
    (the CLIP image tower's pooled projected embedding, `visual(x)`), labels is
    (N,) ImageFolder class indices, meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_clip")

    cfg = _load_config()
    train = cfg["train"]
    resolution = int(train["resolution"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    model = model.float().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, resolution, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, train, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "clip-vit"),
        "image_size": resolution,
        "feature_source": ("CLIP image tower pooled projected embedding "
                           "visual(x) (encode_image)"),
        "preprocessing": ("CLIP eval: bicubic resize to the resolution + centre "
                          "crop, convert RGB, [0,1], CLIP mean/std (NOT "
                          "ImageNet: mean (0.48145466, 0.4578275, 0.40821073), "
                          "std (0.26862954, 0.26130258, 0.27577711)); feature "
                          "is the pooled projected image embedding visual(x), "
                          "before the probe's mean-centre + L2-normalise"),
    }
    return feats, labels, meta
