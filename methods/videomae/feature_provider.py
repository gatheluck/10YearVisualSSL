"""Feature-extraction provider for videomae (VideoMAE).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how
VideoMAE turns an image into a vector stays in one place.

**How VideoMAE's eval turns an ImageNet image into a vector, measured -- not
guessed** (`evaluate_linear_videomae.build_model` / `.extract_feature` /
`._build_loader`): this is an eval-only, download-backed port. There is **no**
`adapter.load_encoder`; the eval main builds the frozen self-contained VideoMAE
ViT with `build_model(train, device)`, reading its checkpoint from
`train["ckpt"]`. So the `encoder_path` this provider is handed is the official
`MCG-NJU/videomae-base` checkpoint (the pinned `backbone_artifact` download, a
safetensors file whose `videomae.*` encoder tensors load into the ViT), not a
trained `encoder.pt`. This wrapper mirrors that call: it sets `train["ckpt"]` to
`encoder_path` and builds through the very same `build_model`.

The architecture comes from the shipped config (ViT-B/16, embed_dim 768, 12
blocks, a 16-frame clip with tubelet 2), exactly as the eval main reads it; the
checkpoint's `videomae.*` tensors must load into it with no missing or
unexpected encoder key, so the built backbone can never disagree with the
weights.

VideoMAE consumes a **video clip** `(B, 3, T, H, W)`, not a still image. The
capture's ImageNet linear eval -- which this port mirrors exactly -- feeds a
**still image replicated `num_frames` times** along the temporal axis (never
PyAV / a video dataset), runs the ViT, and mean-pools the tokens. So the
image-val contract holds: `_build_loader` is a plain torchvision `ImageFolder`
over `<data_root>/<split>`, and `extract_feature` expands each image to the clip
and returns one vector per image.

- the feature is the backbone's own global feature: VideoMAE's ViT has **no CLS
  token**, so `extract_feature` returns the **mean over all N spatio-temporal
  patch tokens** (after the final LayerNorm). That is one embed_dim (768-d for
  the real MCG-NJU/videomae-base) vector per image;
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: bicubic square resize to `img_size` (224), [0,1], the
  backbone's own ImageNet mean/std normalisation, no centre crop and no
  augmentation), using the very constants (`VIDEOMAE_MEAN`, `VIDEOMAE_STD`) the
  eval module defines;
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise (`normalize_features`). Raw features are what the
  visualisation asked for.

Imports are bare module names resolved through this method's directory, as the
eval module itself does. That is safe because the driver runs each method in
isolation; do not rely on this module and another method's modules coexisting in
one interpreter.
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
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (the mean over VideoMAE's spatio-temporal patch tokens; the ViT has
    no CLS token), labels is (N,) ImageFolder class indices, meta describes the
    run. `encoder_path` names the official VideoMAE backbone checkpoint (a pinned
    download), not a trained encoder."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_videomae")

    cfg = _load_config()
    train = dict(cfg["train"])
    train["ckpt"] = str(encoder_path)
    image_size = int(train["img_size"])

    dev = ev.resolve_device(device)
    model = ev.build_model(train, dev)   # to(device) + eval + freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, dev)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("name", "MCG-NJU/videomae-base"),
        "image_size": image_size,
        "num_frames": int(train["num_frames"]),
        "feature_source": (
            "VideoMAE official pretrained backbone (built by build_model from "
            "the shipped config's architecture; the checkpoint's videomae.* "
            "encoder tensors load strict); NOT a trained encoder.pt -- "
            "encoder_path is the pinned VideoMAE download"),
        "preprocessing": (
            "VideoMAE eval: bicubic square resize to img_size, [0,1], the "
            "backbone's ImageNet mean/std, no centre crop, no augmentation; "
            "each still image is replicated num_frames times along the temporal "
            "axis to form the clip; feature is the mean over all spatio-temporal "
            "patch tokens (no CLS), raw, before the probe's mean-centre + "
            "L2-normalise"),
    }
    return feats, labels, meta
