"""Feature-extraction provider for siglip (SigLIP).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how
SigLIP turns an image into a vector stays in one place.

**How SigLIP's eval builds its model, measured -- not guessed**
(`evaluate_linear_siglip.build_model` / `.extract_features` / `._build_loader`
/ `.run`): this is an eval-only, download-backed port, the timm-sourced
multimodal sibling of `38_clip`/`eva02`. There is **no** `adapter.load_encoder`;
the eval main builds a frozen timm SigLIP image tower with
`build_model(train, device)` and, when `train["ckpt"]` is a path, loads that
checkpoint into the named timm architecture (`timm.create_model(train["name"],
pretrained=False, num_classes=0, img_size=train["img_size"])`). So the
`encoder_path` this provider is handed is the official SigLIP image-tower
checkpoint (the pinned `backbone_artifact` download), not a trained
`encoder.pt`. This wrapper mirrors that call: it sets `train["ckpt"]` to
`encoder_path` and builds through the very same `build_model`, which already
moves the model to the device, calls `eval()` and freezes every parameter.

The shipped `configs/linear_eval.yaml` pins the official ViT-B/16 SigLIP
(`vit_base_patch16_siglip_224.webli`, 768-d); this is the only SigLIP variant
family timm registers below the large sizes (there is no tiny/small SigLIP
arch), so the provider keeps the config's `name` -- unlike `eva02`, there is no
smaller variant to infer, and the handed-in checkpoint is that architecture's
state dict.

- the feature is the image tower's own pooled image embedding: with
  `num_classes=0` the classifier head is `Identity`, and SigLIP's timm ViT has
  `global_pool='map'`, so `forward` returns the **MAP attention-pooling head's
  output** (`attn_pool(x)`) -- SigLIP's pooled image embedding itself, not the
  pre-pool per-token grid and not a separate contrastive projection (the MAP
  head *is* the pooling/projection in timm's SigLIP ViT). That is one embed_dim
  vector per image (768-d for the real ViT-B/16 SigLIP), measured directly:
  `model(zeros(6,3,224,224)).shape == (6, 768)`;
- images go through the method's own deterministic eval pipeline
  (`_build_loader`, driven by timm's data config for the built model:
  `resolve_model_data_config` -> resize to round(img_size / crop_pct) with the
  model's interpolation, centre crop to img_size, [0,1], the model's own
  mean/std -- SigLIP uses a symmetric 0.5/0.5 mean/std, not ImageNet's -- no
  augmentation), exactly as the eval main does. Measured for the base:
  mean/std (0.5, 0.5, 0.5), crop_pct 0.9, bicubic;
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
    output (SigLIP's pooled image embedding: the MAP attention-pool head output,
    768-d for the real ViT-B/16 SigLIP), labels is (N,) ImageFolder class
    indices, meta describes the run. `encoder_path` names the official SigLIP
    image-tower checkpoint (a pinned download), not a trained encoder."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_siglip")

    cfg = _load_config()
    train = dict(cfg["train"])
    image_size = int(train["img_size"])

    # Mirror the eval main: point train["ckpt"] at the checkpoint and build the
    # frozen timm image tower with the very same build_model (which moves to the
    # device, calls eval() and freezes every parameter).
    train["ckpt"] = str(encoder_path)
    dev = ev.resolve_device(device)
    model = ev.build_model(train, dev)

    # The loader follows the built model's own timm data config, exactly as the
    # eval main does (SigLIP uses a symmetric 0.5/0.5 mean/std, not ImageNet's).
    import timm.data
    dc = timm.data.resolve_model_data_config(model)
    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers),
        mean=dc["mean"], std=dc["std"], crop_pct=dc["crop_pct"],
        interpolation=dc["interpolation"])
    feats, labels = ev.extract_features(model, loader, dev)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("name", "vit_base_patch16_siglip_224"),
        "image_size": image_size,
        "feature_source": (
            "SigLIP official pretrained image tower (built by build_model, timm "
            "vit_*_siglip with num_classes=0 so the head is Identity); the "
            "feature is the MAP attention-pooling head output global_pool='map' "
            "-- SigLIP's pooled image embedding, NOT the pre-pool token grid nor "
            "a separate projection; NOT a trained encoder.pt -- encoder_path is "
            "the pinned SigLIP download"),
        "preprocessing": (
            "SigLIP eval: resize to round(img_size/crop_pct) + centre crop to "
            "img_size, [0,1], the model's own mean/std from timm's data config "
            "(SigLIP is symmetric 0.5/0.5, NOT ImageNet); feature is the "
            "MAP-pooled image embedding, raw, before the probe's mean-centre + "
            "L2-normalise"),
    }
    return feats, labels, meta
