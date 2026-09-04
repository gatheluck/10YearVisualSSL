"""Feature-extraction provider for beitv2.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how BEiT v2
turns an image into a vector stays in one place.

beitv2 is a **pure eval-only** port: there is no `adapter.load_encoder`. The
frozen backbone is rebuilt from a checkpoint by the eval module's
`build_model`, which reads the checkpoint path from `train["ckpt"]` and applies
beit2's own rel-pos-bias surgery (`load_pt1k_checkpoint`). So this provider sets
the shipped config's `ckpt` to `encoder_path` and lets `build_model` load it --
the same path a real linear-eval run takes. The `encoder_path` handed in is the
official `beitv2_base_patch16_224_pt1k` checkpoint (a sha256-pinned download).

- the feature is the backbone's own canonical global feature: with
  `num_classes=0` and `use_mean_pooling=True`, `forward` returns
  `fc_norm(patch_tokens.mean(1))` -- the **mean over the patch tokens with the
  CLS token (position 0) excluded**, then the finetune LayerNorm. That is one
  768-d (embed_dim) vector per image;
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: bicubic resize to round(224 / (224/256)) = 256, centre crop
  to 224, [0,1], **ImageNet** mean/std normalisation, no augmentation), using
  the very constants (`IMAGENET_MEAN`, `IMAGENET_STD`, `CROP_PCT`) the eval main
  passes;
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

Imports are bare module names resolved through this method's directory, as the
eval module itself does. That is safe because the driver runs each method in
isolation; do not rely on this module and another method's `models` coexisting
in one interpreter.
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
    (mean of BEiT v2 patch tokens, CLS excluded, then fc_norm), labels is (N,)
    ImageFolder class indices, meta describes the run."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_beitv2")
    models = importlib.import_module("models")

    cfg = _load_config()
    train = dict(cfg["train"])
    train["ckpt"] = str(encoder_path)
    image_size = int(train["img_size"])

    model = ev.build_model(train, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers),
        mean=models.IMAGENET_MEAN, std=models.IMAGENET_STD,
        crop_pct=models.CROP_PCT)
    feats, labels = ev.extract_features(model, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("name", "beitv2"),
        "image_size": image_size,
        "preprocessing": ("BEiT v2 eval: bicubic resize to round(224/(224/256))"
                          " + centre crop to 224, [0,1], ImageNet mean/std; "
                          "feature is the mean of the patch tokens with CLS "
                          "excluded, then fc_norm"),
    }
    return feats, labels, meta
