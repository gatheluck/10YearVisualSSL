"""Feature-extraction provider for sam3 (Meta SAM 3).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how SAM 3
turns an image into a vector stays in one place:

- this is an **eval-only** port (the `transformers`-sourced sibling of
  `data2vec2`): there is no `encoder.pt` from training and no
  `adapter.load_encoder`. The frozen backbone is built by the eval module's
  `build_model`, which reads its checkpoint from `train["ckpt"]`. The provider
  therefore sets `train["ckpt"]` to the passed `encoder_path` -- the official
  `sam3.pt` (its ViTDet-style trunk keys are converted onto `Sam3ViTModel` by
  `sam3_trunk.load_official_trunk`; a random tiny build only happens when the
  checkpoint is empty, which the provider never does);
- the feature is the vision encoder's patch tokens **mean-pooled over the
  sequence** (SAM 3's ViT has no CLS token); one `hidden_size`-d vector per
  image (1024 for the released ViT-L);
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> bicubic resize to `img_size` (336, the config's
  `pretrain_image_size`), centre crop to the same, [0,1], **ImageNet**
  mean/std normalisation, no augmentation);
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

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
    """Return (features, labels, meta): features is (N, hidden_size) raw encoder
    output (mean of SAM 3 patch tokens, the ViT has no CLS), labels is (N,)
    ImageFolder class indices, meta describes the run."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_sam3")

    cfg = _load_config()
    train = dict(cfg["train"])
    train["ckpt"] = str(encoder_path)
    image_size = int(train["img_size"])

    dev = ev.torch.device(device)
    model = ev.build_model(train, dev)
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
        "arch": train.get("name", "facebook/sam3"),
        "image_size": image_size,
        "preprocessing": ("SAM 3 eval: bicubic resize + centre crop to "
                          "img_size, [0,1], ImageNet mean/std; feature is the "
                          "mean of the vision encoder's patch tokens (no CLS)"),
    }
    return feats, labels, meta
