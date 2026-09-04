"""Feature-extraction provider for data2vec2.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how
data2vec-vision turns an image into a vector stays in one place:

- this port is **eval-only**: there is no `adapter.load_encoder`. The frozen
  backbone is rebuilt by `evaluate_linear_data2vec2.build_model`, which reads
  the shipped `linear_eval.yaml` architecture keys and loads the checkpoint
  handed to it through `train["ckpt"]`. The driver passes the resolved encoder
  path there. The model class is `transformers`' `Data2VecVisionModel`
  (`add_pooling_layer=False`), and the probed feature is the **CLS token of
  `last_hidden_state`** -- one embed_dim (768 for data2vec-vision-base) vector
  per image;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> bicubic resize to a square `img_size` x `img_size`,
  `[0,1]`, symmetric mean/std 0.5 normalisation, **no centre crop**), the
  data2vec-vision preprocessor rather than ImageNet's;
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

Imports are bare module names resolved through this method's directory, as the
evaluator itself does. That is safe because the driver runs each method in
isolation; do not rely on this module and another method's modules coexisting
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
    """Return (features, labels, meta): features is (N, D) raw encoder output
    (the CLS token of data2vec-vision's last_hidden_state), labels is (N,)
    ImageFolder class indices, meta describes the run."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_data2vec2")

    cfg = _load_config()
    train = dict(cfg["train"])
    train["ckpt"] = str(encoder_path)
    image_size = int(train["img_size"])

    model = ev.build_model(train, device)
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
        "arch": train.get("name", "data2vec-vision"),
        "image_size": image_size,
        "preprocessing": ("data2vec-vision eval: bicubic resize to a square "
                          f"{image_size}x{image_size}, [0,1], symmetric "
                          "mean/std 0.5, no centre crop; feature is the CLS "
                          "token of last_hidden_state"),
    }
    return feats, labels, meta
