"""Feature-extraction provider for image_gpt.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how iGPT turns
an image into a vector stays in one place -- and it reproduces the linear-eval
main path exactly (`evaluate_linear_igpt.run` via the adapter's
`run_linear_eval`):

- the frozen model is rebuilt from `encoder.pt` by the adapter's `load_encoder`
  (the same call the probe makes); the whole `IGPT` is the representation this
  port trains, so `model.extract_features` is read directly -- there is no
  separate backbone to peel off;
- iGPT reads discrete colour tokens, so an image must be quantised with the
  **same** colour clusters the model was trained on. The probe main is handed
  those clusters as a path (`config["clusters"]` -> `np.load`); the adapter
  writes that `clusters.npy` **beside** `encoder.pt`, so given only the encoder
  path the provider reads them from there, through this method's own
  `downstream_backbone._load_clusters_beside` (one implementation of the
  "beside encoder.pt" rule, reused, not re-coded). A missing or wrong-sized
  cluster table is refused there;
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: resize + centre crop, [0,1], **no** ImageNet normalisation
  -- iGPT quantises raw pixels), then `extract_features` quantises them to
  colour tokens and reads the middle transformer layer, mean-pooled over the
  token sequence;
- features are that raw representation (`extract_features`), *before* the
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
    """Return (features, labels, meta): features is (N, n_embd) raw iGPT
    representation (a middle transformer layer, mean-pooled), labels is (N,)
    ImageFolder class indices, meta describes the run."""
    import numpy as np
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_igpt")
    db = importlib.import_module("downstream_backbone")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])
    vocab_size = int(train["vocab_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # The colour clusters the model was trained on, co-located with encoder.pt,
    # exactly as the probe reads them (config["clusters"]).
    clusters = np.asarray(
        db._load_clusters_beside(Path(encoder_path), vocab_size),
        dtype=np.float32)

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, clusters, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": "igpt",
        "image_size": image_size,
        "clusters_source": "clusters.npy beside encoder.pt",
        "pooling": ("middle transformer layer (block n_layer // 2), "
                    "mean-pooled over the token sequence"),
        "preprocessing": ("iGPT eval: resize + centre crop, [0,1], no ImageNet "
                          "normalisation, then quantised to colour-cluster "
                          "tokens with the model's clusters"),
    }
    return feats, labels, meta
