"""Feature-extraction provider for 02_vae.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how the VAE
turns an image into a vector stays in one place:

- **The VAE has no `get_encoder`.** Its representation is the encoder's latent
  mean `mu` -- `VAE_CNN.get_features(x)` (the conv encoder, flattened, then
  `fc_mu`), one `latent_dim`-d vector per image. The frozen model is rebuilt
  from `encoder.pt` by the adapter's `load_encoder`, exactly as the linear-eval
  stage does (`adapter.run_linear_eval`), and read through the eval's own
  `_extract`, which calls `get_features`. So the feature returned here **is the
  VAE posterior mean `mu`**, before the probe's linear layer.
- images go through the method's own deterministic eval pipeline: the eval is
  **dataset-agnostic** and keeps inputs in `[0,1]` with **no ImageNet mean/std**
  (matching the VAE's reconstruction training). It auto-detects MNIST versus an
  ImageFolder and resizes to the encoder's `img_size`. The provider reuses the
  eval's own `_is_mnist` / `_mnist_transform` / `_imagefolder_transform`, so the
  preprocessing is byte-identical to the probe's.
- the eval only exposes `_loaders`, which builds *both* splits at once (it needs
  a `train/` dir and compares the two class lists). A feature dump needs only
  the requested split, so the provider builds a single-split loader from those
  same transform helpers rather than the paired one -- same preprocessing, one
  split.

Imports are bare module names resolved through this method's directory, as the
adapter itself does. That is safe because the driver runs each method in
isolation; do not rely on this module and another method's `adapter`/`models`
coexisting in one interpreter.
"""

from __future__ import annotations

import importlib
import os
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
    """Return (features, labels, meta): features is (N, latent_dim) raw
    encoder output -- the VAE posterior mean `mu` -- labels is (N,) dataset
    class indices, meta describes the run."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_vae")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    # The VAE encoder is not one submodule; the adapter rebuilds the whole
    # VAE_CNN from the config's shapes and loads the encoder weights into it.
    # `get_features` reads only the encoder half, so the decoder's default init
    # never touches the feature.
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # A single-split loader built from the eval's own transform helpers: the
    # eval's `_loaders` builds train and val together and needs a train/ dir,
    # which a val-only dump does not have. Same preprocessing, one split.
    if ev._is_mnist(str(data_root)):
        tf = ev._mnist_transform(image_size)
        dataset = datasets.MNIST(str(data_root), train=(split == "train"),
                                 download=False, transform=tf)
    else:
        tf = ev._imagefolder_transform(image_size)
        dataset = datasets.ImageFolder(os.path.join(str(data_root), split),
                                       transform=tf)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                        num_workers=int(num_workers), drop_last=False)

    feats, labels = ev._extract(model, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": "vae_cnn",
        "latent_dim": int(train["latent_dim"]),
        "image_size": image_size,
        "feature": ("VAE posterior mean mu (VAE_CNN.get_features: encoder -> "
                    "fc_mu), before the linear probe"),
        "preprocessing": ("VAE eval: resize to img_size, [0,1], no ImageNet "
                          "normalisation; dataset-agnostic (MNIST or "
                          "ImageFolder)"),
    }
    return feats, labels, meta
