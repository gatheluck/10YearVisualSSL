"""Feature-extraction provider for 17_swav.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how SwAV
turns an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which for the ResNet-50 path hands back the pooled backbone
  itself (a `Sequential` ending in a flatten -- 2048-d), not a wrapper you have
  to call `get_encoder()` on. That is the same object `evaluate_linear.run`
  receives and reads through directly;
- images go through this method's own deterministic eval pipeline, replicated
  from `evaluate_linear.get_dataloaders`' validation branch: resize to 256,
  centre crop to `img_size`, [0,1], then SwAV's captured ImageNet-style
  normalisation. That normalisation is reproduced exactly, including the
  captured mean/std whose red std is 0.228 (not the usual 0.229) -- the port
  matches what the method trained and evaluates with, quirk and all;
- features are the raw encoder output, extracted exactly as `run` does
  (`feats = encoder(imgs); if feats.dim() > 2: flatten`), *before* the probe's
  linear head. Raw features are what the visualisation asked for.

Unlike some sibling methods, `evaluate_linear.py` exposes no standalone
`_build_loader`/`extract_features` helpers -- both live inline in
`get_dataloaders` (train+val) and in `run`'s epoch loop. `get_dataloaders`
would demand a `train/` tree and apply training augmentation, so the val-only
pipeline is mirrored here rather than called; the transform and the extraction
are copied line for line from that source.

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
    """Return (features, labels, meta): features is (N, 2048) raw encoder
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch
    from torchvision import datasets, transforms

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    # `load_encoder` returns the ResNet-50 backbone itself (2048-d pooled +
    # flattened), the object `evaluate_linear.run` reads through directly.
    encoder = adapter.load_encoder(state, cfg).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Copied from evaluate_linear.get_dataloaders' validation branch, captured
    # normalisation and all (red std 0.228, not 0.229).
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.228, 0.224, 0.225])
    va_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])
    dataset = datasets.ImageFolder(str(Path(data_root) / split), va_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=int(num_workers), drop_last=False)

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            out = encoder(imgs.to(device))
            if out.dim() > 2:                       # as evaluate_linear.run does
                out = out.view(out.size(0), -1)
            feats.append(out.float().cpu())
            labels.append(lbs)
    feats = torch.cat(feats).numpy()
    labels = torch.cat(labels).numpy()

    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "resnet"),
        "image_size": image_size,
        "preprocessing": ("SwAV eval: resize 256 + centre crop, [0,1], "
                          "captured ImageNet normalisation (red std 0.228)"),
    }
    return feats, labels, meta
