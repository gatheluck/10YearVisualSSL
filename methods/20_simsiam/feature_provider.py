"""Feature-extraction provider for 20_simsiam.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how SimSiam
turns an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which for this method returns the backbone **directly**
  (`SimSiamResNet.get_encoder()` -- the 2048-d ResNet-50). Unlike the first
  provider, whose adapter hands back the whole model, this one hands back the
  encoder itself, so there is no further `get_encoder()` call;
- images go through the method's own deterministic eval pipeline: the exact
  validation transform of `evaluate_linear_official.get_dataloaders`
  (`Resize(256)` + `CenterCrop(img_size)`, `[0,1]`, ImageNet mean/std --
  SimSiam trained on ImageNet-normalised inputs);
- features are the raw encoder output, computed the way the official
  evaluation's own `FrozenBackboneLinear.forward` computes them --
  `encoder(x).flatten(1)` (the ResNet-50 pool is `(N, 2048, 1, 1)`, flattened
  to `(N, 2048)`) -- *before* the linear probe's head. Raw features are what
  the visualisation asked for.

The official evaluation has no split-scoped loader or feature-extraction
helper to call: its `get_dataloaders` builds train *and* val together and
returns no features. The loop is therefore written here, mirroring that
module's `val` transform and `FrozenBackboneLinear.forward` exactly and
reusing its ImageNet normalisation constants so the numbers stay in one place.

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
    ev = importlib.import_module("evaluate_linear_official")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    # For this method `load_encoder` already returns the backbone, not the
    # whole SimSiam model -- mirror the eval main, which passes what it gets
    # straight to `run(..., encoder=...)`.
    encoder = adapter.load_encoder(state, cfg).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # The official evaluation's `val` transform (get_dataloaders), reusing its
    # normalisation constants so mean/std live in one place.
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(ev._IMAGENET_MEAN, ev._IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(str(Path(data_root) / split),
                                   transform=val_tf)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=int(num_workers), drop_last=False)

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            # FrozenBackboneLinear.forward: feats = encoder(x).flatten(1)
            out = encoder(imgs.to(device, non_blocking=True)).flatten(1)
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
        "preprocessing": ("SimSiam official eval val: Resize(256) + centre "
                          "crop, [0,1], ImageNet normalisation"),
    }
    return feats, labels, meta
