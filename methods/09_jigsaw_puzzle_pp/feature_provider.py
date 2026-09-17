"""Feature-extraction provider for 09_jigsaw_puzzle_pp.

The hash-bound imagenet_cluster_cls profile instead uses normalized 224px
images and the knowledge-transfer AlexNet trunk (9216 dimensions).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how Jigsaw++
turns an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, then read through `get_encoder()`. The shipped `linear_eval`
  config is `arch=vgg16` (the default), so this is the shared VGG16 pretext
  encoder (a 1024-d feature);
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: BASIC5 rule `b` -- Resize (shorter side) + CenterCrop at the
  encoder's native size, aspect-preserving, [0,1], **no** ImageNet mean/std --
  the probe uses unnormalised inputs), at the size the probe reads (`tile_size`
  for `arch=vgg16`);
- features are the raw encoder output (`extract_features`), *before* the
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
    """Return (features, labels, meta): features is (N, 1024) raw VGG16 encoder
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch
    import provider_support
    native_options = provider_support.load_native_options(
        encoder_path, METHOD_NAME, {'feature_profile', 'config_overrides'})
    profile = native_options.get('feature_profile')
    if profile is not None and profile != 'imagenet_cluster_cls':
        raise ValueError('unsupported native feature profile')

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    adapter = importlib.import_module("adapter")
    ev = importlib.import_module("evaluate_linear_jigsaw_pp")

    cfg = provider_support.configure_native_config(_load_config(), native_options)
    train = cfg["train"]
    if profile:
        train.update(arch='alexnet_cluster_cls', image_size=224)
    arch = train.get("arch", "vgg16")
    # The probe reads images at the encoder's native size: the VGG16 tile size,
    # the AlexNet knowledge-transfer image size, or the ViT puzzle size.
    if arch == "alexnet_cluster_cls":
        image_size = int(train["image_size"])
    elif arch == "vit":
        image_size = int(train["puzzle_size"])
    else:
        image_size = int(train["tile_size"])

    if profile:
        image_size = 224
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    encoder = model.get_encoder().to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    if profile:
        from torchvision import transforms as T
        _dataset.transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
    feats, labels = ev.extract_features(encoder, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "native_feature_options": native_options,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": arch,
        "image_size": image_size,
        "preprocessing": ("Jigsaw++ eval (BASIC5 rule b): Resize (shorter side) "
                          "+ CenterCrop at the encoder's native size, "
                          "aspect-preserving, [0,1], no ImageNet normalisation"),
    }
    if profile:
        meta['preprocessing'] = 'Native ImageNet val: bilinear resize 256, centre crop 224, ImageNet mean/std'
        meta['feature_profile'] = profile
    return feats, labels, meta
