"""Feature-extraction provider for 31_dinov3.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how DINOv3
turns an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which returns the DINOv3 teacher ViT (register tokens + axial
  RoPE) with the backbone weights loaded. The eval main uses that model
  **directly** as the backbone; its feature is the **CLS token** read with the
  released output norm (`backbone(x, is_global=True)[0]`) -- one 768-d
  (embed_dim) vector per image;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `val_transform`: bicubic resize to 256, centre crop to
  224, [0,1], **ImageNet** mean/std normalisation, no augmentation);
- features are the raw encoder output (`extract_features`), *before* the
  probe's mean-centre + L2-normalise. Raw features are what the visualisation
  asked for.

Sibling imports are scoped to this method through provider_support. An explicit
native sidecar selects the official 512-pixel Step-1 checkpoint and architecture;
the default path continues to use the port's trained teacher backbone.
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
    """Return raw frozen CLS features and the exact input protocol metadata."""
    import torch
    import provider_support
    options = provider_support.load_native_options(encoder_path, METHOD_NAME, {'feature_profile'})
    profile = options.get('feature_profile')
    if profile not in (None, 'official_vitb16_cls512'):
        raise ValueError('unsupported native feature profile')
    adapter = provider_support.import_sibling(METHOD_DIR, 'adapter')
    ev = provider_support.import_sibling(METHOD_DIR, 'evaluate_linear_dinov3')
    cfg = _load_config()
    state = torch.load(encoder_path, map_location='cpu', weights_only=True)
    if bool(profile) != ('storage_tokens' in state):
        raise ValueError('official checkpoint and native feature profile must be selected together')
    image_size = 512 if profile else int(cfg['train']['img_size'])
    backbone = adapter.load_encoder(state, cfg).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    dataset, loader = ev._build_loader(str(data_root), split, image_size, int(batch_size), int(num_workers))
    if profile:
        native = provider_support.import_sibling(METHOD_DIR, 'native_step1')
        dataset.transform = native.make_transform()
    feats, labels = ev.extract_features(backbone, loader, device)
    feats, labels = feats.numpy(), labels.numpy()
    meta = {'method': METHOD_NAME, 'native_feature_options': options,
            'representation': 'raw', 'feat_dim': int(feats.shape[1]), 'count': int(feats.shape[0]),
            'image_size': image_size, 'arch': cfg['train'].get('arch', 'dinov3'),
            'preprocessing': 'DINOv3 eval: bicubic resize to 256 + centre crop to 224, [0,1], ImageNet mean/std; feature is the teacher CLS token (is_global=True, released output norm)'}
    if profile:
        meta.update(arch='official_dinov3_vitb16', feature_profile=profile,
                    preprocessing='Official DINOv3 ViT-B/16: bicubic resize 585, centre crop 512, ImageNet mean/std, normalized CLS token, CUDA float16 autocast',
                    inference_precision='cuda_autocast_float16' if str(device).startswith('cuda') else 'float32')
    return feats, labels, meta
