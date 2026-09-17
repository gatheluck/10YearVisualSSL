"""Feature-extraction provider for 35_vjepa.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how V-JEPA
turns an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which returns the V-JEPA EMA target encoder (a ViT trunk)
  directly -- there is no separate head. Its feature is the **mean of all its
  output tokens** (`encoder(images).mean(dim=1)`); V-JEPA carries no CLS token,
  so every token is pooled -- one embed_dim vector per image (768-d for the
  shipped vit_base);
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `val_transform`: bicubic resize to round(crop_size*256/224),
  centre crop to crop_size, [0,1], **ImageNet** mean/std normalisation, no
  augmentation);
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

Method imports are scoped through provider_support; the driver also isolates
each method in a subprocess. Hash-bound export sidecars opt into the recorded
Step-1 protocol, while ordinary exports retain the default representation.
"""

from __future__ import annotations

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
    output (mean of the V-JEPA target encoder's output tokens), labels is (N,)
    ImageFolder class indices, meta describes the run."""
    import torch

    import provider_support
    options = provider_support.load_native_options(encoder_path, METHOD_NAME,
                                                  {'feature_profile', 'config_overrides'})
    profile = options.get('feature_profile')
    if profile not in (None, 'official_video_meanpool'):
        raise ValueError('unsupported native feature profile')
    adapter = provider_support.import_sibling(METHOD_DIR, 'adapter')
    ev = provider_support.import_sibling(METHOD_DIR, 'evaluate_linear_vjepa')
    cfg = provider_support.configure_native_config(_load_config(), options)
    train = cfg["train"]
    image_size = int(train["crop_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    encoder = adapter.load_encoder(state, cfg).to(device)
    if profile:
        encoder = provider_support.import_sibling(METHOD_DIR, 'native_step1').NativeFeatures(encoder)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(encoder, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("model_name", "vit_base"),
        "image_size": image_size,
        "preprocessing": ("V-JEPA eval: bicubic resize to round(crop_size*256/"
                          "224) + centre crop to crop_size, [0,1], ImageNet "
                          "mean/std; feature is the mean of all the target "
                          "encoder's output tokens (no CLS token)"),
    }
    meta['native_feature_options'] = options
    if profile:
        meta['feature_profile'] = profile
        meta['preprocessing'] = 'Native V-JEPA: bicubic resize256/crop224, ImageNet normalization, repeat16 frames, mean tokens, CUDA float16 autocast, L2 then float16 cache rounding'
    return feats, labels, meta
