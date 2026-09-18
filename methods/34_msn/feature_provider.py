"""Feature-extraction provider for 34_msn.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how MSN turns
an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which builds the bare MSN anchor ViT trunk from the eval
  config dims and loads the trunk weights into it. `load_encoder` returns that
  ViT model directly (no `get_encoder` step); its feature is the **CLS token at
  embed_dim** -- one 384-d (deit_small embed_dim, shipped `linear_eval.yaml`)
  vector per image, read through `forward_features` via `return_before_head`;
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `val_transform`: bicubic resize to 256, centre crop to
  224, [0,1], **ImageNet** mean/std normalisation, no augmentation);
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

The eval main and the adapter both take the model directly (`backbone =
model.to(device)`); this provider mirrors that exact line and the same
`extract_features(backbone, loader, device)` call.

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
    """Return (features, labels, meta): features is (N, 384) raw encoder output
    (the MSN anchor ViT CLS token), labels is (N,) ImageFolder class indices,
    meta describes the run."""
    import torch

    import provider_support
    options = provider_support.load_native_options(encoder_path, METHOD_NAME,
                                                  {'feature_profile', 'config_overrides', 'interpolation'})
    profile = options.get('feature_profile')
    if profile not in (None, 'target_block1'):
        raise ValueError('unsupported native feature profile')
    if profile and options.get('interpolation') != 'bilinear':
        raise ValueError('native MSN requires explicit bilinear interpolation')
    adapter = provider_support.import_sibling(METHOD_DIR, 'adapter')
    ev = provider_support.import_sibling(METHOD_DIR, 'evaluate_linear_msn')
    cfg = provider_support.configure_native_config(_load_config(), options)
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    backbone = model.to(device)
    if profile:
        backbone = provider_support.import_sibling(METHOD_DIR, 'native_step1').NativeFeatures(backbone)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    if profile:
        provider_support.configure_native_resize(_dataset.transform, options)
    feats, labels = ev.extract_features(backbone, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "msn"),
        "image_size": image_size,
        "preprocessing": ("MSN eval: bicubic resize to 256 + centre crop to "
                          "224, [0,1], ImageNet mean/std; feature is the anchor "
                          "ViT CLS token at embed_dim"),
    }
    meta['native_feature_options'] = options
    if profile:
        meta['feature_profile'] = profile
        meta['preprocessing'] = 'Native MSN: bilinear resize256/crop224, ImageNet normalization, target last-block CLS before final norm'
    return feats, labels, meta
