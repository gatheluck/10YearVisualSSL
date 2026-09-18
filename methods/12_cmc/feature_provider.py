"""Feature-extraction provider for 12_cmc.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how CMC turns
an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, then read through `get_encoder()` -- the two-branch AlexNet's
  layer-6 (fc6) features of the L and ab branches, concatenated (2 x 2048 =
  4096-d);
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> `CMCDataset(mode="val")`): resize + centre crop, then the
  **CIE Lab** conversion (`RGB2Lab`: sRGB -> linear -> XYZ(D65) -> Lab) and a
  per-channel Lab normalisation (LAB_MEAN/LAB_STD, the Lab-gamut extremes). CMC
  trains on Lab, not RGB; there is no ImageNet mean/std here;
- features are the raw encoder output (`extract_features`), *before* the
  probe's mean-centre + L2-normalise. Raw features are what the visualisation
  asked for.

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
    """Return (features, labels, meta): features is (N, 4096) raw encoder
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch

    import provider_support
    options = provider_support.load_native_options(encoder_path, METHOD_NAME,
                                                  {'feature_profile', 'config_overrides'})
    profile = options.get('feature_profile')
    if profile not in (None, 'layer5_pool6'):
        raise ValueError('unsupported native feature profile')
    adapter = provider_support.import_sibling(METHOD_DIR, 'adapter')
    ev = provider_support.import_sibling(METHOD_DIR, 'evaluate_linear_cmc')
    cfg = provider_support.configure_native_config(_load_config(), options)
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = adapter.load_encoder(state, cfg)
    encoder = (provider_support.import_sibling(METHOD_DIR, 'native_step1').NativeFeatures(model)
               if profile else model.get_encoder()).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    if profile:
        data = provider_support.import_sibling(METHOD_DIR, 'data.cmc_dataset')
        steps = _dataset.base.transform.transforms
        for i, transform in enumerate(steps):
            if isinstance(transform, data.RGB2Lab):
                steps[i] = data.RGB2Lab(native_step1=True)
    feats, labels = ev.extract_features(encoder, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "alexnet"),
        "image_size": image_size,
        "preprocessing": ("CMC eval: resize + centre crop, CIE Lab conversion "
                          "(RGB2Lab) + per-channel Lab normalisation "
                          "(LAB_MEAN/LAB_STD), no ImageNet normalisation; "
                          "feature is the concatenated layer-6 (fc6) output of "
                          "the L and ab branches (2 x 2048)"),
    }
    meta['native_feature_options'] = options
    if profile:
        meta['feature_profile'] = profile
        meta['preprocessing'] = 'Native CMC: resize256/crop224, CIE Lab (scikit-image constants), concatenate layer5 branches, max pool6 and flatten'
    return feats, labels, meta
