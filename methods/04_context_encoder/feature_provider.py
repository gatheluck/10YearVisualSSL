"""Feature-extraction provider for 04_context_encoder.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how the
Context Encoder turns an image into a vector stays in one place:

- the frozen representation is rebuilt from `encoder.pt` by the adapter's
  `load_encoder` -- for the native AlexNet path (the shipped config) that is the
  conv encoder plus its 4096-d bottleneck; the whole model is handed to the
  evaluation, which reads the bottleneck as `model(x) -> (_, features)`;
- images go through the method's own deterministic eval pipeline
  (`create_dataloader('linear_probe', ..., split, preprocess='torch')`: on the
  val split, Resize(256) + centre crop, [0,1], **with** ImageNet mean/std --
  the AlexNet encoder was trained on ImageNet-normalised inputs);
- features are the raw encoder output (`extract_features`), *before* the
  probe's linear head. Raw features are what the visualisation asked for.

The val pipeline has no random augmentation (centre crop, no flip) and the
model runs frozen in eval mode, so extraction is deterministic; there is no RNG
to seed here, unlike the GAN pretrain.

Sibling modules (`adapter`, `evaluate_linear`) are imported through
`provider_support.import_sibling`, which resolves them against THIS method even
when another method already imported a module of the same name in one
interpreter. `evaluate_linear` is defined by four methods, so the test suite and
the driver's in-process path would otherwise hand this provider another method's
copy; the isolated worker subprocess only ever holds one method, so it is a
no-op there.
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
    output (the AlexNet bottleneck), labels is (N,) ImageFolder class indices,
    meta describes the run."""
    import torch

    import provider_support
    native_options = provider_support.load_native_options(encoder_path, METHOD_NAME, {'feature_profile'})
    profile = native_options.get('feature_profile')
    if profile not in (None, 'official_caffe_pool5'):
        raise ValueError('unsupported native feature profile')
    adapter = provider_support.import_sibling(METHOD_DIR, "adapter")
    ev = provider_support.import_sibling(METHOD_DIR, "evaluate_linear")

    cfg = _load_config()
    train = cfg["train"]
    arch = train.get("arch", "alexnet")
    model_type = "vit" if arch == "vit" else "alexnet"
    image_size = int(train["image_size"] if arch == "vit" else train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    official = any(k.startswith('features.') for k in state)
    if bool(profile) != official:
        raise ValueError('official checkpoint and native feature profile must be selected together')
    if profile:
        image_size = 227
        native = provider_support.import_sibling(METHOD_DIR, 'native_step1')
    model = adapter.load_encoder(state, cfg).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    loader = ev.create_dataloader(
        "linear_probe", str(data_root), split=split, batch_size=int(batch_size),
        num_workers=int(num_workers), img_size=image_size, preprocess="torch")
    if profile:
        loader.dataset.dataset.transform = native.make_transform()
    feats, labels = ev.extract_features(model, loader, device, model_type)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": arch,
        "image_size": image_size,
        "preprocessing": ("Context Encoder linear-probe eval: resize + centre "
                          "crop, [0,1], ImageNet normalisation"),
    }
    meta['native_feature_options'] = native_options
    if profile:
        meta.update(arch='official_caffe_alexnet', feature_profile=profile,
                    preprocessing='Official Caffe features: bilinear resize 256, centre crop 227, BGR 0-255 minus (104,117,123), flattened pool5')
    return feats, labels, meta
