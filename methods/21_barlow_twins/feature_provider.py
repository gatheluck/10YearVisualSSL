"""Feature-extraction provider for 21_barlow_twins.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how Barlow
Twins turns an image into a vector stays in one place:

- the frozen encoder is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which returns the backbone directly -- the 2048-d ResNet-50
  (`get_encoder()` is `self.backbone`, whose `fc` is `Identity`, so a forward
  yields the pooled 2048-d feature);
- images go through the method's own deterministic eval preprocessing: resize
  to 256, centre crop, `[0, 1]`, **with** ImageNet mean/std -- the exact
  `val` transform `evaluate_linear.get_dataloaders` builds, with its
  normalisation constants taken from that module rather than restated here;
- features are the raw encoder output (`evaluate_linear.extract_features`,
  the same call the evaluation makes over its validation loader), *before* any
  probe-side normalisation. Raw features are what the visualisation asked for.

Two deviations from the 13_mocov1 exemplar, both measured, not assumed:

1. This method's evaluation returns the backbone directly from `load_encoder`
   (`model.get_encoder()`), so there is no second `get_encoder()` call here as
   there was for MoCo, whose adapter handed back the whole model.
2. Its evaluation has no per-split loader helper: the only loader,
   `get_dataloaders`, builds `train` and `val` together and demands both
   directories. Honouring the `split` argument -- and a val-only extraction
   root that has no `train/` -- the loader is built here for the one requested
   split, reusing that function's exact `val` transform (and its ImageNet
   constants). The feature-extraction call itself is the evaluation's own,
   unchanged.

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
    """Return (features, labels, meta): features is (N, 2048) raw encoder
    output, labels is (N,) ImageFolder class indices, meta describes the run."""
    import torch
    from torchvision import datasets, transforms

    import provider_support
    adapter = provider_support.import_sibling(METHOD_DIR, "adapter")
    ev = provider_support.import_sibling(METHOD_DIR, "evaluate_linear")

    cfg = _load_config()
    train = cfg["train"]
    image_size = int(train["img_size"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    # The adapter's `load_encoder` returns the backbone itself (get_encoder()),
    # not the whole model, so there is nothing more to unwrap.
    encoder = adapter.load_encoder(state, cfg).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # The evaluation's own `val` transform, rebuilt for the one requested split.
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=ev._IMAGENET_MEAN, std=ev._IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(str(Path(data_root) / split), val_tf)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=int(num_workers), drop_last=False)

    feats, labels = ev.extract_features(encoder, loader, device)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train.get("arch", "resnet"),
        "image_size": image_size,
        "preprocessing": ("Barlow Twins eval: resize 256 + centre crop, "
                          "[0,1], ImageNet normalisation"),
    }
    return feats, labels, meta
