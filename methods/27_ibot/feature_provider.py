"""Feature-extraction provider for 27_ibot.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how iBOT
turns an image into a vector stays in one place:

- the frozen backbone is rebuilt from `encoder.pt` by the adapter's
  `load_encoder`, which returns the plain teacher ViT trunk directly (a
  `VisionTransformer`; there is no separate `get_encoder()` -- the model *is*
  the backbone, and its `forward` returns a `(cls_token, patch_tokens)` tuple,
  so features are read through `get_intermediate_layers`, not by calling it);
- the feature is exactly the eval main's frozen feature: `extract_features`
  with the shipped config's `n_last_blocks` and `avgpool_patchtokens`. The
  official recipe this port ships is `n_last_blocks=4, avgpool_patchtokens=0`,
  which **concatenates the [CLS] token from each of the last four blocks**
  (each after the final norm), giving one `embed_dim * n_last_blocks`-d vector
  per image -- 1536-d for vit_small (measured: embed_dim 384 x 4);
- images go through iBOT's deterministic eval pipeline: bicubic resize to 256,
  centre crop to 224, [0,1], **ImageNet** mean/std normalisation, no
  augmentation. iBOT ships no per-split loader helper (its `get_dataloaders`
  builds a fixed train+val pair); the loader is built here over
  `data_root/split` with the eval module's exact val transform, reusing its
  normalisation constants so those stay in one place;
- features are the raw encoder output (`extract_features`), *before* the
  probe's mean-centre + L2-normalise. Raw features are what the visualisation
  asked for.

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

# iBOT's eval hardcodes the ImageNet val pipeline at img_size 224 (its
# get_dataloaders default); the shipped linear_eval config carries no img_size.
_IMAGE_SIZE = 224


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim*n_last_blocks)
    raw encoder output (the teacher ViT's last-N-blocks CLS tokens concatenated
    under the shipped avgpool_patchtokens=0 recipe), labels is (N,) ImageFolder
    class indices, meta describes the run."""
    import torch
    from torchvision import datasets, transforms

    import provider_support
    adapter = provider_support.import_sibling(METHOD_DIR, "adapter")
    ev = provider_support.import_sibling(METHOD_DIR, "evaluate_linear")

    cfg = _load_config()
    model = cfg["model"]
    n_last_blocks = int(model["n_last_blocks"])
    avgpool_patchtokens = int(model["avgpool_patchtokens"])

    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    # load_encoder returns the bare teacher VisionTransformer; the model *is*
    # the backbone. The eval main does the same to/eval/freeze afterwards.
    encoder = adapter.load_encoder(state, cfg).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # iBOT ships no per-split loader; build one over data_root/split with the
    # eval's exact val transform (its normalisation constants reused).
    val_tf = transforms.Compose([
        transforms.Resize(
            256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(ev._IMAGENET_MEAN, ev._IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(str(Path(data_root) / split),
                                   transform=val_tf)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=int(num_workers), drop_last=False)

    feats, labels = ev.extract_features(
        encoder, loader, device,
        n_last_blocks=n_last_blocks,
        avgpool_patchtokens=avgpool_patchtokens)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": model.get("arch", "vit_small"),
        "image_size": _IMAGE_SIZE,
        "n_last_blocks": n_last_blocks,
        "avgpool_patchtokens": avgpool_patchtokens,
        "preprocessing": ("iBOT eval: bicubic resize to 256 + centre crop to "
                          "224, [0,1], ImageNet mean/std; feature is the "
                          "teacher ViT's last-4-blocks CLS tokens concatenated "
                          "(n_last_blocks=4, avgpool_patchtokens=0)"),
    }
    return feats, labels, meta
