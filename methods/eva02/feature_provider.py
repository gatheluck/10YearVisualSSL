"""Feature-extraction provider for eva02 (EVA-02).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how
EVA-02 turns an image into a vector stays in one place.

**How EVA-02's eval builds its model, measured -- not guessed**
(`evaluate_linear_eva02.build_model` / `.extract_features` / `._build_loader`
/ `.run`): this is an eval-only, download-backed port. There is **no**
`adapter.load_encoder`; the eval main builds a frozen timm EVA-02 with
`build_model(train, device)` and, when `train["ckpt"]` is a path, loads that
checkpoint into the named timm architecture (`timm.create_model(train["name"],
pretrained=False, num_classes=0, img_size=train["img_size"])`, strict on the
backbone weights). So the `encoder_path` this provider is handed is the official
EVA-02 backbone checkpoint (the pinned `backbone_artifact` download), not a
trained `encoder.pt`. This wrapper mirrors that call: it sets `train["ckpt"]` to
`encoder_path` and builds through the very same `build_model`.

**Which variant to build is read from the checkpoint, not the config.** The
shipped `configs/linear_eval.yaml` pins the base EVA-02-B/14 (768-d); a
checkpoint fully determines its own architecture, and `build_model` requires the
architecture it constructs to match the state dict strictly. So the timm arch
name is derived from the checkpoint's embed dim (`patch_embed.proj.weight`'s
first dimension), the way `28_dinov2`'s provider infers its ViT variant from the
checkpoint: the model built can never disagree with the weights it is handed, and
both the real 768-d base and a tiny test checkpoint load. `img_size` stays the
config's, and the test checkpoint is built at that same size (EVA-02 has a
learned `pos_embed` sized to the token grid, so the size must match).

- the feature is the backbone's own canonical pooled feature: with
  `num_classes=0` and timm's default `global_pool='avg'`, `forward` returns the
  **mean over the patch tokens with the single CLS/prefix token excluded**
  (`num_prefix_tokens=1`), then the `fc_norm` LayerNorm. That is one embed_dim
  vector per image (768-d for the real EVA-02-B/14, 192-d for a tiny test
  checkpoint);
- images go through the method's own deterministic eval pipeline
  (`_build_loader`, driven by timm's data config for the built model:
  `resolve_model_data_config` -> resize to round(img_size / crop_pct) with the
  model's interpolation, centre crop to img_size, [0,1], the model's own
  mean/std -- EVA-02 uses a CLIP-style mean/std, not ImageNet's -- no
  augmentation), exactly as the eval main does;
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise (`normalize_features`). Raw features are what the
  visualisation asked for.

Imports are bare module names resolved through this method's directory, as the
eval module itself does. That is safe because the driver runs each method in
isolation; do not rely on this module and another method's modules coexisting in
one interpreter.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent
METHOD_NAME = METHOD_DIR.name

# EVA-02 ViT embed dim -> timm arch name. The four variants have distinct embed
# dims, so the checkpoint's `patch_embed.proj.weight` first dimension names it
# uniquely; `build_model` then loads the checkpoint strict into that arch.
_EMBED_TO_NAME = {192: "eva02_tiny_patch14_224", 384: "eva02_small_patch14_224",
                  768: "eva02_base_patch14_224", 1024: "eva02_large_patch14_224"}


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def _variant_from_state(state: dict) -> str:
    """The timm arch name for an EVA-02 checkpoint, from its embed dim.

    A checkpoint fully determines its architecture; `build_model` loads it strict
    into `timm.create_model(name)`, so the name must match the weights. Read the
    embed dim from `patch_embed.proj.weight` (shape [embed_dim, 3, p, p])."""
    w = state.get("patch_embed.proj.weight")
    if w is None:
        raise KeyError(
            "checkpoint has no 'patch_embed.proj.weight'; cannot tell which "
            "EVA-02 variant it is. Is this an official EVA-02 backbone "
            "checkpoint?")
    embed = int(w.shape[0])
    if embed not in _EMBED_TO_NAME:
        raise ValueError(
            f"EVA-02 embed dim {embed} matches no known variant "
            f"{sorted(_EMBED_TO_NAME)}")
    return _EMBED_TO_NAME[embed]


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (EVA-02's pooled feature: mean of the patch tokens, CLS excluded, then
    fc_norm), labels is (N,) ImageFolder class indices, meta describes the run.
    `encoder_path` names the official EVA-02 backbone checkpoint (a pinned
    download), not a trained encoder."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_eva02")

    cfg = _load_config()
    train = dict(cfg["train"])
    image_size = int(train["img_size"])

    # Mirror the eval main: point train["ckpt"] at the checkpoint and build the
    # frozen timm backbone with the very same build_model. The variant is read
    # from the checkpoint (never the shipped name), so build_model's strict load
    # matches whatever backbone we are actually handed.
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    state = state.get("model", state) if isinstance(state, dict) else state
    train["name"] = _variant_from_state(state)
    train["ckpt"] = str(encoder_path)

    dev = ev.resolve_device(device)
    model = ev.build_model(train, dev)   # to(device) + eval + freeze

    # The loader follows the built model's own timm data config, exactly as the
    # eval main does (EVA-02 uses a CLIP-style mean/std, not ImageNet's).
    import timm.data
    dc = timm.data.resolve_model_data_config(model)
    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers),
        mean=dc["mean"], std=dc["std"], crop_pct=dc["crop_pct"],
        interpolation=dc["interpolation"])
    feats, labels = ev.extract_features(model, loader, dev)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train["name"],
        "image_size": image_size,
        "feature_source": (
            "EVA-02 official pretrained backbone (built by build_model, timm "
            "variant read from the checkpoint's embed dim); NOT a trained "
            "encoder.pt -- encoder_path is the pinned EVA-02 download"),
        "preprocessing": (
            "EVA-02 eval: resize to round(img_size/crop_pct) + centre crop to "
            "img_size, [0,1], the model's own (CLIP-style) mean/std from timm's "
            "data config; feature is timm global_pool='avg' -- the mean of the "
            "patch tokens with the single CLS/prefix token excluded, then "
            "fc_norm, raw, before the probe's mean-centre + L2-normalise"),
    }
    return feats, labels, meta
