"""Feature-extraction provider for vjepa2_ac (V-JEPA 2-AC).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how
V-JEPA 2-AC turns an image into a vector stays in one place.

vjepa2_ac is a **pure eval-only, download-backed** port: there is no
`adapter.load_encoder` and no trained `encoder.pt`. The frozen backbone is built
from a checkpoint by the eval module's `build_model`, which reads its path from
`train["ckpt"]` and loads the checkpoint's `encoder` sub-dict (its `module.`/
`backbone.` prefixes stripped) strict into the ViT it constructs from the
submodule (`third_party/vjepa2`), running V-JEPA 2's real rotary-position
attention (`use_rope=True`). So this provider sets `train["ckpt"]` to the passed
`encoder_path` -- the official `vjepa2-ac-vitg.pt` training-state checkpoint (the
sha256-pinned `backbone_artifact` download), not a trained encoder.

**The architecture is read from the checkpoint, not blindly from the config.**
The shipped `configs/linear_eval.yaml` pins `vit_giant_xformers` (embed_dim
1408); `build_model` loads the checkpoint strict, so the arch it constructs must
match the state dict. The submodule's ViT factories each have a distinct
embed_dim, so the checkpoint's `encoder`'s `patch_embed.proj.weight` first
dimension names the factory uniquely (the way `28_dinov2` / `eva02` infer their
variant from the checkpoint). For the real giant download this resolves to
`vit_giant_xformers` -- exactly the shipped config's arch -- and for a tiny CPU
smoke checkpoint (embed_dim 192) to `vit_tiny`; the model built can never
disagree with the weights it is handed. `patch_size` / `tubelet_size` /
`num_frames` / `img_size` stay the config's; with `use_rope=True` there is no
learned `pos_embed`, so the state dict is input-size-independent and a checkpoint
built at one size loads at the config's size.

- the feature is V-JEPA 2's still-image proxy: each image is resized to the
  model's square input and replicated `tubelet_size` times along a new temporal
  axis (one temporal token; never PyAV / a video dataset), run through the ViT,
  and the resulting tokens are **mean-pooled** (V-JEPA 2 carries no CLS token, so
  every token is pooled) -- one embed_dim vector per image (1408-d for the real
  vit_giant_xformers, 192-d for a tiny test checkpoint);
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: bilinear resize to a square `img_size`, [0,1], **ImageNet**
  mean/std normalisation, no augmentation), exactly as the eval main does;
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

# V-JEPA 2 ViT embed dim -> the submodule factory of that width. Every factory
# has a distinct embed_dim, so the checkpoint's width names it uniquely; the 1408
# / 1664 widths map to the xformers factories the shipped config uses (22 / 26
# heads), and build_model always drives them with use_rope=True. build_model then
# loads the checkpoint strict into the named factory.
_EMBED_TO_ARCH = {192: "vit_tiny", 384: "vit_small", 768: "vit_base",
                  1024: "vit_large", 1280: "vit_huge",
                  1408: "vit_giant_xformers", 1664: "vit_gigantic_xformers"}


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def _arch_from_checkpoint(ev, encoder_path: str) -> str:
    """The submodule factory name for a V-JEPA 2-AC checkpoint, from its width.

    The checkpoint is the official training-state dict whose `encoder` sub-dict
    carries the ViT weights `module.`-prefixed. `patch_embed.proj.weight` has
    shape [embed_dim, in_chans, tubelet, patch, patch]; its first dimension is
    the width, which names the factory `build_model` must construct so its strict
    load matches the weights."""
    import torch
    state = torch.load(str(encoder_path), map_location="cpu",
                       weights_only=False)
    if not isinstance(state, dict) or "encoder" not in state:
        raise RuntimeError(
            f"checkpoint {encoder_path} has no 'encoder' sub-dict; it is not the "
            "official V-JEPA 2-AC training-state checkpoint this port pins")
    encoder = ev._clean_key(state["encoder"])
    w = encoder.get("patch_embed.proj.weight")
    if w is None:
        raise KeyError(
            "checkpoint's encoder has no 'patch_embed.proj.weight'; cannot tell "
            "which V-JEPA 2 ViT width it is")
    embed = int(w.shape[0])
    if embed not in _EMBED_TO_ARCH:
        raise ValueError(
            f"V-JEPA 2 embed dim {embed} matches no known factory "
            f"{sorted(_EMBED_TO_ARCH)}")
    return _EMBED_TO_ARCH[embed]


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (mean of V-JEPA 2's output tokens, no CLS), labels is (N,) ImageFolder
    class indices, meta describes the run. `encoder_path` names the official
    V-JEPA 2-AC training-state checkpoint (a pinned download), not a trained
    encoder."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_vjepa2_ac")

    cfg = _load_config()
    train = dict(cfg["train"])
    image_size = int(train["img_size"])

    # Mirror the eval main: point train["ckpt"] at the checkpoint and build the
    # frozen ViT with the very same build_model. The arch is read from the
    # checkpoint's width (never blindly the shipped name), so build_model's strict
    # load matches whatever backbone we are actually handed.
    train["arch"] = _arch_from_checkpoint(ev, encoder_path)
    train["ckpt"] = str(encoder_path)

    dev = ev.resolve_device(device)
    model = ev.build_model(train, dev)   # to(device) + eval + freeze

    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, dev)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train["arch"],
        "image_size": image_size,
        "num_frames": int(train["num_frames"]),
        "tubelet_size": int(train["tubelet_size"]),
        "feature_source": (
            "V-JEPA 2-AC official pretrained ViT (built by build_model from the "
            "third_party/vjepa2 submodule, use_rope=True; arch read from the "
            "checkpoint width); NOT a trained encoder.pt -- encoder_path is the "
            "pinned vjepa2-ac-vitg.pt download"),
        "preprocessing": (
            "V-JEPA 2 eval: bilinear resize to a square img_size, [0,1], "
            "ImageNet mean/std; each image is replicated tubelet_size times "
            "along a new temporal axis (one temporal token), run through the "
            "ViT, and the tokens are mean-pooled (no CLS token) -- raw, before "
            "the probe's mean-centre + L2-normalise"),
    }
    return feats, labels, meta
