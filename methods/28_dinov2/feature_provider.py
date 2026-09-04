"""Feature-extraction provider for 28_dinov2.

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own pieces, so the knowledge of how DINOv2
turns an image into a vector stays in one place.

**How DINOv2's Step-1 (as-is) eval builds its model, measured -- not guessed**
(`evaluate_linear_dinov2.build_model` / `.extract_features` / `._build_loader`):
this is an eval-only, download-backed port. There is **no** `adapter.load_encoder`
for the as-is backbone; instead the eval main builds the pinned upstream ViT with
`build_model(official_dir, train, device)` and, when `train["ckpt"]` is a path,
loads the **official pretrained checkpoint** into it (strict: 0 missing / 0
unexpected). So the `encoder_path` this provider is handed is that official
DINOv2 checkpoint (e.g. the pinned `dinov2_vitg14_pretrain.pth`), not a trained
`encoder.pt`. This wrapper mirrors that call exactly: it sets `train["ckpt"]` to
`encoder_path` and calls the very same `build_model`.

**Which variant to build is read from the checkpoint, not the config.** The
shipped `configs/linear_eval.yaml` pins the giant `dinov2_vitg14` (1536-d); a
checkpoint fully determines its own architecture, and `build_model` requires the
architecture it constructs (`builders[train["name"]]`) to match the state dict
strictly. So the variant name is derived from the checkpoint's embed dim
(`cls_token`'s last dimension), the way `var`'s provider infers its VQVAE
architecture from its checkpoint: the model built can never disagree with the
weights it is handed, and both the real 1536-d giant and a tiny test checkpoint
load. `build_model` is otherwise invoked identically to the eval main.

- the feature is DINOv2's **CLS token** read through `extract_cls`: the official
  backbone returns a dict and its `feature_key` output (`x_norm_clstoken`, the
  released output-norm CLS vector) is taken as is -- one embed_dim vector per
  image (1536-d for the real ViT-g/14; 384-d for a ViT-S/14 test checkpoint);
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> bicubic resize to round(resolution / 0.875), centre crop
  to resolution, [0,1], **ImageNet** mean/std normalisation, no augmentation);
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

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

# DINOv2 ViT embed dim -> upstream builder name. The four variants have distinct
# embed dims, so the checkpoint's `cls_token` last dimension names it uniquely.
_EMBED_TO_NAME = {384: "dinov2_vits14", 768: "dinov2_vitb14",
                  1024: "dinov2_vitl14", 1536: "dinov2_vitg14"}


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def _variant_from_state(state: dict) -> str:
    """The upstream builder name for a DINOv2 checkpoint, from its embed dim.

    A checkpoint fully determines its architecture; `build_model` loads it strict
    into `builders[name]`, so the name must match the weights. Read the embed dim
    from `cls_token` (shape [1, 1, embed_dim])."""
    tok = state.get("cls_token")
    if tok is None:
        raise KeyError(
            "checkpoint has no 'cls_token'; cannot tell which DINOv2 variant "
            "it is. Is this an official DINOv2 backbone checkpoint?")
    embed = int(tok.shape[-1])
    if embed not in _EMBED_TO_NAME:
        raise ValueError(
            f"DINOv2 embed dim {embed} matches no known variant "
            f"{sorted(_EMBED_TO_NAME)}")
    return _EMBED_TO_NAME[embed]


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (DINOv2's `x_norm_clstoken` CLS vector), labels is (N,) ImageFolder
    class indices, meta describes the run. `encoder_path` names the official
    DINOv2 backbone checkpoint (a pinned download), not a trained encoder."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_dinov2")

    cfg = _load_config()
    train = dict(cfg["train"])
    resolution = int(train["resolution"])

    # Mirror the eval main: point train["ckpt"] at the checkpoint and build the
    # pinned upstream backbone with the very same build_model. The variant is
    # read from the checkpoint (never the giant shipped name), so build_model's
    # strict load matches whatever backbone we are actually handed.
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    train["name"] = _variant_from_state(state)
    train["ckpt"] = str(encoder_path)

    official_dir = METHOD_DIR.parent.parent / "third_party" / "dinov2"
    dev = ev.resolve_device(device)
    model = ev.build_model(official_dir, train, dev)   # freezes + evals

    _dataset, loader = ev._build_loader(
        str(data_root), split, resolution, int(batch_size), int(num_workers))
    feats, labels = ev.extract_features(model, loader, train, dev)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": train["name"],
        "image_size": resolution,
        "feature_source": (
            "DINOv2 official pretrained backbone (built by build_model, variant "
            "read from the checkpoint's embed dim); NOT a trained encoder.pt -- "
            "encoder_path is the pinned DINOv2 download"),
        "preprocessing": (
            "DINOv2 eval: bicubic resize to round(resolution/0.875) + centre "
            "crop to resolution, [0,1], ImageNet mean/std; feature is the CLS "
            "token (forward_features[x_norm_clstoken]), raw, before the probe's "
            "mean-centre + L2-normalise"),
    }
    return feats, labels, meta
