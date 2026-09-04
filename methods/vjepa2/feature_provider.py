"""Feature-extraction provider for vjepa2 (V-JEPA 2).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how
V-JEPA 2 turns an image into a vector stays in one place.

**How V-JEPA 2's eval builds its model, measured -- not guessed**
(`evaluate_linear_vjepa2.build_model` / `.extract_features` / `._build_loader`
/ `.extract_feature`): this is a **pure eval-only** port, the frozen-backbone
sibling of eva02 / sam3 / data2vec2 / videomae. There is **no**
`adapter.load_encoder`; the eval main builds a frozen self-contained V-JEPA 2
ViT with `build_model(train, device)`, and when `train["ckpt"]` is a path it
loads that checkpoint's `encoder.*` tensors into the ViT (dropping the
`predictor.*` JEPA-predictor keys, then a strict missing/unexpected-key check).
So the `encoder_path` this provider is handed is the official
`facebook/vjepa2-vitl-fpc64-256` backbone checkpoint (a sha256-pinned
safetensors download), not a trained `encoder.pt`. This wrapper mirrors that
call: it sets `train["ckpt"]` to `encoder_path` and builds through the very same
`build_model`.

**The image-val contract holds: V-JEPA 2's eval consumes still images, not
video.** V-JEPA 2's ViT expects a clip `(B, 3, T, H, W)` via a Conv3d tubelet
patch embed, but this port's ImageNet linear eval never opens a video file: it
uses `torchvision.datasets.ImageFolder` (`_build_loader`) and replicates each
still image `num_frames` times along a new temporal axis to form the clip
(`extract_feature`), then mean-pools the tokens. This provider mirrors that
exactly, so the feature is the same one the linear probe is fit on.

**Which architecture to build is read from the checkpoint, not the config.**
The shipped `configs/linear_eval.yaml` pins the official ViT-L/16 (embed_dim
1024, depth 24). `build_model` builds strictly from architecture keys and loads
the checkpoint strict, so those keys must match the weights handed in. As
`eva02` / `28_dinov2` do, the architecture-shaping keys (`embed_dim`,
`tubelet_size`, `patch_size` from `encoder.embeddings.patch_embeddings.proj.weight`
of shape `(embed_dim, 3, tubelet_size, patch_size, patch_size)`, and `depth`
from the `encoder.layer.N.*` indices) are read from the checkpoint, so the model
built can never disagree with the weights it is handed -- the real ViT-L
checkpoint reproduces the shipped config's architecture exactly, and a tiny test
checkpoint loads too. `num_heads`, `num_frames` and `img_size` stay the config's
(V-JEPA 2 holds no learned position parameters -- rotary at run time, plain here
-- so `num_frames`/`img_size` do not shape the state dict, and `num_heads` is
not stored in it; `embed_dim` must be divisible by the config's `num_heads`).

- the feature is the backbone's own representation: the ViT has **no CLS token**;
  `extract_feature` runs the encoder and returns the **mean over all patch
  tokens** (after the final LayerNorm) -- one `embed_dim`-d vector per image
  (1024-d for the real ViT-L);
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: bicubic square resize to `img_size`, [0,1], **ImageNet**
  mean/std normalisation (`VJEPA2_MEAN`/`VJEPA2_STD`), no augmentation), each
  image then replicated to a `num_frames`-frame clip;
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise (`normalize_features`). Raw features are what the
  visualisation asked for.

A faithfulness note carried from the port: this backbone runs plain
(non-rotary) attention -- it loads the weights and mean-pools without
re-deriving V-JEPA 2's rotary positional mechanism, mirroring the capture's
forward exactly (see `evaluate_linear_vjepa2` and `provenance.json`).

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


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def _arch_from_checkpoint(encoder_path: str) -> dict:
    """The architecture-shaping keys for a V-JEPA 2 checkpoint, from its shapes.

    A checkpoint fully determines its own architecture; `build_model` builds
    strictly from the config's keys and loads the checkpoint strict, so the keys
    must match the weights. Read shapes only (no full tensor load, so the 1.3 GB
    official file is not materialised) via safetensors' lazy `safe_open`:
      - `encoder.embeddings.patch_embeddings.proj.weight` has shape
        `(embed_dim, 3, tubelet_size, patch_size, patch_size)`;
      - `depth` is one past the largest `encoder.layer.N.*` index.
    """
    from safetensors import safe_open

    proj = "encoder.embeddings.patch_embeddings.proj.weight"
    with safe_open(str(encoder_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if proj not in keys:
            raise KeyError(
                f"checkpoint {encoder_path} has no {proj!r}; cannot tell its "
                "V-JEPA 2 architecture. Is this the official "
                "facebook/vjepa2-vitl-fpc64-256 checkpoint?")
        shape = tuple(f.get_slice(proj).get_shape())
    embed_dim, _c, tubelet_size, patch_size, _p = shape
    layers = [int(k.split(".")[2]) for k in keys
              if k.startswith("encoder.layer.")]
    if not layers:
        raise KeyError(
            f"checkpoint {encoder_path} has no encoder.layer.N.* blocks")
    depth = max(layers) + 1
    return {"embed_dim": int(embed_dim), "tubelet_size": int(tubelet_size),
            "patch_size": int(patch_size), "depth": int(depth)}


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (V-JEPA 2's mean over all patch tokens, the ViT has no CLS), labels
    is (N,) ImageFolder class indices, meta describes the run. `encoder_path`
    names the official V-JEPA 2 backbone checkpoint (a pinned safetensors
    download), not a trained encoder."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_vjepa2")

    cfg = _load_config()
    train = dict(cfg["train"])
    # Mirror the eval main: point train["ckpt"] at the checkpoint and build the
    # frozen ViT with the very same build_model. The architecture-shaping keys
    # are read from the checkpoint (never the shipped ones), so build_model's
    # strict load matches whatever backbone we are actually handed.
    train.update(_arch_from_checkpoint(encoder_path))
    train["ckpt"] = str(encoder_path)
    image_size = int(train["img_size"])
    num_frames = int(train["num_frames"])

    dev = ev.resolve_device(device)
    model = ev.build_model(train, dev)   # to(device) + eval + freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

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
        "arch": train.get("name", "facebook/vjepa2-vitl-fpc64-256"),
        "image_size": image_size,
        "num_frames": num_frames,
        "feature_source": (
            "V-JEPA 2 official pretrained backbone (built by build_model, arch "
            "shape read from the checkpoint's encoder.* keys); NOT a trained "
            "encoder.pt -- encoder_path is the pinned V-JEPA 2 download"),
        "preprocessing": (
            "V-JEPA 2 eval: bicubic square resize to img_size, [0,1], ImageNet "
            "mean/std; each still image is replicated num_frames times along a "
            "new temporal axis to form the clip; feature is the mean over all "
            "patch tokens (the ViT has no CLS), raw, before the probe's "
            "mean-centre + L2-normalise. Plain (non-rotary) attention, "
            "mirroring the capture's forward."),
    }
    return feats, labels, meta
