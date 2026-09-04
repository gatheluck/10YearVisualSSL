"""Feature-extraction provider for aimv2 (AIMv2; Fini et al., 2024).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how AIMv2
turns an image into a vector stays in one place.

**How AIMv2's eval builds its model, measured -- not guessed**
(`evaluate_linear_aimv2.build_model` / `.extract_features` / `._build_loader` /
`.run`): this is an eval-only, download-backed port (the timm-sourced sibling of
eva02 / franca). There is **no** `adapter.load_encoder`; the eval main builds a
frozen backbone with `build_model(train, device)`. That function has two shapes:
with `train["ckpt"]` set it builds the named timm architecture
(`timm.create_model(train["name"])`, e.g. `aimv2_large_patch14_224`, embed 1024,
24 blocks -- 309M params) and loads the checkpoint; with `ckpt` empty it builds
an AIMv2-style `VisionTransformer` **directly** from the config's architecture
keys (RMSNorm, SwiGLU, SiLU, no class token, average pooling over patch tokens,
no qkv/proj bias). The `encoder_path` this provider is handed is the official
AIMv2 backbone checkpoint (the pinned `backbone_artifact` download), not a
trained `encoder.pt`.

**Why this provider builds via the direct branch, and why that is faithful.**
The shipped architecture (aimv2_large, 309M params) is too big to build and
checkpoint in a CPU unit test, so -- as the timm ckpt branch fixes the size to
the named variant and cannot be shrunk -- the provider builds via the direct
branch with the architecture read *from the checkpoint*: `embed_dim`,
`patch_size`, `depth` and `img_size` are inferred from the state dict, then
`build_model` (empty `ckpt`) constructs the correctly shaped model and the
checkpoint is loaded into it. This has been **measured to be structurally
identical** to `timm.create_model('aimv2_large_patch14_224')`: same state-dict
keys and shapes, and, given the same weights, a bit-identical forward (max abs
diff 0.0). So both the real 1024-d aimv2_large and a tiny test checkpoint build
and load, and the real feature equals the one the eval main computes.

- the feature is AIMv2's own canonical pooled feature: with `num_classes=0` and
  `global_pool='avg'` (no class token, `num_prefix_tokens=0`), `forward` returns
  the **mean over the patch tokens**, one embed_dim vector per image (1024-d for
  the real aimv2_large);
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: resize to round(img_size / crop_pct) with the model's
  interpolation, centre crop to img_size, [0,1], the backbone's own mean/std --
  AIMv2 uses a **CLIP-style** mean/std, not ImageNet's, with crop_pct 1.0 and
  bicubic). Because the direct-branch model carries no `pretrained_cfg`, those
  values are read from the pinned variant's registered config
  (`timm.get_pretrained_cfg(train['name'])`) -- exactly the values the eval
  main's `resolve_model_data_config` returns for the real ckpt-branch model;
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
import math
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent
METHOD_NAME = METHOD_DIR.name


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def _arch_from_state(state: dict) -> dict:
    """The AIMv2 architecture keys read from a checkpoint's state dict.

    A checkpoint fully determines its own shape; building the direct-branch model
    at these dimensions means the model can never disagree with the weights it is
    handed. `embed_dim`/`patch_size` come from `patch_embed.proj.weight`
    (shape [embed_dim, 3, p, p]); `depth` from the number of transformer blocks;
    `img_size` from the number of position tokens (no class token, so
    num_tokens = (img_size / patch_size) ** 2)."""
    w = state.get("patch_embed.proj.weight")
    if w is None:
        raise KeyError(
            "checkpoint has no 'patch_embed.proj.weight'; cannot tell which "
            "AIMv2 architecture it is. Is this an official AIMv2 backbone "
            "checkpoint?")
    embed_dim = int(w.shape[0])
    patch_size = int(w.shape[-1])
    depth = len({int(k.split(".")[1]) for k in state
                 if k.startswith("blocks.") and k.split(".")[1].isdigit()})
    if depth == 0:
        raise ValueError("checkpoint has no transformer blocks (blocks.*)")
    pos = state.get("pos_embed")
    if pos is None:
        raise KeyError("checkpoint has no 'pos_embed'; cannot infer img_size")
    num_tokens = int(pos.shape[1])
    grid = int(round(math.sqrt(num_tokens)))
    if grid * grid != num_tokens:
        raise ValueError(
            f"pos_embed has {num_tokens} tokens, not a square grid; this "
            "provider assumes an AIMv2 ViT with no class token")
    return {"embed_dim": embed_dim, "patch_size": patch_size, "depth": depth,
            "img_size": grid * patch_size}


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (AIMv2's pooled feature -- the mean of the patch tokens, no class
    token), labels is (N,) ImageFolder class indices, meta describes the run.
    `encoder_path` names the official AIMv2 backbone checkpoint (a pinned
    download), not a trained encoder."""
    import torch

    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_aimv2")

    cfg = _load_config()
    train = dict(cfg["train"])
    name = str(train["name"])

    # Read the architecture from the checkpoint and build via build_model's
    # direct branch (empty ckpt) so both the real 1024-d aimv2_large and a tiny
    # test checkpoint construct at the right shape; then load the weights. num
    # heads is not stored in the state dict, so it follows the config (correct
    # for the pinned aimv2_large: 1024/8); guard divisibility for tiny checks.
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    state = state.get("model", state) if isinstance(state, dict) else state
    train.update(_arch_from_state(state))
    nh = int(train["num_heads"])
    if int(train["embed_dim"]) % nh != 0:
        nh = 1
    train["num_heads"] = nh
    train["ckpt"] = ""

    dev = ev.resolve_device(device)
    model = ev.build_model(train, dev)   # to(device) + eval + freeze
    missing, _unexpected = model.load_state_dict(state, strict=False)
    backbone_missing = [k for k in missing if not k.startswith("head")]
    if backbone_missing:
        raise RuntimeError(
            f"checkpoint is missing backbone weights: {backbone_missing[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # The loader follows the pinned backbone's own data config (AIMv2 uses a
    # CLIP-style mean/std, crop_pct 1.0, bicubic), read from the registered
    # pretrained config for the shipped variant name -- the direct-branch model
    # carries none, but these are exactly the values the eval main's
    # resolve_model_data_config returns for the real ckpt-branch model.
    import timm
    image_size = int(train["img_size"])
    pc = timm.get_pretrained_cfg(name)
    _dataset, loader = ev._build_loader(
        str(data_root), split, image_size, int(batch_size), int(num_workers),
        mean=pc.mean, std=pc.std, crop_pct=pc.crop_pct,
        interpolation=pc.interpolation)
    feats, labels = ev.extract_features(model, loader, dev)

    feats = feats.numpy()
    labels = labels.numpy()
    meta = {
        "method": METHOD_NAME,
        "representation": "raw",
        "feat_dim": int(feats.shape[1]),
        "count": int(feats.shape[0]),
        "arch": name,
        "image_size": image_size,
        "feature_source": (
            "AIMv2 official pretrained backbone (built by build_model's direct "
            "branch at the architecture read from the checkpoint, then the "
            "checkpoint loaded; verified structurally identical to "
            "timm.create_model('aimv2_large_patch14_224')). NOT a trained "
            "encoder.pt -- encoder_path is the pinned AIMv2 download"),
        "preprocessing": (
            "AIMv2 eval: resize to round(img_size/crop_pct) + centre crop to "
            "img_size, [0,1], the backbone's own (CLIP-style) mean/std from "
            "timm's registered pretrained config (crop_pct 1.0, bicubic); "
            "feature is global_pool='avg' -- the mean of the patch tokens (no "
            "class token), raw, before the probe's mean-centre + L2-normalise"),
    }
    return feats, labels, meta
