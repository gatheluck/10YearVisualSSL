"""Feature-extraction provider for cae (Context Autoencoder).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how CAE
turns an image into a vector stays in one place.

**How CAE's eval builds its model, measured -- not guessed**
(`evaluate_linear_cae.build_model` / `.extract_features` / `._build_loader`):
this is a **pure eval-only**, download-backed port. There is **no**
`adapter.load_encoder`; the eval main builds a frozen, self-contained BEiT-style
ViT with `build_model(train, device)`, which -- when `train["ckpt"]` is a path --
loads that checkpoint's `backbone.*` tensors into the ViT (strict: any missing or
unexpected backbone key is a hard error). So the `encoder_path` this provider is
handed is the official CAE backbone checkpoint (the pinned OpenMMLab mmselfsup
reproduction), not a trained `encoder.pt`. This wrapper mirrors that call: it sets
`train["ckpt"]` to `encoder_path` and builds through the very same `build_model`.

The architecture is the shipped config's own keys (`img_size`, `patch_size`,
`embed_dim`, `depth`, `num_heads`); `build_model` builds exactly that ViT and the
checkpoint's `backbone.*` tensors must match it strictly, so the built model can
never disagree with the weights it is handed.

- the feature is the backbone's own canonical global feature: `CAEViT.forward`
  returns `x[:, 0]` -- the **final-LayerNorm'd CLS token** (position 0), the
  representation CAE's linear probe is fit on. That is one embed_dim vector per
  image (768-d for the real CAE ViT-B/16);
- images go through the method's own deterministic eval pipeline
  (`_build_loader`: a bicubic **square resize** to img_size, **no centre crop**,
  [0,1], ImageNet mean/std -- the constants `CAE_MEAN`/`CAE_STD` the eval main
  uses -- no augmentation), exactly as the eval main does;
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


def _load_config() -> dict:
    import yaml
    with open(METHOD_DIR / "configs" / "linear_eval.yaml") as f:
        return yaml.safe_load(f)


def extract_val_features(*, encoder_path: str, data_root: str, split: str,
                         device: str, batch_size: int, num_workers: int):
    """Return (features, labels, meta): features is (N, embed_dim) raw encoder
    output (CAE's final-norm'd CLS token), labels is (N,) ImageFolder class
    indices, meta describes the run. `encoder_path` names the official CAE
    backbone checkpoint (a pinned download), not a trained encoder."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_cae")

    cfg = _load_config()
    train = dict(cfg["train"])
    train["ckpt"] = str(encoder_path)
    image_size = int(train["img_size"])

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
        "arch": train.get("name", "cae"),
        "image_size": image_size,
        "feature_source": (
            "CAE official pretrained backbone (built by build_model, a "
            "self-contained BEiT-style ViT with the checkpoint's backbone.* "
            "tensors loaded strict); NOT a trained encoder.pt -- encoder_path "
            "is the pinned OpenMMLab mmselfsup reproduction download"),
        "preprocessing": (
            "CAE eval: bicubic square resize to img_size (no centre crop), "
            "[0,1], ImageNet mean/std (CAE_MEAN/CAE_STD); feature is the "
            "final-LayerNorm'd CLS token, raw, before the probe's mean-centre "
            "+ L2-normalise"),
    }
    return feats, labels, meta
