"""Feature-extraction provider for cosmos3_super (NVIDIA Cosmos3-Super).

`bin/extract-features.py` discovers this file and calls `extract_val_features`
to obtain one raw feature vector per image over a dataset split. It is a thin
wrapper that reuses this method's own eval pieces, so the knowledge of how
Cosmos3-Super's vision tower turns an image into a vector stays in one place:

- this is an **eval-only** port (the `transformers`-sourced sibling of `sam3`
  and `data2vec2`): there is no `encoder.pt` from training and no
  `adapter.load_encoder`. The frozen backbone is built by the eval module's
  `build_model`, which reads its checkpoint from `train["ckpt"]`. Unlike sam3,
  the released checkpoint is **directly loadable** by the HF class -- a real run
  calls `Qwen3VLVisionModel.from_pretrained(local_files_only=True)` on the
  released `vision_encoder/` **directory** (config.json + model.safetensors), so
  no trunk conversion is needed;
- the driver hands the provider a single-file encoder path and hashes that file
  (`bin/extract-features.py:sha256_of`, which requires a file, not a directory).
  The released `backbone_artifact` is exactly that single file,
  `vision_encoder/model.safetensors`. So the provider accepts the
  `model.safetensors` file, takes its **parent directory** (the `vision_encoder/`
  layout `from_pretrained` reads) as `train["ckpt"]`, and lets `build_model`
  load it. A directory handed in directly is also accepted, so both the driver's
  single-file contract and a directly-passed `vision_encoder/` work;
- the feature is the vision tower's patch tokens **mean-pooled over the
  sequence**; one `hidden_size`-d vector per image (1152 for the released tower,
  not the merger's `out_hidden_size` 5120);
- images go through the method's own deterministic eval pipeline
  (`_build_loader` -> bicubic resize to `img_size` (448, the shipped recipe),
  centre crop to the same, [0,1], **ImageNet** mean/std normalisation, no
  augmentation);
- features are the raw encoder output (`extract_features`), *before* the probe's
  mean-centre + L2-normalise. Raw features are what the visualisation asked for.

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
    """Return (features, labels, meta): features is (N, hidden_size) raw encoder
    output (mean of the Qwen3-VL vision tower's patch tokens), labels is (N,)
    ImageFolder class indices, meta describes the run."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    ev = importlib.import_module("evaluate_linear_cosmos3_super")

    cfg = _load_config()
    train = dict(cfg["train"])
    # `build_model` loads with `from_pretrained` on the vision_encoder/ directory.
    # The driver hands (and hashes) the single `model.safetensors` file; use its
    # parent directory. A directory handed in directly is used as-is.
    enc = Path(encoder_path)
    ckpt_dir = enc if enc.is_dir() else enc.parent
    train["ckpt"] = str(ckpt_dir)
    image_size = int(train["img_size"])

    dev = ev.torch.device(device)
    model = ev.build_model(train, dev)
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
        "arch": train.get("name", "nvidia/Cosmos3-Super"),
        "image_size": image_size,
        "preprocessing": ("Cosmos3-Super eval: bicubic resize + centre crop to "
                          "img_size, [0,1], ImageNet mean/std; feature is the "
                          "mean of the Qwen3-VL vision tower's patch tokens"),
    }
    return feats, labels, meta
