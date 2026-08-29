"""BEiT v2 backbone construction, imported from the pinned `microsoft/unilm` submodule.

The author's finetune Vision Transformer (`modeling_finetune.VisionTransformer`)
is referenced from `third_party/unilm/beit2` (pinned, MIT) and **never copied**.
beit2 is a directory of scripts, not a package, so it exposes a *top-level* module
name (`modeling_finetune`) that could collide with another submodule; the import is
therefore done lazily, inside the build functions, after purging that name from
`sys.modules` and stripping every other `third_party/` root from `sys.path` -- the
same collision-safe pattern the other submodule-import ports use.

Two things are fixed for BEiT v2 base and are not config settings -- they are the
architecture the released `pt1k` checkpoint was trained with (beit2's
`run_class_finetuning.py` defaults): relative position bias on, absolute position
embedding off, a mean-pooling head, and a layer-scale init of 0.1. The config keeps
only the size knobs (embed_dim/depth/num_heads/patch_size/img_size) so the hermetic
smoke can shrink them.

The probed feature is the model's own canonical global feature: with
`num_classes=0` and `use_mean_pooling=True`, `forward` returns
`fc_norm(patch_tokens.mean(1))` -- the mean over patch tokens (the CLS token at
position 0 is excluded), then the finetune LayerNorm. The `pt1k` checkpoint is a
*pre-finetune* self-supervised checkpoint, so it carries no `fc_norm`; that layer
stays at its identity-affine init (weight 1, bias 0), a deterministic normalisation.
"""

from __future__ import annotations

import hashlib
import os
import sys
from functools import partial
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

# The pinned upstream (see provenance.json upstream.commit and the adapter UPSTREAM).
OFFICIAL_COMMIT = "ca43e4cd19445a536f133bf2bc25b573b2f0c7c5"
# The BEiT v2 finetune-eval normalisation (beit2 datasets.py build_transform with
# --imagenet_default_mean_and_std): the standard ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The eval resize/crop ratio (beit2 build_transform: crop_pct = 224/256 at 224px).
CROP_PCT = 224.0 / 256.0
# BEiT v2 base architecture constants (run_class_finetuning.py defaults for the
# released pt1k checkpoint); not config settings, the checkpoint requires them.
INIT_VALUES = 0.1
USE_REL_POS_BIAS = True
USE_ABS_POS_EMB = False
USE_MEAN_POOLING = True

_BEIT2_SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "unilm" / "beit2"
# The top-level module names beit2 exposes that we might shadow / be shadowed by.
_BEIT2_TOPLEVEL = ("modeling_finetune",)


def sha256_file(path: "str | Path", chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_beit2_path() -> None:
    """Make the pinned submodule's `modeling_finetune` the one that imports.

    Purge any cached `modeling_finetune` (a stale import or a sibling submodule's
    same-named module), strip other `third_party/` roots so a sibling cannot shadow
    it, then put this submodule first on the path."""
    for name in _BEIT2_TOPLEVEL:
        for key in [k for k in sys.modules
                    if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
    third_party = str(_BEIT2_SUBMODULE.parents[1]) + os.sep
    sys.path[:] = [q for q in sys.path if not q.startswith(third_party)]
    sys.path.insert(0, str(_BEIT2_SUBMODULE))


def load_beit2_module() -> ModuleType:
    """Import the pinned beit2 `modeling_finetune` module (lazily, collision-safe)."""
    module_path = _BEIT2_SUBMODULE / "modeling_finetune.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"the pinned BEiT v2 submodule is missing: {module_path}. Run "
            "`git submodule update --init third_party/unilm`.")
    _prepare_beit2_path()
    import modeling_finetune  # noqa: E402 -- the submodule module, not a pip package
    return modeling_finetune


def build_beit_vit(model_cfg: dict):
    """A BEiT v2 finetune `VisionTransformer` with `num_classes=0`.

    Dimensions come from the config so the hermetic smoke can build a tiny model;
    the shipped config is the official base (embed_dim 768, depth 12, 12 heads,
    patch 16, 224px). The architecture constants above are fixed."""
    mf = load_beit2_module()
    return mf.VisionTransformer(
        img_size=int(model_cfg["img_size"]),
        patch_size=int(model_cfg["patch_size"]),
        in_chans=3,
        num_classes=0,
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=INIT_VALUES,
        use_rel_pos_bias=USE_REL_POS_BIAS,
        use_abs_pos_emb=USE_ABS_POS_EMB,
        use_mean_pooling=USE_MEAN_POOLING,
    )


def load_pt1k_checkpoint(model, checkpoint: "str | Path") -> None:
    """Load the official pt1k checkpoint into `model`, applying beit2's own surgery.

    Replicates the essential, resolution-independent parts of
    `run_class_finetuning.py`'s `--finetune` loading: take the `model` sub-dict,
    expand the *shared* relative position bias table into a per-block table (the
    finetune model uses per-block bias), drop the `relative_position_index` buffers
    (the model regenerates them), then load non-strictly. At the pretraining
    resolution (224) the position-bias / position-embedding interpolation branches
    are no-ops, so they are not reproduced here. The only backbone weight the pt1k
    checkpoint lacks is `fc_norm` (a finetune-only layer), which stays at its init.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"BEiT v2 checkpoint is missing: {checkpoint}. Fetch it with "
            "bin/fetch-weights.py (provenance.json backbone_artifact).")
    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = raw.get("model", raw.get("module", raw)) if isinstance(raw, dict) else raw

    if getattr(model, "use_rel_pos_bias", False) and \
            "rel_pos_bias.relative_position_bias_table" in state:
        shared = state.pop("rel_pos_bias.relative_position_bias_table")
        for i in range(model.get_num_layers()):
            state[f"blocks.{i}.attn.relative_position_bias_table"] = shared.clone()
    for key in [k for k in list(state) if "relative_position_index" in k]:
        state.pop(key)

    missing, _unexpected = model.load_state_dict(state, strict=False)
    # num_classes=0 drops the head, and pt1k (pre-finetune) has no fc_norm; any
    # other missing weight means the checkpoint does not match the architecture.
    backbone_missing = [k for k in missing
                        if not k.startswith("head")
                        and not k.startswith("fc_norm")
                        and "relative_position_index" not in k]
    if backbone_missing:
        raise RuntimeError(
            f"checkpoint is missing backbone weights: {backbone_missing[:5]}")
