"""CLIP model construction, imported from the pinned `openai/CLIP` submodule.

The official CLIP model, its BPE tokenizer and its ImageNet prompt metadata are
referenced from `third_party/CLIP` (pinned, MIT) and **never copied**. openai/CLIP
exposes a *top-level* ``clip`` package, whose name collides with the PyPI ``clip``
distribution and with any other top-level package a sibling submodule might expose;
so the import is done lazily, inside the build functions, after purging ``clip``
from ``sys.modules`` and stripping every other ``third_party/`` root from
``sys.path`` -- the same collision-safe pattern the other submodule-import ports
use for their own top-level packages.

Two towers, two uses:
- **Step 2 (pretrain)** builds the full CLIP (image tower + text tower) with
  ``build_clip`` and trains it; the saved ``encoder.pt`` is only the image tower
  (``visual.*``).
- **Step 1 / Step 2 probe** builds just the image tower: a random one for the
  hermetic smoke (``build_clip_visual``), or the official ViT-B/32 loaded from the
  sha256-pinned download (``load_official_vit_b32``).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType

import torch

# The pinned upstream (see provenance.json upstream.commit and the adapter UPSTREAM).
OFFICIAL_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
# sha256 of the released OpenAI ViT-B/32 weight (research.md; the Step-1 backbone).
OFFICIAL_VIT_B32_SHA256 = (
    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)
# The official CLIP evaluation normalisation (clip/clip.py `_transform`).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Stamped into every Step-2 checkpoint; a supervised label-text adaptation, not
# unlabeled VSSL (README, provenance.json).
STEP2_PROTOCOL = "clip_imagenet_label_text_vitb16_horizon300_v1"

_CLIP_SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "CLIP"


def sha256_file(path: "str | Path", chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_clip_path() -> None:
    """Make the pinned submodule's ``clip`` package the one that imports.

    Purge any cached ``clip`` (the PyPI package or a stale import), strip other
    ``third_party/`` roots so a sibling submodule cannot shadow it, then put this
    submodule first on the path."""
    for key in [k for k in sys.modules if k == "clip" or k.startswith("clip.")]:
        del sys.modules[key]
    third_party = str(_CLIP_SUBMODULE.parent) + os.sep
    sys.path[:] = [q for q in sys.path if not q.startswith(third_party)]
    sys.path.insert(0, str(_CLIP_SUBMODULE))


def load_official_clip_package() -> ModuleType:
    """Import the pinned OpenAI ``clip`` package (lazily, collision-safe)."""
    _prepare_clip_path()
    init_path = _CLIP_SUBMODULE / "clip" / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(
            f"the pinned OpenAI CLIP submodule is missing: {init_path}. Run "
            "`git submodule update --init third_party/CLIP`.")
    import clip  # noqa: E402  -- the submodule package, not the PyPI distribution
    return clip


def build_clip(model_cfg: dict):
    """The full CLIP (image + text towers) for Step-2 training.

    Dimensions come from the config so the hermetic smoke can build a tiny CLIP;
    the shipped Step-2 config is the official ViT-B/16 (vision_width 768, patch 16,
    embed_dim 512, a 12-layer text tower). ``vocab_size``/``context_length`` are
    kept at the tokenizer's own 49408/77 even in the smoke, so the real BPE
    tokenizer's ids index the text embedding."""
    official = load_official_clip_package()
    return official.model.CLIP(
        embed_dim=int(model_cfg["embed_dim"]),
        image_resolution=int(model_cfg["image_resolution"]),
        vision_layers=int(model_cfg["vision_layers"]),
        vision_width=int(model_cfg["vision_width"]),
        vision_patch_size=int(model_cfg["vision_patch_size"]),
        context_length=int(model_cfg["context_length"]),
        vocab_size=int(model_cfg["vocab_size"]),
        transformer_width=int(model_cfg["transformer_width"]),
        transformer_heads=int(model_cfg["transformer_heads"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
    )


def build_clip_visual(vision_cfg: dict):
    """Just the CLIP image tower (a ``VisionTransformer``).

    Used to rebuild the trained Step-2 encoder from ``encoder.pt`` and to build the
    random tiny image tower for the Step-1 hermetic smoke. ``heads`` defaults to
    ``width // 64`` -- CLIP's own convention."""
    official = load_official_clip_package()
    width = int(vision_cfg["width"])
    heads = int(vision_cfg.get("heads", width // 64))
    return official.model.VisionTransformer(
        input_resolution=int(vision_cfg["resolution"]),
        patch_size=int(vision_cfg["patch_size"]),
        width=width,
        layers=int(vision_cfg["layers"]),
        heads=heads,
        output_dim=int(vision_cfg["output_dim"]),
    )


def load_official_vit_b32(
    checkpoint: "str | Path",
    device: "torch.device",
    *,
    verify_checksum: bool = True,
):
    """Load the released OpenAI ViT-B/32 through the pinned ``clip.load``.

    Returns the full CLIP model; the caller takes ``.visual`` as the frozen image
    tower. The checksum is verified against the pinned sha256 unless the caller has
    already verified it."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"official ViT-B/32 checkpoint is missing: {checkpoint}. Fetch it with "
            "bin/fetch-weights.py (provenance.json backbone_artifact).")
    if verify_checksum:
        actual = sha256_file(checkpoint)
        if actual != OFFICIAL_VIT_B32_SHA256:
            raise RuntimeError(
                f"official ViT-B/32 checksum mismatch: "
                f"expected={OFFICIAL_VIT_B32_SHA256} actual={actual} "
                f"path={checkpoint}")
    official = load_official_clip_package()
    model, _preprocess = official.load(str(checkpoint), device=device, jit=False)
    return model


def load_official_imagenet_metadata() -> "tuple[list[str], list[str]]":
    """The official 1000 ImageNet class names and 80 prompt templates.

    Read verbatim from the pinned notebook
    ``notebooks/Prompt_Engineering_for_ImageNet.ipynb`` in the submodule, so the
    Step-2 label-text prompts are exactly OpenAI's."""
    import ast
    import json

    notebook_path = (
        _CLIP_SUBMODULE / "notebooks" / "Prompt_Engineering_for_ImageNet.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    def _literal(cell_source: str, name: str):
        try:
            tree = ast.parse(cell_source)
        except SyntaxError:
            return None
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                try:
                    return ast.literal_eval(node.value)
                except (TypeError, ValueError, SyntaxError):
                    return None
        return None

    classes = templates = None
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        classes = classes or _literal(source, "imagenet_classes")
        templates = templates or _literal(source, "imagenet_templates")
    if not isinstance(classes, list) or len(classes) != 1000:
        raise RuntimeError(
            f"expected 1000 official ImageNet class names, got {type(classes)}")
    if not isinstance(templates, list) or len(templates) < 2:
        raise RuntimeError(
            f"official ImageNet prompt templates were not found: {type(templates)}")
    if not all(isinstance(v, str) for v in (*classes, *templates)):
        raise RuntimeError("official ImageNet metadata contains non-string values")
    return classes, templates


def tokenize_prompts(
    class_names: "list[str]", templates: "list[str]"
) -> torch.Tensor:
    """Tokenize ``templates`` x ``class_names`` -> LongTensor [C, T, context_length].

    Uses the pinned submodule's BPE tokenizer (``clip.tokenize``); the vocab ships
    inside the submodule (``clip/bpe_simple_vocab_16e6.txt.gz``)."""
    official = load_official_clip_package()
    rows = []
    for name in class_names:
        rows.append(official.tokenize([t.format(name) for t in templates]))
    return torch.stack(rows, dim=0).to(torch.long)
