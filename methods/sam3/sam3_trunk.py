"""Load official facebook/sam3 trunk weights into transformers Sam3ViTModel.

The published `sam3.pt` stores its vision encoder under
`detector.backbone.vision_backbone.trunk.*` (ViTDet style: fused `attn.qkv`, an
optional CLS row in `pos_embed`, a bias on the patch-embed convolution). The HF
`Sam3ViTModel` uses split q/k/v projections, no CLS, and a bias-free patch
projection. This converter maps the official tensors onto that architecture so the
backbone is not left randomly initialised.

RoPE (`freqs_cis` / `rope_embeddings_*`) is not copied: transformers recomputes
the 2D axial rotary tables from `rope_theta` and the window size, matching the
official HF conversion note. The converter is a pure function of the state dict;
it is exercised on synthetic tensors in `tests/test_method_sam3.py`
(`TestTheTrunkConverter`), so the real-run path is covered without the gated
weights.

This mirrors the capture's `methods_step3/SegFM/SAM3/sam3_trunk.py`, kept to the
frozen-backbone read the port needs.
"""

from __future__ import annotations

from pathlib import Path

import torch


# The official trunk may be nested under any of these prefixes, longest first.
PREFIXES = (
    "detector.backbone.vision_backbone.trunk.",
    "backbone.vision_backbone.trunk.",
    "vision_backbone.trunk.",
    "trunk.",
)


def _load_raw(weight_path: str) -> dict:
    """Read a `sam3.pt` (or `model.safetensors`) into a flat tensor dict."""
    path = Path(weight_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj \
            and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return obj


def _strip_prefix(sd: dict) -> dict:
    """Return the trunk-relative tensors, or refuse a file that has none."""
    for prefix in PREFIXES:
        trunk = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if trunk:
            return trunk
    # Already trunk-relative?
    if any(k.startswith("blocks.") or k.startswith("patch_embed.") for k in sd):
        return sd
    sample = list(sd.keys())[:12]
    raise RuntimeError(f"No SAM3 trunk keys found. sample={sample}")


def convert_official_trunk_to_hf(trunk_sd: dict) -> dict:
    """Map official ViTDet-style trunk tensors to Sam3ViTModel names."""
    out = {}

    proj_w = trunk_sd.get("patch_embed.proj.weight")
    if proj_w is None:
        raise RuntimeError("Missing patch_embed.proj.weight in official trunk")
    out["embeddings.patch_embeddings.projection.weight"] = proj_w
    # HF projection is bias-free; a non-zero official bias is recorded (not
    # silently dropped) so the caller can decide.
    if "patch_embed.proj.bias" in trunk_sd:
        bias = trunk_sd["patch_embed.proj.bias"]
        if float(bias.abs().max()) > 0:
            out["_unused_patch_embed_bias"] = bias

    pos = trunk_sd.get("pos_embed")
    if pos is None:
        raise RuntimeError("Missing pos_embed in official trunk")
    # Official often stores CLS + patches as [1, 1+N, D]; HF drops the CLS row.
    # A perfect square token count is already CLS-free.
    if pos.dim() == 3:
        n = pos.shape[1]
        root = int(round(n ** 0.5))
        if root * root != n and (root_m1 := int(round((n - 1) ** 0.5))) \
                and root_m1 * root_m1 == n - 1:
            pos = pos[:, 1:, :]
    elif pos.dim() == 4:
        # [1, H, W, D] or [1, D, H, W] -> [1, N, D]
        if pos.shape[-1] not in (pos.shape[1],):
            pos = pos.reshape(1, -1, pos.shape[-1])
        else:
            pos = pos.flatten(2).transpose(1, 2)
    out["embeddings.position_embeddings"] = pos

    if "ln_pre.weight" in trunk_sd:
        out["layer_norm.weight"] = trunk_sd["ln_pre.weight"]
        out["layer_norm.bias"] = trunk_sd["ln_pre.bias"]

    block_ids = sorted({int(k.split(".")[1]) for k in trunk_sd
                        if k.startswith("blocks.") and k.split(".")[1].isdigit()})
    for i in block_ids:
        p = f"blocks.{i}."
        hf = f"layers.{i}."
        out[hf + "layer_norm1.weight"] = trunk_sd[p + "norm1.weight"]
        out[hf + "layer_norm1.bias"] = trunk_sd[p + "norm1.bias"]
        out[hf + "layer_norm2.weight"] = trunk_sd[p + "norm2.weight"]
        out[hf + "layer_norm2.bias"] = trunk_sd[p + "norm2.bias"]
        out[hf + "mlp.fc1.weight"] = trunk_sd[p + "mlp.fc1.weight"]
        out[hf + "mlp.fc1.bias"] = trunk_sd[p + "mlp.fc1.bias"]
        out[hf + "mlp.fc2.weight"] = trunk_sd[p + "mlp.fc2.weight"]
        out[hf + "mlp.fc2.bias"] = trunk_sd[p + "mlp.fc2.bias"]

        qkv_w = trunk_sd[p + "attn.qkv.weight"]
        qkv_b = trunk_sd[p + "attn.qkv.bias"]
        dim = qkv_w.shape[0] // 3
        qw, kw, vw = qkv_w.split(dim, dim=0)
        qb, kb, vb = qkv_b.split(dim, dim=0)
        out[hf + "attention.q_proj.weight"] = qw
        out[hf + "attention.k_proj.weight"] = kw
        out[hf + "attention.v_proj.weight"] = vw
        out[hf + "attention.q_proj.bias"] = qb
        out[hf + "attention.k_proj.bias"] = kb
        out[hf + "attention.v_proj.bias"] = vb
        out[hf + "attention.o_proj.weight"] = trunk_sd[p + "attn.proj.weight"]
        out[hf + "attention.o_proj.bias"] = trunk_sd[p + "attn.proj.bias"]

    return out


def load_official_trunk(weight_path: str, img_size: int = 336):
    """Build a frozen-ready `Sam3ViTModel` and load `weight_path`'s official
    trunk into it.

    The architecture is **inferred from the converted checkpoint itself** --
    `hidden_size` from the position embedding's width, `num_hidden_layers` from
    the block count, `intermediate_size` from the first block's MLP width -- so
    the released ViT-L (whose `intermediate_size` is 4736, not 4x hidden) loads
    without a size mismatch. A checkpoint that is missing a backbone weight after
    conversion is refused, not half-loaded. `img_size` sets `image_size` (the
    executable input resolution); `pretrain_image_size` is derived from the token
    count so the position table matches. Mirrors the capture's
    `methods_step3/SegFM/SAM3/sam3_trunk.py`."""
    from transformers import Sam3ViTConfig, Sam3ViTModel

    raw = _load_raw(weight_path)
    trunk = _strip_prefix(raw)
    converted = convert_official_trunk_to_hf(trunk)
    converted.pop("_unused_patch_embed_bias", None)

    hidden = int(converted["embeddings.position_embeddings"].shape[-1])
    n_patches = int(converted["embeddings.position_embeddings"].shape[1])
    pretrain = int(n_patches ** 0.5) * 14
    depth = 1 + max(int(k.split(".")[1]) for k in converted
                    if k.startswith("layers."))
    mlp_hidden = int(converted["layers.0.mlp.fc1.weight"].shape[0])

    cfg = Sam3ViTConfig(
        hidden_size=hidden,
        num_hidden_layers=depth,
        num_attention_heads=hidden // 64,
        intermediate_size=mlp_hidden,
        patch_size=14,
        image_size=img_size,
        pretrain_image_size=pretrain,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        window_size=24,
        global_attn_indexes=[7, 15, 23, 31],
        rope_theta=10000.0,
        hidden_dropout=0.0,
        attention_dropout=0.0)
    model = Sam3ViTModel(cfg)
    result = model.load_state_dict(converted, strict=False)
    # RoPE tables are non-persistent buffers transformers rebuilds; ignore them.
    missing = [k for k in result.missing_keys if "rope" not in k]
    if missing:
        raise RuntimeError(
            f"checkpoint is missing backbone weights after conversion: "
            f"{missing[:5]}")
    model.eval()
    return model
