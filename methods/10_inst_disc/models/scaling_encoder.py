"""Shared native-preset construction and versioned scaling export identity."""
import torch

PRESETS = ("vit_base_patch16_224", "vit_large_patch16_224")


def build_encoder(arch):
    import timm
    from timm.models.vision_transformer import VisionTransformer
    if arch not in PRESETS:
        raise ValueError("scaling requires the ViT-B/16 or ViT-L/16 preset")
    encoder = timm.create_model(arch, pretrained=False, num_classes=0,
                                global_pool="", dynamic_img_size=True)
    if not isinstance(encoder, VisionTransformer):
        raise TypeError("scaling preset must construct a VisionTransformer")
    return encoder


def read_export(path):
    export = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(export, dict)
            or type(export.get("scaling_encoder_version")) is not int
            or export["scaling_encoder_version"] != 1
            or export.get("readout") != "cls"
            or export.get("model") not in PRESETS):
        raise ValueError("incompatible scaling encoder identity")
    return export


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_native(*, source, sha256, arch, completed_epoch, state_key, prefix, out):
    """Convert explicit native encoder weights; never restore training state."""
    import json
    from pathlib import Path

    from downstream.native_checkpoint import load_native

    out = Path(out)
    if out.exists():
        raise FileExistsError("output already exists; choose a new directory")
    if arch not in PRESETS or type(completed_epoch) is not int or completed_epoch < 1:
        raise ValueError("explicit supported architecture and positive completed epoch required")
    if state_key not in ("state_dict", "model") or prefix not in (
        "vit.", "module.vit.", "encoder.vit.", "module.encoder.vit."
    ):
        raise ValueError("explicit native state key and encoder prefix required")
    if _sha256(source) != sha256:
        raise ValueError("source sha256 mismatch")
    blob = load_native(source)
    if (not isinstance(blob, dict)
            or type(blob.get("completed_epoch")) is not int
            or blob["completed_epoch"] != completed_epoch
            or type(blob.get("step_in_epoch")) is not int
            or blob["step_in_epoch"] != 0):
        raise ValueError("native checkpoint must match the explicit completed epoch boundary")
    raw = blob.get(state_key)
    if not isinstance(raw, dict) or any(not isinstance(k, str) for k in raw):
        raise ValueError("native state must be an explicit string-keyed mapping")
    state = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    encoder = build_encoder(arch)
    expected = encoder.state_dict()
    if set(state) != set(expected):
        raise ValueError("native encoder keys do not exactly match the preset")
    for key, value in state.items():
        target = expected[key]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or not torch.isfinite(value).all()):
            raise ValueError("native encoder tensor shape, dtype or finiteness mismatch")
    encoder.load_state_dict(state, strict=True)
    if _sha256(source) != sha256:
        raise ValueError("source changed during import")
    out.mkdir(parents=True, exist_ok=False)
    dest = out / "encoder.pt"
    torch.save({"scaling_encoder_version": 1, "model": arch, "readout": "cls",
                "state_dict": encoder.state_dict()}, dest)
    record = {"source_sha256": sha256, "encoder_sha256": _sha256(dest),
              "model": arch, "completed_epoch": completed_epoch,
              "state_key": state_key, "prefix": prefix, "tensor_count": len(state),
              "validation": "strict native encoder import; historical score attribution unverified"}
    (out / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Import native IDv2 scaling encoder weights")
    for name in ("source", "sha256", "arch", "state-key", "prefix", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--completed-epoch", required=True, type=int)
    import_native(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
