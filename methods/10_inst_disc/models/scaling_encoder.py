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
