"""Captured dense FT photometric order, shared by segmentation and depth."""
import torch
from torchvision.transforms import functional as TF


def captured_color_jitter(image, strength):
    """Jitter brightness, contrast, saturation in fixed order; preserve geometry."""
    if strength > 0:
        for adjust in (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation):
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
            image = adjust(image, factor)
    return image
