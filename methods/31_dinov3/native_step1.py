"""Offline official DINOv3 ViT-B/16 inference through a pinned submodule."""
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms as T


def make_transform():
    return T.Compose([T.Resize(585, interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(512), T.ToTensor(),
                      T.Normalize([.485, .456, .406], [.229, .224, .225])])


def official_factory():
    # Import only the backbone: hubconf also imports unrelated task heads.
    import importlib
    import sys
    source = Path(__file__).resolve().parents[2] / 'third_party' / 'dinov3'
    expected = source / 'dinov3' / 'hub' / 'backbones.py'
    if not expected.is_file():
        raise RuntimeError('initialize the pinned third_party/dinov3 submodule first')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module('dinov3.hub.backbones')
    if Path(module.__file__).resolve() != expected.resolve():
        raise RuntimeError('a foreign dinov3 package is already imported')
    return module.dinov3_vitb16


def build_backbone():
    return official_factory()(pretrained=False)


class OfficialFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images, is_global=True):
        # The existing evaluator reads element zero, the normalized CLS token.
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=images.is_cuda):
            return self.backbone.forward_features(images)['x_norm_clstoken'], None


def load_encoder(state):
    backbone = build_backbone()
    backbone.load_state_dict(state, strict=True)
    return OfficialFeatures(backbone)
