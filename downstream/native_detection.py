"""Explicit source-family detector compositions; legacy geometry remains separate."""
from collections import OrderedDict

import torch
from torch import nn
from downstream.captured_readers import spatial_adapter

PROFILE = 'captured_native_detection_v1'


class TransposedFeaturePyramid(nn.Module):
    """Four learned scales of the actual native grid, without stride relabeling."""
    def __init__(self, channels, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.p2 = nn.Sequential(nn.ConvTranspose2d(channels,out_channels,2,2),nn.GELU(),
                                nn.ConvTranspose2d(out_channels,out_channels,2,2),nn.GELU(),
                                nn.Conv2d(out_channels,out_channels,1))
        self.p3 = nn.Sequential(nn.ConvTranspose2d(channels,out_channels,2,2),nn.GELU(),
                                nn.Conv2d(out_channels,out_channels,1))
        self.p4 = nn.Conv2d(channels,out_channels,1)
        self.p5 = nn.Sequential(nn.MaxPool2d(2,2),nn.Conv2d(channels,out_channels,1))
        for module in self.modules():
            if isinstance(module,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features):
        return OrderedDict((str(i),layer(features)) for i,layer in enumerate((self.p2,self.p3,self.p4,self.p5)))


class NativePyramidBackbone(nn.Module):
    def __init__(self, body, *, adaptation, reader_profile=None):
        super().__init__()
        self.body, self.adaptation = body, adaptation
        self.adapter = spatial_adapter(body,reader_profile) if adaptation == 'attentive' else None
        if body.native_pyramid_style == 'bilinear':
            from downstream.coco import SimpleFeaturePyramid
            self.fpn = SimpleFeaturePyramid(body.out_channels)
        elif body.native_pyramid_style == 'transposed':
            self.fpn = TransposedFeaturePyramid(body.out_channels)
        else:
            raise ValueError('unverified native detection pyramid')
        self.out_channels = self.fpn.out_channels

    def forward(self, images):
        from contextlib import nullcontext
        with nullcontext() if self.adaptation == 'finetune' else torch.no_grad():
            feat = self.body.forward_detection_features(images)
        if feat.ndim != 4 or feat.shape[1] != self.body.out_channels or min(feat.shape[-2:]) < 2:
            raise ValueError('native pyramid requires a real grid at least 2 by 2')
        if self.adapter is not None:
            feat = self.adapter(feat)
        return self.fpn(feat)


def validate_profile(cfg):
    if 'detector_profile' not in cfg:
        from downstream.spatial_backbones import attentive_profile
        if cfg.get('adaptation') == 'attentive' and attentive_profile(cfg.get('backbone',{}).get('kind')):
            raise ValueError('family attentive detection requires detector_profile')
        return
    from downstream.spatial_backbones import supports_native_detection
    if (cfg['detector_profile'] != PROFILE or cfg.get('profile') != 'capture_basic5_components'
            or not supports_native_detection(cfg.get('backbone',{}).get('kind'))):
        raise ValueError('detector_profile requires an explicitly verified native provider and component profile')
