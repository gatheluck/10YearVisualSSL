"""Local official C-RADIOv4-H, preserving distinct summary/spatial readouts."""
from contextlib import nullcontext
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

KIND = 'cradiov4_h'
TRAINABLE = True
FINETUNE_GROUPS = True
COMPONENT_ONLY = True
IMAGE_CLASSIFICATION = True
IMAGENET_FT_RECIPE = "captured_bicubic_random_python_v1"
IMAGENET_PROBE_INTERPOLATION = "bicubic"
NATIVE_VIDEO = True
NATIVE_DETECTION = True
CAPTURE_PYRAMID = False
SUPPORTED_ADAPTATIONS = ('frozen', 'attentive', 'finetune')
ATTENTIVE_PROFILE = 'captured_cross_self_v1'


def _load_local_model(path):
    from transformers import AutoModel
    model, info = AutoModel.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float32, output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('incomplete or incompatible official checkpoint')
    return model


def _build(spec, trainable):
    if spec.get('arch') not in ('released', 'fixture'):
        raise ValueError('arch must explicitly select released or fixture')
    if spec.get('patch_size') != 16 or spec.get('img_size') != 224:
        raise ValueError('BasicFive requires patch_size=16 and img_size=224')
    root = Path(spec.get('encoder', ''))
    if not spec.get('encoder') or not (root/'config.json').is_file() or not (root/'model.safetensors').is_file():
        raise ValueError('encoder requires a complete local official snapshot')
    cfg = json.loads((root/'config.json').read_text())
    if (cfg.get('version') != 'c-radio_v4-h' or cfg.get('architectures') != ['RADIOModel']
            or cfg.get('patch_size') != 16 or cfg.get('adaptor_names')
            or cfg.get('args', {}).get('model') != 'vit_huge_patch16_224'
            or cfg.get('args', {}).get('pretrained')
            or cfg.get('args', {}).get('initial_checkpoint')):
        raise ValueError('snapshot is not the selected official vision-only model')
    if spec['arch'] == 'released':
        if 'embed_dim' in spec or 'depth' in spec:
            raise ValueError('released dimensions cannot be overridden')
        dim, depth = 1280, 32
    else:
        dim, depth = spec.get('embed_dim'), spec.get('depth')
        if any(type(x) is not int or x < 1 for x in (dim, depth)):
            raise ValueError('fixture requires explicit positive dimensions')
    return Backbone(_load_local_model(root), dim=dim, depth=depth, trainable=trainable)


class Backbone(nn.Module):
    patch_size = 16
    reader_profile = ATTENTIVE_PROFILE
    native_pyramid_style = 'transposed'
    classifier_init_std = .01

    def __init__(self, model, *, dim, depth, trainable=False):
        super().__init__()
        self.model, self.trainable = model, trainable
        self.out_channels, self.global_channels, self.depth = dim, 2*dim, depth
        radio = model.radio_model
        idxs = getattr(radio, 'summary_idxs', None)
        if (getattr(model, 'patch_size', None) != 16 or radio.num_summary_tokens != 10
                or idxs is None or idxs.tolist() != [0, 1]
                or getattr(radio, 'summary_dim', None) != 2*dim
                or getattr(radio, 'input_conditioner', None) is None
                or getattr(radio, 'adaptors', None)
                or len(radio.model.blocks) != depth):
            raise ValueError('official summary, conditioner or visual layout differs')
        self.requires_grad_(False)
        radio.model.requires_grad_(trainable)
        self.train(trainable)

    def train(self, mode=True):
        super().train(mode if self.trainable else False)
        return self

    def _outputs(self, images, *, raw=False):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16:
            raise ValueError('images require [B,3,H,W] with at least one complete patch')
        pixels = images.float()
        if not raw:
            pixels = pixels*pixels.new_tensor((.229,.224,.225))[None,:,None,None] + pixels.new_tensor((.485,.456,.406))[None,:,None,None]
        height, width = pixels.shape[-2:]
        pixels = F.pad(pixels, (0,-width%16,0,-height%16))
        with nullcontext() if self.trainable else torch.no_grad():
            out = self.model(pixels)
        if hasattr(out, 'summary') and hasattr(out, 'features'):
            summary, features = out.summary, out.features
        elif isinstance(out, (tuple, list)) and len(out) == 2:
            summary, features = out
        else:
            raise ValueError('official summary/features output required')
        summary = summary.flatten(1)
        batch, gh, gw = images.shape[0], pixels.shape[-2]//16, pixels.shape[-1]//16
        if summary.shape != (batch, self.global_channels):
            raise ValueError('official summary shape differs')
        if features.ndim == 3 and features.shape == (batch, gh*gw, self.out_channels):
            features = features.transpose(1,2).reshape(batch,self.out_channels,gh,gw)
        if features.shape != (batch,self.out_channels,gh,gw):
            raise ValueError('official spatial grid differs; refusing token substitution')
        return summary.float(), features[:,:,:height//16,:width//16].float()

    def forward_features(self, images):
        return self._outputs(images)[1]

    def _video(self, clip):
        if clip.ndim != 5 or clip.shape[1] < 1 or clip.shape[2] != 3:
            raise ValueError('video requires [B,frames,3,H,W]')
        radio, frames = self.model.radio_model, clip.shape[1]
        ctx = radio.cpe_video_mode(t=frames) if hasattr(radio,'cpe_video_mode') else nullcontext()
        with ctx:
            summary, features = self._outputs(clip.flatten(0,1))
        return (summary.reshape(clip.shape[0],frames,self.global_channels),
                features.mean((-2,-1)).reshape(clip.shape[0],frames,self.out_channels))

    def video_tokens(self, clip):
        return self._video(clip)[1]

    def classification_features(self, images, *, adaptation, video=False):
        if adaptation not in ('frozen','finetune'):
            raise ValueError('classification requires frozen or finetune readout')
        if video:
            return self._video(images)[0].mean(1)
        summary = self._outputs(images)[0]
        return F.normalize(summary,dim=-1) if adaptation == 'frozen' else summary

    def detection_normalization(self):
        return (0.,0.,0.), (1.,1.,1.)

    def forward_detection_features(self, images):
        return self._outputs(images,raw=True)[1]

    def forward(self, images):
        return self.forward_features(images)

    def finetune_group_policy(self):
        entries, blocks = {}, set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            block = re.search(r'\.blocks\.(\d+)\.',name)
            layer = 0
            if block:
                index=int(block[1]); blocks.add(index); layer=index+1
            low=name.lower()
            exempt = ('bias' in low or any(k in low for k in ('norm','ln','position_embedding','pos_embed','positional','class_token','cls_token'))
                      or 'token' in low.split('.')[-1])
            entries[name]=(layer,exempt)
        if blocks != set(range(self.depth)):
            raise ValueError('finetune requires the complete visual tower')
        return self.depth, entries


def build(spec):
    return _build(spec, False)


def build_trainable(spec):
    return _build(spec, True)
