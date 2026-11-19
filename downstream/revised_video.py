"""Revised source SSv2 clips and differentiable provider-specific readouts."""
import importlib
import json
import re
from pathlib import Path

from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from downstream.video_recipe import sample_indices, transform

_RECIPE = {'temporal_rng': 'python', 'augmentation': 'plain', 'interpolation': 'bicubic'}


def _clean(label):
    return re.sub(r'\[([^\]]*)\]', r'\1', label)


class RevisedSSV2(Dataset):
    """Track absent videos separately; never substitute another clip or zeros."""
    def __init__(self, root, split):
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        root = Path(root)
        self.train = split == 'train'
        self.label_files = (root / 'labels/labels.json', root / 'labels' / (
            'train.json' if self.train else 'validation.json'))
        self.class_to_idx = {_clean(k): int(v) for k, v in json.loads(
            (root / 'labels/labels.json').read_text()).items()}
        entries = json.loads((root / 'labels' / (
            'train.json' if self.train else 'validation.json')).read_text())
        self.n_entries, self.items, self.missing = len(entries), [], []
        for entry in entries:
            label = self.class_to_idx[_clean(entry.get('template', entry.get('label', '')))]
            identifier = str(entry['id'])
            path = root / 'videos' / f'{identifier}.webm'
            if not path.is_file():
                path = path.with_suffix('.mp4')
            if path.is_file():
                self.items.append((path, label))
            else:
                self.missing.append(identifier)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        VideoReader = importlib.import_module("decord").VideoReader
        path, label = self.items[index]
        reader = VideoReader(str(path), num_threads=1)
        indices = sample_indices(len(reader), 16, self.train, _RECIPE)
        frames = [Image.fromarray(frame).convert('RGB')
                  for frame in reader.get_batch(indices).asnumpy()]
        return transform(frames, self.train, 224, _RECIPE), label


class VideoTask(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(getattr(backbone, 'global_channels', backbone.out_channels), 174)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, clips):
        if clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError('video requires B,T,C,H,W RGB clips')
        readout = getattr(self.backbone, 'classification_features', None)
        if callable(readout):
            return readout(clips, adaptation='frozen', video=True)
        tokens = self.backbone.video_tokens(clips)
        return F.normalize(tokens.mean(1).float(), dim=-1)

    def forward(self, clips):
        return self.head(self.features(clips).float())
