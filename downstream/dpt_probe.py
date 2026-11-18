"""Dense probe adapter around the pinned upstream implementation.

The upstream DPT remains in its submodule, with its original license. This
adapter adds per-tap normalization and replaces its final resize/readout.
"""
import importlib.util
from functools import lru_cache
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path

from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def _upstream():
    source = Path(__file__).resolve().parents[1] / 'third_party/geosubprobe/evals/models/probes.py'
    if not source.is_file():
        raise FileNotFoundError('initialize the pinned geosubprobe submodule')
    loader = SourceFileLoader('_pinned_geosubprobe_probes', str(source))
    spec = ModuleSpec(loader.name, loader, origin=str(source))
    spec.has_location = True
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class DPTProbe(nn.Module):
    def __init__(self, in_dims, *, out_channels, hidden=512, positive=False):
        super().__init__()
        self.in_dims = tuple(in_dims)
        if len(self.in_dims) != 4 or any(type(v) is not int or v <= 0 for v in (*self.in_dims, out_channels, hidden)):
            raise ValueError('DPT requires four positive channel dimensions and a positive width')
        self.positive = positive
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in self.in_dims)
        self.probe = _upstream().DPT(list(self.in_dims), out_channels, hidden, 3)

    def forward(self, features, out_hw):
        if len(out_hw) != 2 or any(type(v) is not int or v <= 0 for v in out_hw):
            raise ValueError('output height and width must be positive integers')
        if len(features) != 4 or any(f.ndim != 4 or f.shape[1] != dim for f, dim in zip(features, self.in_dims)):
            raise ValueError('four BCHW features must match the declared channels')
        if any((f.shape[0], *f.shape[-2:]) != (features[0].shape[0], *features[0].shape[-2:]) for f in features):
            raise ValueError('DPT taps must share batch and spatial dimensions')
        normalized = [norm(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                      for norm, f in zip(self.norms, features)]
        captured = []
        handle = self.probe.out_conv.register_forward_hook(lambda module, args, value: captured.append(value))
        try:
            # Capture before upstream's fixed x2 resize, so the experimental
            # target resize does not interpolate an already resized tensor.
            self.probe(normalized)
        finally:
            handle.remove()
        output = F.interpolate(captured[0], size=out_hw, mode='bilinear', align_corners=False)
        return F.softplus(output) + .001 if self.positive else output
