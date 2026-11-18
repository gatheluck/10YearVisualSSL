"""Training-facing adapter checks and optimizer grouping."""
import math

import torch
from torch import nn

from downstream.lora import assert_adapter_usage, inject_attention_lora


def install_backbone_lora(backbone, *, rank=16, alpha=16):
    """Install adapters using a provider's explicit, validated block binding.

    Load/place the base model first and build the optimizer afterwards. This
    selects neither a task head nor an experiment recipe. Check adapter effects
    and gradients using representative task inputs before training.
    """
    if not all(callable(getattr(backbone, name, None)) for name in
               ('adaptation_block_groups', 'enable_adaptation_grad')):
        raise ValueError('backbone has no explicit adaptation binding')
    groups = backbone.adaptation_block_groups()
    adapters = inject_attention_lora(backbone, groups, rank=rank, alpha=alpha)
    backbone.enable_adaptation_grad()
    return adapters


class FrozenDPTModel(nn.Module):
    """Frozen model-specific taps plus a trainable dense probe.

    Inputs follow the provider's ImageNet-normalized image convention. Dataset
    loss, metrics, optimizer, scheduling and distributed execution remain owned
    by the caller; this composition does not select a canonical task recipe.
    """
    def __init__(self, backbone, in_dims, *, out_channels, hidden=512, positive=False):
        super().__init__()
        from downstream.dpt_probe import DPTProbe
        if not callable(getattr(backbone, 'forward_dpt_features', None)):
            raise TypeError('backbone has no explicit four-tap binding')
        self.head = DPTProbe(in_dims, out_channels=out_channels, hidden=hidden, positive=positive)
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images, out_hw):
        with torch.no_grad():
            features = [f.detach() for f in self.backbone.forward_dpt_features(images)]
        return self.head(features, out_hw)


def adapter_parameter_groups(named, *, weight_decay, head_lr, adapter_lr):
    for value, minimum in ((weight_decay, 0), (head_lr, 1), (adapter_lr, 1)):
        if not math.isfinite(value) or value < 0 or (minimum and value == 0):
            raise ValueError('invalid adapter optimizer rate or weight decay')
    buckets = {}
    seen = set()
    for name, parameter in named:
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError('duplicate trainable parameter')
        seen.add(id(parameter))
        is_adapter = name.rsplit('.', 1)[-1] in ('lora_A', 'lora_B')
        if not is_adapter and not name.startswith('head.'):
            raise ValueError('unexpected trainable parameter: ' + name)
        no_decay = parameter.ndim < 2 or 'norm' in name.lower()
        buckets.setdefault((is_adapter, no_decay), []).append(parameter)
    if not buckets:
        raise ValueError('no trainable head or adapter parameters')
    groups = []
    for (is_adapter, no_decay), parameters in sorted(buckets.items()):
        rate = adapter_lr if is_adapter else head_lr
        groups.append({'params': parameters, 'lr': rate, 'base_lr': rate,
                       'weight_decay': 0. if no_decay else weight_decay,
                       'kind': 'lora' if is_adapter else 'head'})
    return groups


@torch.no_grad()
def check_adapter_effect(forward, adapters):
    """Compare baseline/perturbed/restored features without retaining mutations.

    The caller supplies the real feature forward, including autocast and inputs.
    This diagnostic does not verify backward gradients or a training recipe.
    """
    if not adapters:
        raise ValueError('no adapters to check')
    layers = list(adapters.values())
    saved = [layer.lora_B.detach().clone() for layer in layers]
    calls = [layer.calls for layer in layers]
    devices = sorted({layer.lora_B.device.index for layer in layers if layer.lora_B.is_cuda})
    with torch.random.fork_rng(devices=devices):
        try:
            for layer in layers:
                layer.calls = 0
            baseline = forward().float()
            assert_adapter_usage(adapters)
            generators = {layer.lora_B.device: torch.Generator(device=layer.lora_B.device).manual_seed(1234)
                          for layer in layers}
            for layer in layers:
                layer.lora_B.normal_(0., .02, generator=generators[layer.lora_B.device])
            perturbed = forward().float()
            for layer, value in zip(layers, saved):
                layer.lora_B.copy_(value)
            restored = forward().float()
            scale = baseline.abs().mean().item()
            delta = (perturbed - baseline).abs().mean().item()
            repeated = (restored - baseline).abs().mean().item()
            if not all(math.isfinite(v) for v in (scale, delta, repeated)) or not delta > 10 * max(repeated, 1e-5 * scale):
                raise RuntimeError('adapter perturbation has no reliable effect on features')
            return {'feature_mean_abs': scale, 'delta_perturbed': delta,
                    'delta_restored': repeated, 'n_wrapped': len(layers)}
        finally:
            for layer, value, count in zip(layers, saved, calls):
                layer.lora_B.copy_(value)
                layer.calls = count
