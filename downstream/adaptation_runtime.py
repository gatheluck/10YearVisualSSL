"""Training-facing adapter checks and optimizer grouping."""
import math

import torch

from downstream.lora import assert_adapter_usage


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
