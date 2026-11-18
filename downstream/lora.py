"""Explicit attention adapters; task recipes and backbone taps are separate."""
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


def _validate(rank, alpha):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")


class LowRankLinear(nn.Module):
    def __init__(self, base, *, rank=16, alpha=16):
        super().__init__()
        _validate(rank, alpha)
        if not isinstance(base, nn.Linear):
            raise TypeError("adapter target must be nn.Linear")
        self.base = base
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        base.requires_grad_(False)
        self.calls = 0

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        self.calls += 1
        output = self.base(x)
        adapter_input = x if torch.is_autocast_enabled() else x.to(self.lora_A.dtype)
        delta = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
        return output + (delta * self.scaling).to(output.dtype)


def inject_attention_lora(backbone, block_groups, *, rank=16, alpha=16):
    """Preflight explicit block groups, freeze the backbone, then install adapters.

    Only two-projection fused attention and four-projection split attention
    are accepted. No block discovery or task-head selection is implicit.
    """
    _validate(rank, alpha)
    paths = {id(module): path for path, module in backbone.named_modules()}
    selected = []
    seen = set()
    leaves = {"qkv", "proj", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}
    for blocks in block_groups:
        for block in blocks:
            if id(block) not in paths or id(block) in seen:
                raise ValueError("blocks must be distinct members of the backbone")
            seen.add(id(block))
            targets = []
            for path, module in block.named_modules():
                parts = path.split(".")
                if isinstance(module, nn.Linear) and parts[-1] in leaves and any(
                    "attn" in part or "attention" in part for part in parts[:-1]
                ):
                    targets.append((path, module))
            if len(targets) not in (2, 4):
                raise ValueError("each block must have two or four attention projections")
            for path, module in targets:
                full_path = ".".join(filter(None, (paths[id(block)], path)))
                selected.append((full_path, module))
    if not selected:
        raise ValueError("no attention projections selected")
    selected_ids = [id(module) for _, module in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("attention projections must not share modules")
    backbone.requires_grad_(False)
    adapters = {}
    for path, base in selected:
        parent_path, _, leaf = path.rpartition(".")
        parent = backbone.get_submodule(parent_path)
        adapted = LowRankLinear(base, rank=rank, alpha=alpha)
        setattr(parent, leaf, adapted)
        adapters[path] = adapted
    return adapters


def assert_adapter_usage(adapters, *, reset=False):
    """Check after a representative forward; optionally start a new window."""
    if not adapters:
        raise ValueError("no adapters to check")
    unused = [path for path, module in adapters.items() if module.calls == 0]
    if unused:
        raise RuntimeError("attention adapters bypassed: " + ", ".join(unused))
    if reset:
        for module in adapters.values():
            module.calls = 0
