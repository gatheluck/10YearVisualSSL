"""NCE memory bank for Contrastive Multiview Coding (Tian et al., 2019).

Maintains separate memory banks for each view (L and ab) as ``register_buffer``
and computes NCE scores by inner-product lookup, cross-view: the ab features are
scored against the L bank and the L features against the ab bank. Negatives are
drawn from the banks by the alias method.

The banks live here, in the loss module, not in the model -- so the model's
``state_dict`` (and thus ``encoder.pt``) never carries them, the same shape as
the inst_disc port.

The lab wrapper trains under DistributedDataParallel; the all-gather / broadcast
/ all-reduce paths are kept but are all guarded by ``dist.is_initialized()``, so
they are inert in this single-process port (negatives and bank writes come from
within the batch).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from .alias_multinomial import AliasMethod


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather ``tensor`` from every rank; inert (returns it unchanged) when
    not running under DDP."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensors = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor, async_op=False)
    return torch.cat(tensors, dim=0)


class NCEAverage(nn.Module):
    """Memory bank for CMC with L and ab views.

    Args:
        feat_dim: feature dimension.
        n_data:   total number of training images.
        K:        number of noise/negative samples.
        T:        temperature.
        momentum: EMA momentum for the memory update.
    """

    def __init__(self, feat_dim: int, n_data: int, K: int = 16384,
                 T: float = 0.07, momentum: float = 0.5):
        super().__init__()
        self.nLem = n_data
        self.K = K

        self.multinomial = AliasMethod(torch.ones(n_data))

        # params: [K, T, Z_l, Z_ab, momentum]
        self.register_buffer("params",
                             torch.tensor([K, T, -1.0, -1.0, momentum]))

        stdv = 1.0 / math.sqrt(feat_dim / 3)
        self.register_buffer(
            "memory_l", torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer(
            "memory_ab", torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, feat_l: torch.Tensor, feat_ab: torch.Tensor,
                y: torch.Tensor, idx: Optional[torch.Tensor] = None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()
        momentum = self.params[4].item()

        bsz = feat_l.size(0)
        feat_dim = self.memory_l.size(1)

        # --- sample noise indices ----------------------------------------
        if idx is None:
            idx = self.multinomial.draw(bsz * (K + 1)).view(bsz, -1)
            idx[:, 0].copy_(y.data)  # first slot = the positive

        # --- score computation (cross-view) ------------------------------
        weight_l = self.memory_l.index_select(0, idx.view(-1)).detach()
        weight_l = weight_l.view(bsz, K + 1, feat_dim)
        out_ab = torch.bmm(weight_l, feat_ab.view(bsz, feat_dim, 1)).squeeze(2)

        weight_ab = self.memory_ab.index_select(0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(bsz, K + 1, feat_dim)
        out_l = torch.bmm(weight_ab, feat_l.view(bsz, feat_dim, 1)).squeeze(2)

        out_l = torch.exp(out_l / T)
        out_ab = torch.exp(out_ab / T)

        # Initialise the partition constants once from the first batch.
        if Z_l < 0:
            self.params[2] = out_l.mean() * self.nLem
            Z_l = self.params[2].item()
        if Z_ab < 0:
            self.params[3] = out_ab.mean() * self.nLem
            Z_ab = self.params[3].item()

        out_l = (out_l / Z_l).contiguous()
        out_ab = (out_ab / Z_ab).contiguous()

        # --- update the memory banks (EMA) -------------------------------
        with torch.no_grad():
            all_feat_l = concat_all_gather(feat_l)
            all_feat_ab = concat_all_gather(feat_ab)
            all_y = concat_all_gather(y)

            l_pos = self.memory_l.index_select(0, all_y)
            l_pos.mul_(momentum).add_(all_feat_l * (1.0 - momentum))
            l_pos.div_(l_pos.norm(dim=1, keepdim=True).clamp(min=1e-8))
            self.memory_l.index_copy_(0, all_y, l_pos)

            ab_pos = self.memory_ab.index_select(0, all_y)
            ab_pos.mul_(momentum).add_(all_feat_ab * (1.0 - momentum))
            ab_pos.div_(ab_pos.norm(dim=1, keepdim=True).clamp(min=1e-8))
            self.memory_ab.index_copy_(0, all_y, ab_pos)

        return out_l, out_ab
