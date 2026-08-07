"""Alias method for efficient O(1) multinomial sampling of NCE negatives.

Reference: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
"""

from __future__ import annotations

import torch


class AliasMethod:
    """Efficient multinomial sampler using the alias method."""

    def __init__(self, probs: torch.Tensor):
        if probs.sum() > 1:
            probs = probs / probs.sum()

        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        smaller, larger = [], []
        for k, p in enumerate(probs):
            self.prob[k] = K * p.item()
            if self.prob[k] < 1.0:
                smaller.append(k)
            else:
                larger.append(k)

        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last in smaller + larger:
            self.prob[last] = 1.0

    def to(self, device) -> "AliasMethod":
        """Move the sampler's tables to a device (the port resolves the device
        rather than assuming CUDA)."""
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)
        return self

    def draw(self, N: int) -> torch.Tensor:
        """Draw N samples from the distribution."""
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long,
                         device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        return kk * b.long() + alias * (1 - b).long()
