"""Four-slot instance bank, sharing the baseline NCE and coalescing rules."""
import math
import torch
import torch.nn.functional as F

from .nce_loss import NCELoss


class MultiPrototypeBank(NCELoss):
    def __init__(self, num_samples, feature_dim, temperature, momentum, num_negatives):
        torch.nn.Module.__init__(self)
        self.num_samples, self.feature_dim = num_samples, feature_dim
        self.temperature, self.momentum = temperature, momentum
        self.num_negatives = num_negatives
        bound = 1 / math.sqrt(feature_dim / 3)
        memory = torch.rand(num_samples, 4, feature_dim).mul_(2 * bound).add_(-bound)
        self.register_buffer("memory", F.normalize(memory, dim=2))
        self.register_buffer("Z", torch.tensor(-1.))
        self.register_buffer("slot_initialized", torch.zeros(num_samples, 4, dtype=torch.uint8))
        self.register_buffer("assignment_counts", torch.zeros(4, dtype=torch.long))
        self.register_buffer("fill_counts", torch.zeros(4, dtype=torch.long))

    def instance_scores(self, features, ids):
        dots = torch.einsum("bskd,bd->bsk", self.memory[ids].detach(), features)
        initialized = self.slot_initialized[ids].bool()
        masked = dots.masked_fill(~initialized, float("-inf"))
        return torch.where(initialized.any(-1, keepdim=True), masked, dots).max(-1).values

    def forward(self, features, indices):
        negatives = torch.randint(0, self.num_samples,
            (len(indices), self.num_negatives), device=indices.device)
        ids = torch.cat((indices[:, None], negatives), dim=1)
        return self.loss_from_exp(torch.exp(self.instance_scores(features, ids) / self.temperature))

    @torch.no_grad()
    def update_memory(self, z1, z2, indices):
        for z in (z1, z2):
            ids, features = self.unique_mean(z.detach(), indices)
            counts = self.slot_initialized[ids].sum(1)
            fill = counts < 4
            if fill.any():
                rows, slots = ids[fill], counts[fill].long()
                self.memory[rows, slots] = F.normalize(features[fill], dim=1)
                self.slot_initialized[rows, slots] = 1
                self.fill_counts += torch.bincount(slots, minlength=4)
            if (~fill).any():
                rows, features = ids[~fill], features[~fill]
                slots = torch.einsum("bkd,bd->bk", self.memory[rows], features).argmax(1)
                self.memory[rows, slots] = F.normalize(self.momentum * self.memory[rows, slots]
                    + (1 - self.momentum) * features, dim=1)
                self.assignment_counts += torch.bincount(slots, minlength=4)
