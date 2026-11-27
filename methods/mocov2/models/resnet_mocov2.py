"""ResNet-50 encoder for MoCo v2 (Chen et al., 2020; arXiv:2003.04297), ported
from the lab's own paper-faithful implementation.

MoCo v2 = MoCo v1 + three changes: a **2-layer MLP projection head** (v1 used a
single linear), **Gaussian-blur** augmentation (see the data module), and a
**cosine** LR schedule (see the trainer). The core is unchanged from v1: two
augmented views feed a **query encoder** (ResNet-50 + the MLP head, L2-normalised)
and a **momentum key encoder** (an EMA copy of the query, no gradient); an InfoNCE
loss contrasts the query against the matching key (the positive) and a FIFO
**queue** of K past keys (the negatives). Temperature tau=0.2 (v1 used 0.07).

`encoder.pt` is the query ResNet-50 backbone (`encoder_q.backbone.*`); the MLP
projection head, the key encoder and the queue are training machinery and are
excluded. `get_encoder()` returns the backbone (2048-d) for the linear probe --
the standard SSL convention of probing the backbone, not the projection head.

The lab wrapper carries DistributedDataParallel shuffle-BN / all-gather branches;
they are kept but guarded by ``dist.is_initialized()``, so they are inert in this
single-process port (the queue is filled from within the batch).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as tvm


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ResNetEncoderV2(nn.Module):
    """ResNet-50 trunk with a 2-layer MLP projection head (MoCo v2)."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        base = tvm.resnet50(weights=None)
        # Everything up to and including avgpool; the classifier fc is dropped.
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,2048,1,1)
        # 2-layer MLP head (the MoCo v2 improvement over v1's single Linear).
        self.proj = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=1)


class MoCoV2ResNet(nn.Module):
    """MoCo v2 with a ResNet-50 encoder (Chen et al., 2020)."""

    def __init__(self, feature_dim: int = 128, queue_size: int = 65536,
                 momentum: float = 0.999, temperature: float = 0.2):
        super().__init__()
        self.K = queue_size
        self.m = momentum
        self.T = temperature

        self.encoder_q = ResNetEncoderV2(feature_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(),
                          self.encoder_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        if dist.is_available() and dist.is_initialized():
            keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, (
            f"Queue size ({self.K}) must be divisible by batch size "
            f"({batch_size}).")
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: torch.Tensor):
        """Batch shuffle so key-encoder BatchNorm cannot leak across the batch
        (official MoCo). Inert (returns x unchanged) when not under DDP."""
        if not (dist.is_available() and dist.is_initialized()):
            return x, None
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        num_gpus = x_gather.shape[0] // batch_size_this
        idx_shuffle = torch.randperm(x_gather.shape[0], device=x.device)
        dist.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        idx_this = idx_shuffle.view(num_gpus, -1)[dist.get_rank()]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: torch.Tensor, idx_unshuffle):
        if idx_unshuffle is None:
            return x
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        num_gpus = x_gather.shape[0] // batch_size_this
        idx_this = idx_unshuffle.view(num_gpus, -1)[dist.get_rank()]
        return x_gather[idx_this]

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        """im_q, im_k: two augmented views. Returns (loss, logits, labels)."""
        q = self.encoder_q(im_q)

        with torch.no_grad():
            self._momentum_update()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=logits.device)
        loss = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(k)
        return loss, logits, labels

    def get_encoder(self) -> nn.Module:
        """The query ResNet-50 backbone (2048-d), for downstream probing."""
        return nn.Sequential(self.encoder_q.backbone, _Flatten())


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from every rank and concatenate (DDP only)."""
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    return torch.cat(tensors, dim=0)


def build_mocov2_resnet(feature_dim: int = 128, queue_size: int = 65536,
                        momentum: float = 0.999,
                        temperature: float = 0.2) -> MoCoV2ResNet:
    return MoCoV2ResNet(feature_dim=feature_dim, queue_size=queue_size,
                        momentum=momentum, temperature=temperature)
