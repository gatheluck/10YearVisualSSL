"""iGPT: a causal transformer over colour-cluster tokens (Chen et al., 2020).

Ported from the lab's ARSSL inline model (`src/models/train_igpt_scratch.py`),
which is itself a self-contained re-implementation of OpenAI's Image GPT (the
original is TensorFlow 1.x). Kept faithful in shape -- token + position
embeddings, a stack of pre-norm causal-attention blocks, a final norm and a
generative head over the colour vocabulary -- and made parameterisable so a
hermetic smoke can build a tiny one.

`extract_features` is the representation the linear probe reads: a **middle**
transformer layer, mean-pooled over the sequence. That is the trained model's
own representation, so `encoder.pt` (everything but the generative `head`) is
what the probe evaluates -- unlike the generative ports, whose probe reads a
separate tokeniser.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, seq_len: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd={n_embd} is not divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class IGPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, seq_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, seq_len)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class IGPT(nn.Module):
    """A GPT over `vocab_size` colour tokens for a `img_size`x`img_size` grid."""

    def __init__(self, vocab_size: int, img_size: int, n_layer: int,
                 n_head: int, n_embd: int) -> None:
        super().__init__()
        seq_len = img_size * img_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size + 1, n_embd)   # +1 for SOS
        self.pos_embed = nn.Embedding(seq_len + 1, n_embd)
        self.blocks = nn.ModuleList(
            [IGPTBlock(n_embd, n_head, seq_len + 1) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] colour-token IDs -> logits [B, T, vocab_size]."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

    def extract_token_features(self, x: torch.Tensor) -> torch.Tensor:
        """The per-position representation: a **middle** transformer layer, one
        vector per token. Returns [B, T, n_embd].

        The linear probe mean-pools this over the sequence; a dense downstream
        task instead reshapes the tokens back to their grid. Both read the same
        layer, so there is one implementation of "the iGPT representation" here."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)
        mid = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == mid:
                break
        return h

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """The linear-probe representation: `extract_token_features`, mean-pooled
        over the sequence. Returns [B, n_embd]."""
        return self.extract_token_features(x).mean(dim=1)
