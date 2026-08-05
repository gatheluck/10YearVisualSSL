"""The image_gpt model, built in one place so the trainer, the evaluator and
the adapter all construct the same architecture from the same settings."""

from __future__ import annotations

from .igpt import IGPT


def build_igpt(vocab_size: int, img_size: int, n_layer: int, n_head: int,
               n_embd: int) -> IGPT:
    return IGPT(vocab_size=int(vocab_size), img_size=int(img_size),
                n_layer=int(n_layer), n_head=int(n_head), n_embd=int(n_embd))


__all__ = ["IGPT", "build_igpt"]
