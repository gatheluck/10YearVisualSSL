"""Data loading for VAE training."""

from .vae_dataset import get_vae_dataloader  # noqa: F401

__all__ = ["get_vae_dataloader"]
