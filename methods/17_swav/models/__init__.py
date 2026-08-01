"""Rewritten during the port: the captured file advertised the step 2 ViT
through a lazy `__getattr__`, and that module was not brought across (the
capture has no official-style step 2). Advertising a name that cannot be
imported is a promise the package cannot keep."""

from .resnet_swav import build_resnet_swav, ResNetSwAV

__all__ = ["build_resnet_swav", "ResNetSwAV"]
