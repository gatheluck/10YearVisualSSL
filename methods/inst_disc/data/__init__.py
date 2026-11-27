"""The instance-discrimination dataset and its transforms, in one place."""

from __future__ import annotations

from .instdisc_dataset import ImageFolderWithIndex, get_instdisc_transforms

__all__ = ["ImageFolderWithIndex", "get_instdisc_transforms"]
