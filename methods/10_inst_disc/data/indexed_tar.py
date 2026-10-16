"""Compatibility imports for shared read-only indexed-tar membership."""

from downstream.indexed_tar import (
    IndexedTarDataset,
    checked_row,
    freeze,
    validate_output,
)

__all__ = ["IndexedTarDataset", "checked_row", "freeze", "validate_output"]
