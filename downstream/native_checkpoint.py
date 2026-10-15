"""Restricted loading of native archives carrying NumPy MT19937 state."""

import torch


def _numpy_reconstruct_alias(namespace):
    # Older Torch accepts callables only, not (callable, qualified-name) pairs.
    # Give the same narrow NumPy constructor both historical pickle names.
    def reconstruct(*args):
        from numpy.core.multiarray import _reconstruct

        return _reconstruct(*args)

    reconstruct.__module__ = namespace
    reconstruct.__name__ = reconstruct.__qualname__ = "_reconstruct"
    return reconstruct


def load_native(path):
    import numpy as np

    with torch.serialization.safe_globals(
        [
            np.ndarray,
            np.dtype,
            type(np.dtype("uint32")),
            _numpy_reconstruct_alias("numpy.core.multiarray"),
            _numpy_reconstruct_alias("numpy._core.multiarray"),
            torch.torch_version.TorchVersion,
        ]
    ):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
