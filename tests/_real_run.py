#!/usr/bin/env python3
"""Shared helpers for the real-run tests, so one rule is written once.

Two things both `tests/test_real_run_smoke.py` (one method's stages) and
`tests/test_matrix_run.py` (the grid over all of them) need, and neither should
own a private copy of:

- `driver()` loads `bin/matrix-run.py` as a module, so both tests reach the one
  implementation of "which methods declare a real-run spec" (`discover_specs`)
  and "run one stage through launch.py" (`run_stage`). A second copy would be
  the exact drift this repository keeps being bitten by.
- `build_data()` writes a tiny, real-shaped dataset. It is test-only: the
  driver never fabricates data (on a cluster it points at real data roots), so
  this lives here, not in the tool.

Names no method: only a `data_shape` string and generic class folders.
"""

from __future__ import annotations

import gzip
import importlib.util
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
_DRIVER_NAME = "matrix_run_driver"


def driver():
    """The loaded `bin/matrix-run.py` module (its filename is not importable
    as a name, so it is loaded by path). Loading fails loudly if the tool is
    absent -- a missing driver is a red test, never a silent skip."""
    if _DRIVER_NAME in sys.modules:
        return sys.modules[_DRIVER_NAME]
    spec = importlib.util.spec_from_file_location(_DRIVER_NAME,
                                                  BIN / "matrix-run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_DRIVER_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[_DRIVER_NAME]
        raise
    return mod


def build_data(shape: str, root: Path) -> Path:
    """A real dataset of the shape a spec asks for. An unknown shape fails
    loudly rather than skipping -- an unsupported shape must be visible."""
    root = Path(root)
    if shape == "imagefolder_2class":
        import numpy as np
        from PIL import Image
        rng = np.random.RandomState(0)
        for split in ("train", "val"):
            for label, cls in enumerate(("c0", "c1")):
                d = root / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for i in range(3):
                    base = np.full((128, 128, 3), label * 120, dtype="uint8")
                    noise = rng.randint(0, 64, (128, 128, 3), dtype="uint8")
                    Image.fromarray((base + noise).astype("uint8")).save(
                        d / f"{i}.png")
        return root
    if shape == "mnist":
        # A valid MNIST/ directory (the layout torchvision.datasets.MNIST reads
        # with download=False), fabricated rather than downloaded so no test
        # reaches the network. The IDX format is a short header then raw bytes,
        # so a real one small enough to train on is cheap to write. This is the
        # shape 02_vae asks for -- its dataset-agnostic loader picks MNIST over
        # ImageFolder when a MNIST/ subdirectory is present. Mirrors the proven
        # tiny_mnist fixture in tests/test_method_02_vae.py; stdlib only.
        n = 8
        raw = root / "MNIST" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        images = (bytes(range(256)) * ((n * 28 * 28) // 256 + 1))[:n * 28 * 28]
        labels = bytes(i % 10 for i in range(n))
        for name, header, body in (
                ("train-images-idx3-ubyte",
                 struct.pack(">IIII", 2051, n, 28, 28), images),
                ("train-labels-idx1-ubyte",
                 struct.pack(">II", 2049, n), labels),
                ("t10k-images-idx3-ubyte",
                 struct.pack(">IIII", 2051, n, 28, 28), images),
                ("t10k-labels-idx1-ubyte",
                 struct.pack(">II", 2049, n), labels)):
            (raw / name).write_bytes(header + body)
            with gzip.open(raw / f"{name}.gz", "wb") as f:
                f.write(header + body)
        return root
    raise ValueError(f"real_run_smoke: unknown data_shape {shape!r}")
