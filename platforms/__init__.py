"""Backend resolution. **The core does not know any backend by name.**

Backends are looked up dynamically. Keeping a hard-coded list here would mean
the core knows which machines exist, which is exactly what we are avoiding.

    from platforms import load_backend
    backend = load_backend(name)     # the caller supplies the name

``tests/test_platform_isolation.py`` enforces this separation.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .base import Backend, JobResult, JobSpec

__all__ = ["Backend", "JobSpec", "JobResult", "load_backend",
           "available_backends"]


def available_backends() -> list[str]:
    """Every directory under ``platforms/`` that provides ``backend.py``.

    There is no table to maintain. **Drop one in and it is found.**
    """
    here = Path(__file__).resolve().parent
    return sorted(p.parent.name for p in here.glob("*/backend.py"))


def load_backend(name: str):
    """Import a backend module by name.

    When it is missing, **say what is available**. "Not found" on its own
    leaves the caller with nothing to act on.
    """
    if name not in available_backends():
        raise ValueError(
            f"no such execution backend: {name!r}. "
            f"available: {available_backends()}")
    return importlib.import_module(f"{__name__}.{name}.backend")
