"""Import a method's modules without letting two methods collide.

Every method has its own `data/` and `models/` packages, by design -- the
adapter runs with the method directory as the working directory, so inside a
run there is only ever one of each. **The test suite is the one place where
two are present at once**, and `sys.modules` keeps only the first: method 1's
trainer then does `from data.context_dataset_official import ...` and gets
method 2's `data`.

It surfaced only when the second method arrived, and only when the whole
suite ran -- each file passed on its own. That is the argument for running
everything together rather than per-file.

The rule lives here once, rather than in each method's test file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Package names more than one method defines. Anything here is dropped before
# a load, so the next import resolves against the method being loaded.
SHARED_NAMES = ("data", "models")


def _purge(method: Path) -> None:
    for name in SHARED_NAMES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        origin = getattr(mod, "__file__", "") or ""
        if not origin.startswith(str(method)):
            del sys.modules[name]
            for key in [k for k in sys.modules if k.startswith(f"{name}.")]:
                del sys.modules[key]


def load_from(method: Path, name: str, path: Path):
    """Import `path` with `method` importable, and nothing else's `data`."""
    method = Path(method)
    while str(method) in sys.path:
        sys.path.remove(str(method))
    sys.path.insert(0, str(method))
    _purge(method)

    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", "") == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod
