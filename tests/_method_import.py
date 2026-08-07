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
from collections import Counter
from pathlib import Path

_METHODS_ROOT = Path(__file__).resolve().parent.parent / "methods"
_SHARED_NAMES_CACHE: "tuple[str, ...] | None" = None


def _shared_names() -> "tuple[str, ...]":
    """Top-level package names that more than one method defines -- `data`,
    `models`, `nce`, `adapter`, and whatever a future method shares next.

    **Discovered by scanning `methods/`, not listed.** A hand-kept list was the
    original shape here (`data`, `models`), and it silently missed `nce` the
    moment a second method (`12_cmc`, alongside `10_inst_disc`) defined one:
    `sys.modules["nce"]` then kept the first method's copy and the second's
    `from nce import ...` resolved against the wrong one. A name shared by two
    methods is exactly what must be purged, so the set is computed from the tree
    rather than remembered."""
    global _SHARED_NAMES_CACHE
    if _SHARED_NAMES_CACHE is None:
        counts: Counter = Counter()
        if _METHODS_ROOT.is_dir():
            for method_dir in sorted(_METHODS_ROOT.iterdir()):
                if not method_dir.is_dir():
                    continue
                for child in method_dir.iterdir():
                    if child.is_dir() and (child / "__init__.py").is_file():
                        counts[child.name] += 1
        _SHARED_NAMES_CACHE = tuple(sorted(n for n, c in counts.items()
                                           if c >= 2))
    return _SHARED_NAMES_CACHE


def _purge(method: Path) -> None:
    for name in _shared_names():
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
